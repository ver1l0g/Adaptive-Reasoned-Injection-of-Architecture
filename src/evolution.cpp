#include "evolution.h"
#include "constants.h"
#include "logger.h"
#include "subgraph_library.h"
#include "serialize.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>

namespace aria {

// ============================================================================
// Dataset helpers
// ============================================================================

void Dataset::split(Dataset& train, Dataset& val, double val_fraction, unsigned int seed, bool shuffle) const {
    if (samples.empty()) {
        train.samples.clear();
        val.samples.clear();
        return;
    }

    std::vector<Graph::SampleIODesc> partitioned = samples;
    if (shuffle) {
        std::mt19937 rng(seed);
        std::shuffle(partitioned.begin(), partitioned.end(), rng);
    }
    // When shuffle=false, preserve row order so the tail is validation 鈥?    // required for sequence/recurrence datasets where order is temporal.

    size_t val_count = static_cast<size_t>(std::round(partitioned.size() * val_fraction));
    if (val_count == 0 && !partitioned.empty()) val_count = 1;
    if (val_count >= partitioned.size()) val_count = partitioned.size() - 1;

    train.samples.assign(partitioned.begin(), partitioned.end() - val_count);
    val.samples.assign(partitioned.end() - val_count, partitioned.end());
}

Dataset load_csv_dataset(const std::string& filepath, int input_cols,
                         const std::vector<uint64_t>& input_ids,
                         const std::vector<uint64_t>& output_ids,
                         bool header) {
    Dataset dataset;
    std::ifstream file(filepath);
    if (!file.is_open()) return dataset;

    std::string line;

    // If caller didn't request header skip, auto-detect: peek at first line
    // and check whether the first comma-separated token parses as a number.
    // If not, treat it as a header row and skip it.
    if (!header) {
        std::streampos pos = file.tellg();
        if (pos == std::streampos(-1)) pos = file.tellg();
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            bool first_token_numeric = false;
            if (std::getline(ss, token, ',')) {
                try {
                    size_t consumed = 0;
                    std::stod(token, &consumed);
                    // Require the whole token to be consumed (no trailing junk)
                    first_token_numeric = (consumed == token.size());
                } catch (...) {
                    first_token_numeric = false;
                }
            }
            if (!first_token_numeric) {
                header = true;  // skip the just-read line
            } else {
                file.clear();
                file.seekg(pos);  // rewind so the line gets parsed below
            }
        }
    }

    if (header && std::getline(file, line)) {
        // skip header row
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<Value> row;
        while (std::getline(ss, token, ',')) {
            row.push_back(std::stod(token));
        }

        if (row.size() < static_cast<size_t>(input_cols + static_cast<int>(output_ids.size())))
            continue;

        Graph::SampleIODesc sample;
        for (int i = 0; i < input_cols && i < static_cast<int>(input_ids.size()); ++i) {
            sample.inputs[input_ids[i]] = row[i];
        }
        for (size_t j = 0; j < output_ids.size(); ++j) {
            sample.targets[output_ids[j]] = row[input_cols + j];
        }
        dataset.samples.push_back(std::move(sample));
    }

    return dataset;
}

// ============================================================================
// EvolutionEngine constructor
// ============================================================================

EvolutionEngine::EvolutionEngine(std::unique_ptr<Graph> graph,
                                 Dataset training_data,
                                 Dataset validation_data,
                                 const Config& cfg)
    : graph_(std::move(graph))
    , training_data_(std::move(training_data))
    , validation_data_(std::move(validation_data))
    , cfg_(cfg)
{
    // Seed every RNG path. cfg_.seed == 0 preserves the old non-deterministic
    // behavior (random_device). A nonzero seed makes the whole run
    // reproducible: rng_ drives attribution-seed generation and any future
    // engine-level draws; std::srand drives NeuronNode::set_input_count's
    // Xavier weight init (which uses std::rand). Graph::train has no RNG, and
    // all other mt19937 instances derive their seeds from parameters that
    // ultimately trace back to rng_.
    if (cfg_.seed != 0) {
        rng_.seed(cfg_.seed);
        std::srand(static_cast<unsigned>(cfg_.seed));
    } else {
        rng_.seed(std::random_device{}());
    }
    ensure_minimal_architecture();
}

// ============================================================================
// ensure_minimal_architecture 鈥?build starter graph if empty
// ============================================================================
void EvolutionEngine::ensure_minimal_architecture() {
    if (graph_->node_count() > 0) {
        // Non-empty graph (loaded via --load-graph): rebuild the data-ID
        // mappings from node names so train/eval route inputs correctly.
        // The seeding path names nodes "input_<data_id>"/"output_<data_id>".
        for (const auto& n : graph_->get_nodes()) {
            const std::string& nm = n->get_name();
            if (nm.rfind("input_", 0) == 0) {
                try {
                    uint64_t data_id = std::stoull(nm.substr(6));
                    input_data_to_graph_[data_id] = n->get_id();
                } catch (...) {}
            } else if (nm.rfind("output_", 0) == 0) {
                try {
                    uint64_t data_id = std::stoull(nm.substr(7));
                    output_data_to_graph_[data_id] = n->get_id();
                } catch (...) {}
            }
        }
        return;
    }

    // Collect all unique INPUT and OUTPUT node IDs from the training data
    std::unordered_set<uint64_t> input_ids_set;
    std::unordered_set<uint64_t> output_ids_set;
    for (const auto& sample : training_data_.samples) {
        for (const auto& kv : sample.inputs)  input_ids_set.insert(kv.first);
        for (const auto& kv : sample.targets) output_ids_set.insert(kv.first);
    }

    if (input_ids_set.empty() || output_ids_set.empty()) return;

    // Create INPUT nodes
    std::vector<uint64_t> input_node_ids;
    for (uint64_t id : input_ids_set) {
        uint64_t nid = graph_->add_node(NodeType::INPUT, "input_" + std::to_string(id));
        input_node_ids.push_back(nid);
        input_data_to_graph_[id] = nid;
    }

    // Create OUTPUT nodes
    std::vector<uint64_t> output_node_ids;
    for (uint64_t id : output_ids_set) {
        uint64_t nid = graph_->add_node(NodeType::OUTPUT, "output_" + std::to_string(id));
        output_node_ids.push_back(nid);
        output_data_to_graph_[id] = nid;
    }

    // For each output, build INPUT 鈫?[NEURON|LINEAR] 鈫?OUTPUT chain.
    // Use LINEAR (identity, no tanh) for BCE loss 鈥?avoids double-saturation
    // (tanh + sigmoid) that kills gradients on high-dimensional classification.
    // Use NEURON (tanh) for MSE loss 鈥?the nonlinearity is needed for
    // expressive regression on bounded targets.
    NodeType starter_type = (cfg_.loss_type == Graph::LossType::BCE
                             || cfg_.loss_type == Graph::LossType::SOFTMAX_CE)
                            ? NodeType::LINEAR
                            : NodeType::NEURON;
    for (auto out_id : output_node_ids) {
        uint64_t neuron_id = graph_->add_node(starter_type,
                                              starter_type == NodeType::LINEAR
                                                  ? "starter_linear" : "starter_neuron");

        // Connect each INPUT to the node
        for (size_t j = 0; j < input_node_ids.size(); ++j) {
            graph_->add_connection(input_node_ids[j], 0, neuron_id, j);
        }

        // Node 鈫?OUTPUT
        graph_->add_connection(neuron_id, 0, out_id, 0);
    }
}

// ============================================================================
// evaluate_loss 鈥?forward pass all samples, compute average loss
// ============================================================================
double EvolutionEngine::evaluate_loss(const Dataset& data) {
    if (data.samples.empty()) return 0.0;

    blackboard_registry_.clear();
    graph_->reset_recurrent_state();

    const auto& samples = data.samples;
    int ns = static_cast<int>(samples.size());

    int num_threads = (config::EVOLUTION_PARALLEL && !graph_->has_temporal_state())
        ? std::min(config::EVOLUTION_NUM_THREADS, ns)
        : 1;

    // Sequential fallback for tiny datasets
    if (num_threads <= 1) {
        double total_loss = 0.0;
        int sample_count = 0;
        for (const auto& sample : samples) {
            for (const auto& kv : sample.inputs) {
                auto map_it = input_data_to_graph_.find(kv.first);
                if (map_it != input_data_to_graph_.end()) {
                    graph_->set_input_value(map_it->second, kv.second);
                }
            }
            graph_->execute();
            for (const auto& node : graph_->get_nodes()) {
                // Include INPUT nodes too: their "output" is the raw feature
                // value, which is exactly what search_blackboard() needs to
                // correlate against the target and what MULTIPLY_INJECTION's
                // input-source preference iterates over. Previously INPUTs
                // were skipped, so for a fresh INPUT鈫扤EURON鈫扥UTPUT graph the
                // blackboard held only the bottleneck NEURON (then excluded
                // as the failing node), leaving search_blackboard() empty and
                // MULTIPLY_INJECTION never emitted.
                  Value out = graph_->get_any_node_output(node->get_id());
                  blackboard_registry_[node->get_id()].push_back(out);
              }
              if (cfg_.loss_type == Graph::LossType::SOFTMAX_CE) {
                  total_loss += sample_softmax_ce(sample.targets);
              } else
              for (const auto& kv : sample.targets) {
                  auto map_it = output_data_to_graph_.find(kv.first);
                  if (map_it == output_data_to_graph_.end()) continue;
                  Value pred   = graph_->get_output_value(map_it->second);
                  Value target = kv.second;
                  Value diff   = pred - target;
                  if (cfg_.loss_type == Graph::LossType::BCE) {
                      Value sig_pred = 1.0 / (1.0 + std::exp(-pred));
                      Value eps = config::BCE_LOG_CLAMP_EPSILON;
                      Value clamped = std::max(eps, std::min(1.0 - eps, sig_pred));
                      total_loss += -(target * std::log(clamped) + (1.0 - target) * std::log(1.0 - clamped));
                  } else {
                      total_loss += diff * diff;
                  }
              }
              sample_count++;
          }
          return sample_count > 0 ? total_loss / sample_count : 0.0;
      }

    // ---- Parallel: split samples into chunks, clone graph per thread ----
    int chunk_size = (ns + num_threads - 1) / num_threads;

    struct ThreadResult {
        double loss = 0.0;
        int count = 0;
        std::unordered_map<uint64_t, std::vector<Value>> blackboard;
    };
    std::vector<ThreadResult> results(num_threads);
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int start = t * chunk_size;
            int end   = std::min(start + chunk_size, ns);
            if (start >= end) return;

            auto local_graph = graph_->clone();
            auto& tr = results[t];
            auto& bb = tr.blackboard;

            for (int i = start; i < end; ++i) {
                const auto& sample = samples[i];

                for (const auto& kv : sample.inputs) {
                    auto map_it = input_data_to_graph_.find(kv.first);
                    if (map_it != input_data_to_graph_.end()) {
                        local_graph->set_input_value(map_it->second, kv.second);
                    }
                }

                local_graph->execute();

                for (const auto& node : local_graph->get_nodes()) {
                    // Include INPUT nodes (see evaluate_loss sequential branch):
                    // their raw feature values are needed by search_blackboard()
                    // and MULTIPLY source selection.
                    Value out = local_graph->get_any_node_output(node->get_id());
                    bb[node->get_id()].push_back(out);
                }

                for (const auto& kv : sample.targets) {
                    auto map_it = output_data_to_graph_.find(kv.first);
                    if (map_it == output_data_to_graph_.end()) continue;
                    Value pred   = local_graph->get_output_value(map_it->second);
                    Value target = kv.second;
                    Value diff   = pred - target;

                    if (cfg_.loss_type == Graph::LossType::SOFTMAX_CE) {
                        // Joint softmax-CE over all outputs of this sample
                        // (local_graph, not graph_: this is the threaded path).
                        // Computed ONCE per sample (first target iteration).
                        bool first_target = true;
                        if (!first_target) break;
                        std::vector<std::pair<Value,Value>> sm_pairs;
                        sm_pairs.reserve(sample.targets.size());
                        for (const auto& kv2 : sample.targets) {
                            auto mi2 = output_data_to_graph_.find(kv2.first);
                            if (mi2 == output_data_to_graph_.end()) continue;
                            sm_pairs.emplace_back(
                                local_graph->get_output_value(mi2->second), kv2.second);
                        }
                        if (sm_pairs.size() >= 2) {
                            double mx = -1e300;
                            for (auto& pr : sm_pairs)
                                mx = std::max(mx, static_cast<double>(pr.first));
                            double den = 0.0;
                            for (auto& pr : sm_pairs)
                                den += std::exp(static_cast<double>(pr.first) - mx);
                            double lse = mx + std::log(den);
                            for (auto& pr : sm_pairs) {
                                if (pr.second > 0.5) {
                                    tr.loss += lse - static_cast<double>(pr.first);
                                    break;
                                }
                            }
                        }
                        break;   // once per sample, not per target
                    } else if (cfg_.loss_type == Graph::LossType::BCE) {
                        Value sig_pred = 1.0 / (1.0 + std::exp(-pred));
                        Value eps = config::BCE_LOG_CLAMP_EPSILON;
                        Value clamped = std::max(eps, std::min(1.0 - eps, sig_pred));
                        tr.loss += -(target * std::log(clamped) + (1.0 - target) * std::log(1.0 - clamped));
                    } else {
                        tr.loss += diff * diff;
                    }
                }
                tr.count++;
            }
        });
    }

    for (auto& th : threads) th.join();

    // Merge results
    double total_loss = 0.0;
    int total_count = 0;
    for (int t = 0; t < num_threads; ++t) {
        total_loss += results[t].loss;
        total_count += results[t].count;
        for (const auto& kv : results[t].blackboard) {
            auto& dest = blackboard_registry_[kv.first];
            dest.insert(dest.end(), kv.second.begin(), kv.second.end());
        }
    }

    return total_count > 0 ? total_loss / total_count : 0.0;
}

// ============================================================================
// train_parameters 鈥?SGD training with plateau detection
// ============================================================================
EvolutionEngine::TrainResult EvolutionEngine::train_parameters() {
    double loss_before = evaluate_loss(training_data_);

    Graph::TrainConfig train_cfg;
    train_cfg.epochs         = cfg_.sgd_epochs_per_phase;
    train_cfg.learning_rate  = cfg_.sgd_learning_rate * divergence_lr_mult_;

    // Compound grace-period LR attenuation.
    //
    // When a compound hypothesis (MULTIPLY_NEURON, TANH_SERIES, MULTIPLY3_NEURON)
    // was recently committed, epochs_since_structural_ is negative (set to
    // -COMPOUND_COMMIT_GRACE_EPOCHS at commit time, incremented each epoch).
    // During this window the freshly-added zero-init chains are still growing;
    // applying the full learning rate disrupts both the new chains and the
    // trained graph. Observed failure mode (bench3): TANH_SERIES committed at
    // loss=0.0015, the next SGD epoch at full LR pushed loss to 0.623 (417x),
    // and subsequent IFELSE commits locked in the regressed state.
    //
    // Shadow validation already applies SHADOW_COMPOUND_LR_MULTIPLIER for the
    // same reason; mirror it here so the post-commit main-loop SGD uses the
    // same cautious LR.
    if (epochs_since_structural_ < 0) {
        // MULTI_LAYER_STACK / PATCH_POOLING train fresh sub-networks 鈥?need
        // a milder reduction than the 0.2x used for zero-init residual chains.
        double lr_mult = (last_committed_hyp_type_
                          == static_cast<int>(Hypothesis::MULTI_LAYER_STACK)
                          || last_committed_hyp_type_
                             == static_cast<int>(Hypothesis::PATCH_POOLING))
                       ? 0.5 : config::SHADOW_COMPOUND_LR_MULTIPLIER;
        train_cfg.learning_rate *= lr_mult;
        train_cfg.early_stop_patience = 20;  // give zero-init chains time to grow
    }

    train_cfg.gradient_clip  = cfg_.sgd_gradient_clip;
    train_cfg.momentum       = cfg_.sgd_momentum;
    train_cfg.weight_decay   = cfg_.sgd_weight_decay;
    train_cfg.loss_type      = cfg_.loss_type;
    train_cfg.batch_size     = (cfg_.sequence_mode || training_data_.samples.size() < 500)
                               ? 0 : config::DEFAULT_SGD_BATCH_SIZE;
    train_cfg.input_data_to_graph  = input_data_to_graph_;
    train_cfg.output_data_to_graph = output_data_to_graph_;

    graph_->train(training_data_.samples, train_cfg);
    stats_.sgd_epochs_run += cfg_.sgd_epochs_per_phase;

    double loss_after = evaluate_loss(training_data_);

    // Plateau detection 鈥?use relative improvement rate
    // Relative: (before - after) / max(|before|, small) > threshold
    // This avoids false plateau detection when loss is very small
    // (e.g., mse=0.02 where absolute delta never reaches 1e-4).
    double relative_improvement = 0.0;
    {
        double denom = std::max(std::abs(loss_before), config::PLATEAU_RELATIVE_DENOM_FLOOR);
        relative_improvement = (loss_before - loss_after) / denom;
    }
    if (loss_after < best_phase2_loss_ - cfg_.plateau_min_improvement) {
        best_phase2_loss_ = loss_after;
        plateau_counter_ = 0;
        Logger::verbose("SGD improved: loss " + std::to_string(loss_before)
                        + " -> " + std::to_string(loss_after)
                        + " (delta=" + std::to_string(loss_before - loss_after) + ")");
    } else if (relative_improvement > cfg_.plateau_min_improvement) {
        // Relative improvement detected 鈥?still making progress
        plateau_counter_ = 0;
        Logger::verbose("SGD relative improvement: " + std::to_string(relative_improvement)
                        + " (loss " + std::to_string(loss_before)
                        + " -> " + std::to_string(loss_after) + ")");
    } else {
        plateau_counter_++;
        Logger::verbose("SGD plateau counter=" + std::to_string(plateau_counter_)
                        + "/" + std::to_string(cfg_.plateau_patience)
                        + " (best_phase2=" + std::to_string(best_phase2_loss_) + ")");
    }

    if (loss_after < best_overall_loss_) {
        best_overall_loss_ = loss_after;
        epochs_since_best_ = 0;  // reset outer-loop early-stop counter
        // NOTE: no snapshot here 鈥?the deployed snapshot is validation-
        // selected once per epoch in evolve(). Train-based tracking only
        // drives early stopping and the divergence safety net.
        Logger::info("New best overall loss: " + std::to_string(loss_after));
    }

    return {loss_after < loss_before, loss_before, loss_after};
}

// ============================================================================
// evolve 鈥?main 7-phase evolution loop
// ============================================================================
double EvolutionEngine::evolve(std::function<void(int,double,const std::string&)> progress_cb) {
    // Initial evaluation
    double current_loss = evaluate_loss(training_data_);
    stats_.initial_loss  = current_loss;
    best_phase2_loss_    = current_loss;
    best_overall_loss_   = current_loss;
    best_graph_snapshot_ = graph_->clone();

    if (progress_cb) progress_cb(0, current_loss, "init");

    for (int epoch = 1; epoch <= cfg_.max_epochs; ++epoch) {
        stats_.total_epochs = epoch;
        epochs_since_structural_++;
        ++epochs_since_best_;
        if (structural_cooldown_ > 0) --structural_cooldown_;
        // Slow recovery of divergence-reduced LR (5%/epoch, capped at 1.0)
        if (divergence_lr_mult_ < 1.0) {
            divergence_lr_mult_ = std::min(1.0, divergence_lr_mult_ * 1.05);
        }

        // ---- Phase 1+2: evaluate and SGD micro-evolution ----
        auto tr = train_parameters();
        if (progress_cb) progress_cb(epoch, tr.loss_before, "eval");
        current_loss = tr.loss_after;
        if (progress_cb) progress_cb(epoch, current_loss, "sgd");

        // ---- Global divergence safety net (patience-gated) ----
        // If loss has been > EVOLVE_DIVERGENCE_FACTOR x best for
        // EVOLVE_DIVERGENCE_PATIENCE consecutive epochs, restore the best
        // snapshot. The patience window allows temporary spikes from
        // structural commits to recover via SGD before intervening.
        if (best_overall_loss_ > 1e-9 &&
            current_loss > best_overall_loss_ * config::EVOLVE_DIVERGENCE_FACTOR) {
            if (++divergence_counter_ >= config::EVOLVE_DIVERGENCE_PATIENCE) {
                Logger::info("Divergence safety: loss=" + std::to_string(current_loss)
                             + " > " + std::to_string(config::EVOLVE_DIVERGENCE_FACTOR)
                             + "x best=" + std::to_string(best_overall_loss_)
                             + " for " + std::to_string(config::EVOLVE_DIVERGENCE_PATIENCE)
                             + " epochs; restoring best snapshot");
                if (best_graph_snapshot_) {
                    graph_ = best_graph_snapshot_->clone();
                    current_loss = evaluate_loss(training_data_);
                    best_phase2_loss_ = current_loss;
                    plateau_counter_ = 0;
                    divergence_counter_ = 0;
                    // Halve LR: re-entering the same basin at full LR just
                    // re-diverges (restore loop observed on I.32.8).
                    divergence_lr_mult_ *= 0.5;
                    if (divergence_lr_mult_ < 0.01) divergence_lr_mult_ = 0.01;
                    Logger::info("Divergence LR reduced to "
                                + std::to_string(divergence_lr_mult_) + "x");
                }
                divergence_counter_ = 0;
            }
        } else {
            divergence_counter_ = 0;
        }

        // ---- Handle plateau 鈫?Phases 3-5 ----
        // Scale plateau patience with input dimensionality: high-dimensional
        // tasks (e.g. 64-pixel MNIST) need more SGD epochs to converge the
        // base linear model before structural search interrupts.
        int effective_patience = cfg_.plateau_patience;
        size_t num_inputs = input_data_to_graph_.size();
        if ((int)num_inputs > config::HIGH_DIM_INPUT_THRESHOLD) {
            int extra = static_cast<int>((num_inputs - config::HIGH_DIM_INPUT_THRESHOLD)
                                        * config::HIGH_DIM_PATIENCE_PER_INPUT);
            effective_patience += extra;
        }

        bool force_structural = (cfg_.force_structural_every > 0
            && epochs_since_structural_ >= cfg_.force_structural_every);
        bool structural_trigger = (plateau_counter_ >= effective_patience || force_structural);

        // PERF/correctness: converged-loss guard. When the training loss is
        // already at noise level (e.g. 1e-6), NO structural change can beat
        // the scaled commit gate — cycling is pure waste (I.47.23: 60 cycles
        // at loss 8e-6, each ~65s). Skip unless force_structural (the periodic
        // probe still fires, bounded by inter-cycle gap).
        if (structural_trigger && !force_structural
            && best_overall_loss_ < config::CONVERGED_LOSS_FLOOR) {
            structural_trigger = false;
            Logger::verbose("Structural search skipped: loss "
                          + std::to_string(best_overall_loss_)
                          + " below converged floor "
                          + std::to_string(config::CONVERGED_LOSS_FLOOR));
        }

        // Structural cooldown: after a run of failed structural cycles, back
        // off and let SGD settle on the current architecture. Repeatedly
        // firing structural every epoch when every candidate gets rejected is
        // what produced the 700+ failed-commit thrash on hard tasks.
        if (structural_trigger && structural_cooldown_ > 0) {
            structural_trigger = false;
            Logger::info("Structural search in cooldown (" + std::to_string(structural_cooldown_)
                        + " epochs left) at epoch " + std::to_string(epoch)
                        + " 鈥?letting SGD settle");
        }

        if (structural_trigger) {
            bool structural_commit_succeeded = false;
            int  committed_hyp_type = static_cast<int>(Hypothesis::NONE);
            if (force_structural) {
                Logger::info("Forced structural trigger at epoch " + std::to_string(epoch)
                            + " (epochs since last structural: "
                            + std::to_string(epochs_since_structural_) + ")");
            }
            Logger::info("Plateau triggered at epoch " + std::to_string(epoch)
                        + " (counter=" + std::to_string(plateau_counter_)
                        + ", loss=" + std::to_string(current_loss) + ")");

            // Phase 3: Diagnose failures
            std::vector<FailureDiagnosis> diagnoses = diagnose();
            Logger::verbose("diagnose() returned " + std::to_string(diagnoses.size()) + " failure(s)");

            // Last-resort growth fallback: only fires when diagnose() returned
            // nothing at all (i.e., error attribution produced zero candidates,
            // which only happens for pathological graphs with no perturbable
            // interior nodes). The much more common "all blames positive"
            // plateau case is already handled inside diagnose() via the
            // structural-inability path, which surfaces the highest-blame node
            // with its real blame value.
            if (diagnoses.empty()) {
                for (const auto& node : graph_->get_nodes()) {
                    auto t = node->get_type();
                    if (t != NodeType::INPUT && t != NodeType::OUTPUT
                        && t != NodeType::SINK && node->get_num_outputs() > 0) {
                        FailureDiagnosis growth;
                        growth.failing_node       = node->get_id();
                        growth.mean_blame         = 0.0;
                        growth.is_constant_output = false;
                        growth.has_dead_branch    = false;
                        std::vector<uint64_t> out_ids;
                        for (const auto& n : graph_->get_nodes()) {
                            if (n->get_type() == NodeType::OUTPUT)
                                out_ids.push_back(n->get_id());
                        }
                        for (auto oid : out_ids) compute_targets(growth, oid);
                        diagnoses.push_back(std::move(growth));
                        Logger::info("diagnose() empty 鈥?injecting growth diagnosis for node "
                                    + std::to_string(node->get_id()));
                        break;
                    }
                }
            }

            int fixes_attempted = 0;
            for (auto& diag : diagnoses) {
                if (fixes_attempted >= cfg_.max_failures_to_fix) break;

                Logger::verbose("Diagnosing node " + std::to_string(diag.failing_node)
                               + " blame=" + std::to_string(diag.mean_blame)
                               + (diag.is_constant_output ? " [constant]" : "")
                               + (diag.has_dead_branch ? " [dead_branch]" : ""));

                // Phase 4: Search the Blackboard for context
                std::vector<BlackboardSignal> signals = search_blackboard(diag);
                if (!signals.empty()) {
                    Logger::verbose("Top blackboard signal: node="
                                   + std::to_string(signals[0].node_id)
                                   + " corr=" + std::to_string(signals[0].correlation));
                }

                // Phase 4b: Local Check 鈥?classify the failure type
                FailureType ftype = classify_failure(diag);
                const char* ftype_names[] = {"UNKNOWN", "LINEAR_OFFSET", "BOOLEAN_BOUNDARY", "NON_LINEAR_CURVE"};
                Logger::decision("Classify failure", std::string("node=") + std::to_string(diag.failing_node)
                                + " type=" + ftype_names[static_cast<int>(ftype)]);

                // Phase 4c: Complexity profile of the residual at this bottleneck.
                // Cheap (O(N路F虏)) fingerprint of the residual shape. Used by
                // generate_candidates to emit compound hypotheses (e.g.
                // MULTIPLY+NEURON_TANH for sin(xy)) when the signature matches.
                ComplexityProfile profile = compute_complexity_profile(diag);
                Logger::info(format_profile(profile));

                // Phase 5: Generate scored, ranked candidate hypotheses
                std::vector<Hypothesis> candidates = generate_candidates(diag, signals, ftype, profile);

                // === Parallel candidate evaluation ===
                // Pre-create shadows for all viable candidates, then validate
                // them concurrently. Pick the lowest-val_loss acceptable
                // shadow and commit it. This collapses the wall-clock cost
                // of failed candidates 鈥?previously N rejected candidates
                // each cost ~50 SGD epochs sequentially; now they overlap.
                const char* hyp_names[] = {"NONE", "IFELSE_BOUNDARY_SPLIT", "NEURON_TANH_INJECTION", "CONTEXT_WIRE", "MULTIPLY_INJECTION", "BOOLEAN_COMPOSE", "COMPOUND_MULTIPLY_NEURON", "COMPOUND_TANH_SERIES", "COMPOUND_MULTIPLY3_NEURON", "COMPOUND_MULTIPLY_ABS", "RECURRENT_SELF_WIRE", "SIN_INJECTION", "DEEP_INSERTION", "RECURRENT_XOR", "MULTI_LAYER_STACK", "PATCH_POOLING", "PARITY_TREE", "DIVIDE_INJECTION", "COMPOUND_SIN_PRODUCT", "COMPOUND_DIVIDE_PRODUCT", "RECURRENT_MULTI_TAP", "MUX_INJECTION", "DELAY_LINE", "IFELSE_PRESERVE", "EMBED_TRUNK", "ATTENTION_MIX"};

                struct ShadowSpec {
                    int         rank;
                    int         type;  // Hypothesis::Type as int
                    bool        evidence = false;  // structural-evidence flag
                    std::unique_ptr<Graph> shadow;
                };
                std::vector<ShadowSpec> specs;

                // M1.2 architecture recall (once per run): inject the PRIOR
                // solved graph for this task as a rank-(−1) shadow candidate.
                // It competes in the same validation race — if the prior
                // solution still wins, it commits and the run skips
                // re-discovery; a stale prior simply loses. Requires the
                // prior graph to have compatible arity (the load throws
                // otherwise and we skip gracefully).
                if (!recall_attempted_ && !recall_graph_path_.empty()) {
                    recall_attempted_ = true;
                    try {
                        auto prior = std::make_unique<Graph>();
                        load_graph_from_file(*prior, recall_graph_path_);
                        // Arity check: the prior must accept exactly this
                        // task's input/output count.
                        size_t pin = 0, pout = 0;
                        for (const auto& n : prior->get_nodes()) {
                            if (n->get_type() == NodeType::INPUT) ++pin;
                            else if (n->get_type() == NodeType::OUTPUT) ++pout;
                        }
                        size_t cin = 0, cout = 0;
                        for (const auto& n : graph_->get_nodes()) {
                            if (n->get_type() == NodeType::INPUT) ++cin;
                            else if (n->get_type() == NodeType::OUTPUT) ++cout;
                        }
                        if (pin == cin && pout == cout && cin > 0) {
                            specs.push_back({-1, -1, false, std::move(prior)});
                            Logger::info("  [RECALL] prior graph injected as candidate ("
                                        + std::to_string(pin) + " inputs / "
                                        + std::to_string(pout) + " outputs, from "
                                        + recall_graph_path_ + ")");
                        } else {
                            Logger::info("  [RECALL] skipped: arity mismatch (prior "
                                        + std::to_string(pin) + "/" + std::to_string(pout)
                                        + " vs current " + std::to_string(cin) + "/"
                                        + std::to_string(cout) + ")");
                        }
                    } catch (...) {
                        Logger::info("  [RECALL] skipped: load failed ("
                                    + recall_graph_path_ + ")");
                    }
                }

                int hyp_idx = 0;
                for (auto& hyp : candidates) {
                    if (hyp.type == Hypothesis::NONE) continue;
                    // Skip suppressed hypothesis types (degenerate-loop prevention)
                    if (suppressed_hyp_types_.count(static_cast<int>(hyp.type))) {
                        Logger::verbose("  rank=" + std::to_string(hyp_idx)
                                      + " type=" + hyp_names[static_cast<int>(hyp.type)]
                                      + " SKIPPED (suppressed)");
                        hyp_idx++;
                        continue;
                    }
                    Logger::decision("Try hypothesis", std::string("rank=") + std::to_string(hyp_idx)
                                    + " type=" + hyp_names[static_cast<int>(hyp.type)]);
                    // PHASE LOG (narma30 debug): confirm DELAY_LINE reaches
                    // the routing stage at all, and why not if not.
                    if (hyp.type == Hypothesis::DELAY_LINE) {
                        Logger::info("  [DELAY_LINE-DBG] reached routing (rank="
                                    + std::to_string(hyp_idx) + ")");
                    }

                    std::unique_ptr<Graph> shadow = apply_shadow_routing(hyp, diag);
                    if (!shadow) {
                        Logger::verbose("  shadow routing failed for rank=" + std::to_string(hyp_idx)
                                      + " type=" + hyp_names[static_cast<int>(hyp.type)]);
                        hyp_idx++;
                        continue;
                    }

                    specs.push_back({hyp_idx, static_cast<int>(hyp.type),
                                     hyp.structural_evidence, std::move(shadow)});
                    hyp_idx++;
                    fixes_attempted++;
                    if (fixes_attempted >= cfg_.max_failures_to_fix) break;
                }

                if (specs.empty()) {
                    Logger::verbose("  no viable shadows to validate");
                } else {
                    // Compute baseline_val ONCE on current graph (cheap; avoids
                    // N redundant recomputations when validating N candidates).
                    double baseline_val = compute_validation_loss(*graph_);

                    // Launch one thread per candidate. Each thread trains its
                    // shadow and evaluates it on validation_data_. The shadow
                    // remains owned by specs[i].shadow (passed by reference);
                    // validate_shadow_only does NOT move it. After join, the
                    // winner is committed by moving from specs[winner_idx].shadow.
                    std::vector<ShadowValidationResult> results(specs.size());
                    std::vector<std::thread> threads;
                    threads.reserve(specs.size());

                    for (size_t i = 0; i < specs.size(); ++i) {
                        threads.emplace_back([&, i]() {
                            // CRASH GUARD: an uncaught exception in a thread
                            // calls std::terminate — instant silent process
                            // death (observed: w8-SCE died at MUX_INJECTION
                            // emission, no stderr). Catch everything; a
                            // throwing shadow becomes a rejected candidate
                            // instead of a killed run.
                            try {
                                results[i] = validate_shadow_only(
                                    specs[i].shadow,
                                    current_loss,
                                    baseline_val,
                                    specs[i].rank,
                                    specs[i].type,
                                    specs[i].evidence);
                            } catch (const std::exception& e) {
                                results[i].acceptable = false;
                                results[i].hyp_rank = specs[i].rank;
                                results[i].hyp_type = specs[i].type;
                                results[i].reject_reason =
                                    std::string("shadow validation THREW: ") + e.what();
                                Logger::info("  rank=" + std::to_string(specs[i].rank)
                                            + " shadow exception: " + e.what());
                            } catch (...) {
                                results[i].acceptable = false;
                                results[i].hyp_rank = specs[i].rank;
                                results[i].hyp_type = specs[i].type;
                                results[i].reject_reason = "shadow validation THREW (unknown)";
                                Logger::info("  rank=" + std::to_string(specs[i].rank)
                                            + " shadow exception (unknown type)");
                            }
                        });
                    }
                    for (auto& t : threads) t.join();

                    // Pick the winner using a rank-aware policy.
                    //
                    // Pure lowest-val_loss is too greedy 鈥?it picks marginally
                    // better rank-N hypotheses over proven rank-0/1 candidates,
                    // locking out productive search paths (e.g., T2.4 kept
                    // committing rank-2 NEURON_TANH for tiny gains while never
                    // finding BOOLEAN_COMPOSE).
                    //
                    // Rank-aware tie-breaking: when comparing two acceptable
                    // candidates, the lower-rank (higher-priority) one wins
                    // UNLESS the higher-rank one's val_loss is better by more
                    // than rank_bonus_per_step * rank_gap. This preserves
                    // exploration of proven paths while still allowing a
                    // clearly-better lower-priority candidate to win.
                    //
                    // Concretely: effective_loss[i] = val_loss[i]
                    //                                + rank_bonus * rank[i]
                    // Pick the candidate with the lowest effective_loss.
                    double rank_bonus = std::abs(current_loss)
                                      * config::COMMIT_RANK_BONUS_FRACTION;  // e.g. 1% of baseline per rank step

                    int winner_idx = -1;
                    double winner_effective = 0.0;
                    for (size_t i = 0; i < results.size(); ++i) {
                          if (!results[i].acceptable) {
                              stats_.failed_commits++;
                              // M5.7 diagnostics: evidence-candidate rejections
                              // are INFO-level — measured-structure commits
                              // failing is a signal, not noise (t22 arc).
                              if (specs[i].evidence) {
                                  Logger::info("  [EVIDENCE-REJECT] rank="
                                              + std::to_string(results[i].hyp_rank)
                                              + " type="
                                              + (results[i].hyp_type >= 0
                                                 ? hyp_names[results[i].hyp_type]
                                                 : std::string("RECALL"))
                                              + " val=" + std::to_string(results[i].val_loss)
                                              + " REJECT — " + results[i].reject_reason);
                              }
                              Logger::verbose("  rank=" + std::to_string(results[i].hyp_rank)
                                             + " REJECT — " + results[i].reject_reason);
                              // M1.1: record the failure for the failure
                              // library (cross-task negative experience).
                              if (current_cycle_fp_valid_) {
                                  FailureRecord fr;
                                  fr.fingerprint = current_cycle_fp_;
                                  fr.hyp_type = results[i].hyp_type;
                                  fr.val_delta = results[i].val_loss - baseline_val;
                                  fr.task = current_task_name_;
                                  session_failures_.push_back(std::move(fr));
                              }
                              continue;
                          }
                        double effective = results[i].val_loss
                                         + rank_bonus * static_cast<double>(results[i].hyp_rank);
                        // M5.7: measured-evidence candidates carry a race
                        // advantage (10 rank-steps) — their structure is
                        // label-space fact, not a statistical hypothesis;
                        // micro-gain alternatives shouldn't crowd them out.
                        if (specs[i].evidence) {
                            effective -= rank_bonus * 10.0;
                        }
                        if (winner_idx < 0 || effective < winner_effective) {
                            // Replace previous winner 鈥?release its shadow first
                            if (winner_idx >= 0) {
                                stats_.failed_commits++;
                                Logger::verbose("  rank=" + std::to_string(results[winner_idx].hyp_rank)
                                               + " superseded by rank=" + std::to_string(results[i].hyp_rank)
                                               + " (effective " + std::to_string(effective)
                                               + " < " + std::to_string(winner_effective) + ")");
                            }
                            winner_idx = static_cast<int>(i);
                            winner_effective = effective;
                        }
                    }

                    if (winner_idx >= 0) {
                        // Commit the winning shadow. validate_shadow_only
                        // received the shadow by reference and did NOT move
                        // it, so ownership remains in specs[winner_idx].shadow.
                        bool was_recall = (results[winner_idx].hyp_rank == -1);
                        // M1.3-v2 arc-price: PRE-commit output-residual
                        // fingerprint (computed on the OLD graph before the
                        // move — current_cycle_fp_ is only set inside the
                        // library-matching block, absent in isolated runs).
                        BehavioralFingerprint arc_pre_fp;
                        bool arc_have_pre = false;
                        {
                            Logger::info("  [ARC-DBG] pre-block enter");
                            try {
                                const size_t n_total = training_data_.samples.size();
                                const size_t stride = (n_total > 200) ? n_total / 200 : 1;
                                std::vector<std::vector<double>> Xr;
                                std::vector<double> yr;
                                for (size_t pi = 0; pi < n_total; pi += stride) {
                                    const auto& s = training_data_.samples[pi];
                                    for (const auto& kv : s.inputs) {
                                        auto git = input_data_to_graph_.find(kv.first);
                                        if (git != input_data_to_graph_.end()) {
                                            graph_->set_input_value(git->second, kv.second);
                                        }
                                    }
                                    graph_->execute();
                                    double rsum = 0;
                                    int rcnt = 0;
                                    for (const auto& kv : s.targets) {
                                        auto oit = output_data_to_graph_.find(kv.first);
                                        if (oit != output_data_to_graph_.end()) {
                                            double diff = graph_->get_output_value(oit->second)
                                                       - static_cast<double>(kv.second);
                                            rsum += diff * diff;
                                            ++rcnt;
                                        }
                                    }
                                    if (rcnt > 0) {
                                        yr.push_back(rsum / rcnt);
                                        std::vector<double> xr;
                                        for (const auto& kv : s.inputs) {
                                            xr.push_back(kv.second);
                                        }
                                        Xr.push_back(std::move(xr));
                                    }
                                    graph_->reset_recurrent_state();
                                }
                                if (yr.size() >= 16) {
                                    Logger::info("  [ARC-DBG] computing fingerprint");
                                    arc_pre_fp = compute_fingerprint(Xr, yr);
                                    Logger::info("  [ARC-DBG] fingerprint done");
                                    arc_have_pre = true;
                                }
                            } catch (...) {
                                Logger::info("  [ARC-DBG] pre-block exception");
                            }
                        }
                        graph_ = std::move(specs[winner_idx].shadow);
                        stats_.structural_changes++;
                        if (was_recall) {
                            // M1.2 recall commit: the swapped-in graph carries
                            // its own node-ID space. Rebuild the data-ID maps
                            // from input_N/output_M node names (same
                            // convention as --load-graph) or every subsequent
                            // set_input_value maps to stale IDs (access
                            // violation).
                            input_data_to_graph_.clear();
                            output_data_to_graph_.clear();
                            for (const auto& n : graph_->get_nodes()) {
                                const std::string& nm = n->get_name();
                                try {
                                    if (nm.rfind("input_", 0) == 0) {
                                        input_data_to_graph_[std::stoull(nm.substr(6))] = n->get_id();
                                    } else if (nm.rfind("output_", 0) == 0) {
                                        output_data_to_graph_[std::stoull(nm.substr(7))] = n->get_id();
                                    }
                                } catch (...) {}
                            }
                            Logger::info("  [RECALL] prior graph COMMITTED — maps rebuilt ("
                                        + std::to_string(input_data_to_graph_.size()) + " in / "
                                        + std::to_string(output_data_to_graph_.size()) + " out)");
                        }
                        current_loss = evaluate_loss(training_data_);
                        Logger::info("  COMMIT rank=" + std::to_string(results[winner_idx].hyp_rank)
                                    + " type=" + (results[winner_idx].hyp_type < 0
                                                  ? std::string("RECALL")
                                                  : hyp_names[results[winner_idx].hyp_type])
                                    + " val_loss=" + std::to_string(results[winner_idx].val_loss)
                                    + " train_loss=" + std::to_string(results[winner_idx].train_loss)
                                    + " new train_loss=" + std::to_string(current_loss));
                        structural_commit_succeeded = true;
                        committed_hyp_type = results[winner_idx].hyp_type;
                        // M1.3-v2 investment-arc pricing DIAGNOSTIC (post-
                        // commit): does this commit MOVE the output
                        // residual's fingerprint? Ladders (hetero3 sin
                        // harmonics, d2 mod-3) do — each rung shifts the
                        // residual's statistical character. Sprays
                        // (stripes20 TANH x14) don't. Log-only v1; pricing
                        // wired once the separation is verified.
                        if (arc_have_pre) {
                            try {
                                const size_t n_total = training_data_.samples.size();
                                const size_t stride = (n_total > 200) ? n_total / 200 : 1;
                                std::vector<std::vector<double>> Xr;
                                std::vector<double> yr;
                                for (size_t pi = 0; pi < n_total; pi += stride) {
                                    const auto& s = training_data_.samples[pi];
                                    for (const auto& kv : s.inputs) {
                                        auto git = input_data_to_graph_.find(kv.first);
                                        if (git != input_data_to_graph_.end()) {
                                            graph_->set_input_value(git->second, kv.second);
                                        }
                                    }
                                    graph_->execute();
                                    double rsum = 0;
                                    int rcnt = 0;
                                    for (const auto& kv : s.targets) {
                                        auto oit = output_data_to_graph_.find(kv.first);
                                        if (oit != output_data_to_graph_.end()) {
                                            double diff = graph_->get_output_value(oit->second)
                                                       - static_cast<double>(kv.second);
                                            rsum += diff * diff;
                                            ++rcnt;
                                        }
                                    }
                                    if (rcnt > 0) {
                                        yr.push_back(rsum / rcnt);
                                        std::vector<double> xr;
                                        for (const auto& kv : s.inputs) {
                                            xr.push_back(kv.second);
                                        }
                                        Xr.push_back(std::move(xr));
                                    }
                                    graph_->reset_recurrent_state();
                                }
                                if (yr.size() >= 16) {
                                    BehavioralFingerprint post_fp =
                                        compute_fingerprint(Xr, yr);
                                    double move = fingerprint_distance(arc_pre_fp, post_fp);
                                    Logger::info(
                                        "  [ARC-PRICE] family="
                                        + std::string(hyp_names[committed_hyp_type])
                                        + " fingerprint_move=" + std::to_string(move)
                                        + " (val_delta="
                                        + std::to_string(baseline_val - results[winner_idx].val_loss)
                                        + ")");
                                }
                            } catch (...) {
                                // diagnostic only — never block a commit
                            }
                        }
                    } else {
                        Logger::verbose("  all " + std::to_string(results.size()) + " candidates rejected");
                    }
                }
            }

            // Only reset plateau state if a commit succeeded. Otherwise the
            // best_phase2 baseline would drift upward every epoch (since
            // current_loss at plateau time is by definition worse than
            // best_phase2). Keeping the original baseline forces the next
            // plateau trigger to fire immediately, which means more aggressive
            // structural search 鈥?exactly what we want when the engine is
            // structurally stuck.
            if (structural_commit_succeeded) {
                best_phase2_loss_ = current_loss;
                plateau_counter_ = 0;
                epochs_since_structural_ = 0;
                // A structural commit is genuine progress (a new architecture),
                // even if it didn't immediately beat best_overall_loss_. Reset
                // the early-stop counter so bursty tasks aren't cut off mid-search.
                epochs_since_best_ = 0;
                // Compound grace period: the new MULTIPLY鈫扤EURON鈫扵ANH chain
                // starts with NEURON weight=0 (identity start). SGD needs many
                // epochs to grow the weight from 0 to a useful value. Without
                // this grace, plateau_patience=3 OR force_structural_every=20
                // fires too early and the engine attempts more structural
                // changes that all fail catastrophically.
                if (committed_hyp_type
                       == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY_NEURON)
                    || committed_hyp_type
                       == static_cast<int>(Hypothesis::COMPOUND_TANH_SERIES)
                     || committed_hyp_type
                        == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY3_NEURON)
                     || committed_hyp_type
                        == static_cast<int>(Hypothesis::MULTI_LAYER_STACK)
                     || committed_hyp_type
                        == static_cast<int>(Hypothesis::PATCH_POOLING)
                     || committed_hyp_type
                        == static_cast<int>(Hypothesis::COMPOUND_SIN_PRODUCT)
                     || committed_hyp_type
                        == static_cast<int>(Hypothesis::COMPOUND_DIVIDE_PRODUCT)
                     || committed_hyp_type
                        == static_cast<int>(Hypothesis::EMBED_TRUNK)) {
                    plateau_counter_ = -config::COMPOUND_COMMIT_GRACE_EPOCHS;
                    epochs_since_structural_ = -config::COMPOUND_COMMIT_GRACE_EPOCHS;
                    Logger::info("Compound grace period 鈥?" + std::to_string(config::COMPOUND_COMMIT_GRACE_EPOCHS)
                                + " extra epochs before next structural attempt");
                }
                Logger::info("Plateau state reset 鈥?best_phase2=" + std::to_string(best_phase2_loss_));
                Logger::info("Plateau state reset — best_phase2=" + std::to_string(best_phase2_loss_));
                consecutive_structural_failures_ = 0;

                // PERF: inter-cycle breathing room. A committed-but-marginal
                // change resets the failure counter, so back-to-back structural
                // cycles can still fire every epoch (I.47.23: cycles at 41,42,
                // 43,44... each ~65s of shadow validation at loss 1e-6). A
                // short fixed gap after EVERY cycle bounds structural spend.
                if (structural_cooldown_ < config::STRUCTURAL_INTER_CYCLE_GAP) {
                    structural_cooldown_ = config::STRUCTURAL_INTER_CYCLE_GAP;
                }

                // Degenerate-loop detection: if the same hypothesis type is
                // committed repeatedly without meaningful best-loss improvement,
                // suppress it. Prevents d2's endless SIN_INJECTION loop where
                // each commit provides <0.1% improvement.
                double improve_threshold = best_overall_loss_ * 0.999;  // 0.1% minimum
                if (current_loss >= improve_threshold) {
                    if (committed_hyp_type == last_committed_hyp_type_) {
                        consecutive_ineffective_commits_++;
                    } else {
                        consecutive_ineffective_commits_ = 1;
                    }
                    last_committed_hyp_type_ = committed_hyp_type;
                    if (consecutive_ineffective_commits_ >= 3) {
                        suppressed_hyp_types_.insert(committed_hyp_type);
                        Logger::info("Suppressing hypothesis type " + std::to_string(committed_hyp_type)
                                    + " (3 consecutive commits without meaningful improvement)");
                    }
                } else {
                    consecutive_ineffective_commits_ = 0;
                    last_committed_hyp_type_ = committed_hyp_type;
                }
            } else {
                // Structural attempt ran but committed nothing. After a run
                // of these, impose a cooldown so we stop burning epochs on
                // candidates that keep getting rejected.
                ++consecutive_structural_failures_;
                if (consecutive_structural_failures_ >= cfg_.structural_failure_threshold) {
                    structural_cooldown_ = cfg_.structural_cooldown;
                    consecutive_structural_failures_ = 0;
                    Logger::info("Structural cooldown engaged: " + std::to_string(cfg_.structural_cooldown)
                                + " epochs off after " + std::to_string(cfg_.structural_failure_threshold)
                                + " consecutive failed structural cycles");
                }
            }
        }  // end if (plateau_triggered)

            // Phase 6: Cleanup 鈥?Periodically clean the graph
            if (cfg_.compile_enabled && (epoch % cfg_.compile_interval == 0)) {
                Logger::verbose("Compile at epoch " + std::to_string(epoch));
                auto cr = compile();
            stats_.dead_nodes_removed += cr.dead_nodes_removed;
            stats_.constants_folded   += cr.constants_folded;
            stats_.neurons_compressed += cr.neurons_compressed;
            if (cr.dead_nodes_removed + cr.constants_folded + cr.neurons_compressed > 0) {
                stats_.compiles_performed++;
                current_loss = evaluate_loss(training_data_);
            }
        }

        // Update best (train loss 鈥?drives early stop + divergence net)
        if (current_loss < best_overall_loss_) {
            best_overall_loss_ = current_loss;
            epochs_since_best_ = 0;
        }

        // Validation-based model selection: snapshot the graph at its best
        // VALIDATION loss (evaluated once per epoch, after SGD + structural
        // commits + compiles). This is the model that gets deployed. Without
        // it, train-based selection deploys the most-overfit epoch
        // (CIFAR gray: train 1.9 / val 5.1 — worse than the 3.25 base rate).
        // PERF: only evaluate when the graph moved this epoch — the candidate
        // val model can only improve when the graph did (train improved or a
        // structural commit landed). Saves ~200 sequential executes on every
        // stagnant plateau epoch.
        bool graph_moved = (current_loss < best_overall_loss_)
                        || epochs_since_structural_ <= 0;
        if (!validation_data_.samples.empty() && graph_moved) {
            double val_loss = compute_validation_loss(*graph_);
            if (val_loss < best_val_loss_) {
                best_val_loss_ = val_loss;
                best_val_snapshot_train_loss_ = current_loss;
                best_graph_snapshot_ = graph_->clone();
                Logger::verbose("New best val loss: " + std::to_string(val_loss)
                               + " (train " + std::to_string(current_loss) + ")");
            }
        } else if (validation_data_.samples.empty() && current_loss < best_overall_loss_) {
            // No validation data: fall back to train-based snapshot
            best_graph_snapshot_ = graph_->clone();
        }

        // Periodic checkpoint: serialize the current best snapshot to disk
        // every snapshot_interval epochs. Crash/reboot insurance for long
        // runs 鈥?the file on disk is always 鈮?interval epochs stale.
        if (!cfg_.snapshot_path.empty()
            && epoch % cfg_.snapshot_interval == 0
            && best_graph_snapshot_) {
            try {
                save_graph_to_file(*best_graph_snapshot_, cfg_.snapshot_path);
                Logger::verbose("Checkpoint saved to " + cfg_.snapshot_path
                               + " at epoch " + std::to_string(epoch));
            } catch (...) {
                Logger::info("Checkpoint save FAILED (path: " + cfg_.snapshot_path + ")");
            }
        }

        // Outer-loop early stopping: if no improvement in best_overall_loss_
        // for cfg_.early_stop_patience epochs, stop. Trims wasted epochs on
        // already-converged / noisy tasks and bounds thrash on stuck ones.
        if (cfg_.early_stop_patience > 0 && epochs_since_best_ >= cfg_.early_stop_patience) {
            Logger::info("Early stop at epoch " + std::to_string(epoch)
                        + " 鈥?no best-loss improvement in "
                        + std::to_string(cfg_.early_stop_patience) + " epochs"
                        + " (best=" + std::to_string(best_overall_loss_) + ")");
            break;
        }
    }

    // Deploy the best-generalizing graph.
    // Val-selection active: always restore the best-validation snapshot 鈥?    // the final epoch typically has better TRAIN loss but worse validation
    // (overfitting), so condition-on-train would never fire.
    // Fallback (no val data): restore best-train snapshot if SGD overshot.
    if (best_graph_snapshot_) {
        if (best_val_loss_ < config::LOSS_SENTINEL_INF) {
            graph_ = std::move(best_graph_snapshot_);
            current_loss = best_val_snapshot_train_loss_;
            Logger::info("Restored best-val graph snapshot (val="
                        + std::to_string(best_val_loss_)
                        + ", train=" + std::to_string(current_loss) + ")");
        } else if (current_loss > best_overall_loss_) {
            graph_ = std::move(best_graph_snapshot_);
            current_loss = best_overall_loss_;
            Logger::info("Restored best graph snapshot (loss=" + std::to_string(best_overall_loss_) + ")");
        }
    }

    stats_.final_loss = current_loss;
    stats_.best_loss  = best_overall_loss_;

    // Final checkpoint: the deployed (best-val) graph, written on exit so
    // the file is never stale even when the run ends between intervals.
    if (!cfg_.snapshot_path.empty() && graph_) {
        try {
            save_graph_to_file(*graph_, cfg_.snapshot_path);
            Logger::info("Final graph saved to " + cfg_.snapshot_path);
        } catch (...) {
            Logger::info("Final graph save FAILED (path: " + cfg_.snapshot_path + ")");
        }
    }
    return current_loss;
}

// ============================================================================
// diagnose 鈥?blame analysis + target propagation
// ============================================================================
// ============================================================================
// error_weighted_split — CART-style threshold selection for guard regions
// ============================================================================
// Given (condition value, residual) pairs, find the threshold t that best
// partitions samples into two sets {x <= t} and {x > t} such that residual
// variance WITHIN each set is minimized (weighted SSE criterion — exactly
// the regression-tree split from CART). This replaces the median heuristic
// for non-differentiable nodes (IFELSE/MUX): the guard region {x | a < x
// <= b} is derived from where the ERROR actually changes character.
//
// Returns the chosen threshold via out-param; returns the achieved weighted
// SSE reduction (>= 0) so callers can rank candidate condition sources.
static double error_weighted_split(const std::vector<std::pair<Value, Value>>& vx,
                                   Value& out_threshold,
                                   const std::vector<Value>& exclude_near = {},
                                   Value exclude_margin = 0.0) {
    if (vx.size() < 8) return 0.0;   // too few to split reliably
    std::vector<std::pair<Value, Value>> s(vx);
    std::sort(s.begin(), s.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    size_t n = s.size();
    // Prefix sums for O(n) split scan
    std::vector<double> pre_sum(n + 1, 0.0), pre_sq(n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) {
        pre_sum[i + 1] = pre_sum[i] + s[i].second;
        pre_sq[i + 1]  = pre_sq[i]  + s[i].second * s[i].second;
    }
    double total_sum = pre_sum[n];
    double total_sq  = pre_sq[n];
    double total_sse = total_sq - total_sum * total_sum / static_cast<double>(n);

    double best_reduction = 0.0;
    Value best_thr = s[n / 2].first;
    // Candidate split AFTER index i (left = [0..i], right = [i+1..n-1]);
    // require >= 4 samples per side; split at value MIDPOINT so ties break
    // cleanly.
    for (size_t i = 4; i + 4 < n; ++i) {
        if (s[i].first == s[i + 1].first) continue;   // no actual boundary
        // v2 margin exclusion: skip splits near ALREADY-COMMITTED thresholds
        // (the graph's existing ifelse_threshold/mux_threshold constants);
        // sequential commits then find the NEXT boundary, not the same one.
        if (exclude_margin > 0.0) {
            bool near_committed = false;
            for (Value ev : exclude_near) {
                if (std::abs((s[i].first + s[i + 1].first) * 0.5 - ev) <= exclude_margin) {
                    near_committed = true;
                    break;
                }
            }
            if (near_committed) continue;
        }
        double nl = static_cast<double>(i + 1);
        double nr = static_cast<double>(n - i - 1);
        double sl = pre_sum[i + 1],  sql = pre_sq[i + 1];
        double sr = total_sum - sl,  sqr = total_sq - sql;
        double sse_l = sql - sl * sl / nl;
        double sse_r = sqr - sr * sr / nr;
        double reduction = total_sse - (sse_l + sse_r);
        if (reduction > best_reduction) {
            best_reduction = reduction;
            best_thr = (s[i].first + s[i + 1].first) * 0.5;
        }
    }
    out_threshold = best_thr;
    return best_reduction;
}

// K-split variant: find the top-K variance-reducing thresholds with
// margin exclusion BETWEEN them (greedy CART tree growth in one shot).
// Used by IFELSE_PRESERVE multi-split emission: K boundaries injected
// atomically, so validation sees the whole tree's benefit (single splits
// on striped residuals move ~1/K of the loss — below the strict gate).
static std::vector<Value> error_weighted_multi_split(
    const std::vector<std::pair<Value, Value>>& vx, int K,
    std::vector<Value>* region_means = nullptr) {
    std::vector<Value> found;
    if (vx.size() < 16 || K <= 0) return found;
    std::vector<std::pair<Value, Value>> rem(vx);
    Value cond_min = rem.front().first, cond_max = rem.front().first;
    for (auto& pr : rem) {
        cond_min = std::min(cond_min, pr.first);
        cond_max = std::max(cond_max, pr.first);
    }
    Value margin = (cond_max - cond_min) * 0.05;
    for (int k = 0; k < K; ++k) {
        Value thr = 0.0;
        double red = error_weighted_split(rem, thr, found, margin);
        if (red <= 0.0) break;
        found.push_back(thr);
        // Remove samples within the excluded margin? No — keep all samples;
        // the exclusion list prevents re-finding the same boundary, and the
        // SSE criterion naturally targets the next-biggest reduction.
        if (found.size() >= static_cast<size_t>(K)) break;
    }
    // M7.6(b): compute per-region residual MEANS (sorted thresholds define
    // the regions). The sign/magnitude of each region's mean residual is
    // the correction that region needs — used to bias-init the PRESERVE
    // gates so the tree starts pointing the right way instead of training
    // from zero.
    if (region_means != nullptr && !found.empty()) {
        std::sort(found.begin(), found.end());
        region_means->assign(found.size() + 1, 0.0);
        std::vector<size_t> rcounts(found.size() + 1, 0);
        for (const auto& pr : vx) {
            size_t r = 0;
            while (r < found.size() && pr.first > found[r]) ++r;
            (*region_means)[r] += pr.second;
            rcounts[r]++;
        }
        for (size_t r = 0; r < region_means->size(); ++r) {
            if (rcounts[r] > 0) (*region_means)[r] /= static_cast<Value>(rcounts[r]);
        }
    }
    return found;
}

std::vector<EvolutionEngine::FailureDiagnosis> EvolutionEngine::diagnose() {
    std::vector<FailureDiagnosis> results;

    // Collect OUTPUT node IDs
    std::vector<uint64_t> output_node_ids;
    for (const auto& node : graph_->get_nodes()) {
        if (node->get_type() == NodeType::OUTPUT) output_node_ids.push_back(node->get_id());
    }
    if (output_node_ids.empty()) return results;

    // Run error attribution on the training data
    auto attribution = graph_->compute_error_attribution(
        training_data_.samples, cfg_.attribution_epsilon, 0, rng_() % config::ATTRIBUTION_SEED_RANGE,
        input_data_to_graph_, output_data_to_graph_);

    // Debug: log all attribution results
    for (const auto& attr : attribution) {
        const Node* n = graph_->get_node(attr.node_id);
        std::string ntype = n ? node_type_to_string(n->get_type()) : "?";
        Logger::verbose("attr node=" + std::to_string(attr.node_id) +
                       " type=" + ntype +
                       " base=" + std::to_string(attr.base_error) +
                       " pert=" + std::to_string(attr.perturbed_error) +
                       " blame=" + std::to_string(attr.blame));
    }

    // Helper: build a FailureDiagnosis from an attribution result, running
    // all the existing structural checks (dead-branch, constant-output)
    // and target propagation for each output node.
    auto build_diag = [&](const Graph::ErrorAttributionResult& attr) -> FailureDiagnosis {
        FailureDiagnosis diag;
        diag.failing_node        = attr.node_id;
        diag.mean_blame          = attr.blame;
        diag.is_constant_output  = false;
        diag.has_dead_branch     = false;

        // Check for IFELSE with dead branches
        const Node* n = graph_->get_node(attr.node_id);
        if (n && n->get_type() == NodeType::IFELSE && n->get_num_outputs() >= 2) {
            Value out0 = graph_->get_any_node_output(attr.node_id);
            // If both output ports always produce same sign, flag as potentially dead
            // (simplistic heuristic; in practice we'd track per-sample branch usage)
            if (std::abs(out0) < config::DEAD_BRANCH_SIGNAL_THRESHOLD) diag.has_dead_branch = true;
        }

        // Check constant output
        if (n && n->get_num_outputs() > 0) {
            // Use the blackboard registry for per-sample outputs
            auto it = blackboard_registry_.find(attr.node_id);
            if (it != blackboard_registry_.end() && it->second.size() >= 2) {
                double var = 0.0, mean = 0.0;
                for (auto v : it->second) mean += v;
                mean /= it->second.size();
                for (auto v : it->second) var += (v - mean) * (v - mean);
                var /= it->second.size();
                if (var < config::CONSTANT_OUTPUT_VARIANCE_THRESHOLD)
                    diag.is_constant_output = true;
            }
        }

        // Target propagation: only for OUTPUT nodes that this failing_node
        // actually feeds (forward-reachable). The previous loop ran
        // compute_targets for EVERY output, pooling all outputs' targets into
        // one residual 鈥?for multi-output graphs (e.g. predicting [x, x^2,
        // sin(x)] from one input) this blended three different functions into
        // a meaningless profile and no single structural fix could satisfy all
        // of them. Each output has its own bottleneck in the starter
        // architecture, so scoping here lets each be diagnosed and fixed
        // independently. We test forward-reachability via get_ancestors(out):
        // failing_node feeds out_id iff failing_node is an ancestor of out_id.
        for (auto out_id : output_node_ids) {
            auto anc = graph_->get_ancestors(out_id);
            if (std::find(anc.begin(), anc.end(), diag.failing_node) != anc.end()) {
                compute_targets(diag, out_id);
            }
        }
        return diag;
    };

    // First pass: nodes with negative blame (they actively harm the output).
    int count = 0;
    for (const auto& attr : attribution) {
        if (count >= cfg_.max_failures_to_fix) break;
        if (attr.blame >= 0.0) continue;
        results.push_back(build_diag(attr));
        count++;
    }

    // Structural inability fallback: when no node has negative blame but the
    // model is at a plateau (high loss), the topology itself is insufficient.
    // Surface the node with the HIGHEST positive blame 鈥?perturbing it causes
    // the largest error, meaning the graph depends on it most heavily and
    // would benefit from additional parallel structure (NEURON, CONTEXT_WIRE,
    // or IFELSE split). Using the real blame value (instead of 0.0) lets
    // classify_failure() and generate_candidates() produce meaningful hypotheses.
    if (results.empty() && !attribution.empty()) {
        const bool multi_output = output_node_ids.size() > 1;
        auto outputs_fed = [&](uint64_t node_id) {
            int n = 0;
            for (auto out_id : output_node_ids) {
                auto anc = graph_->get_ancestors(out_id);
                if (std::find(anc.begin(), anc.end(), node_id) != anc.end()) ++n;
            }
            return n;
        };

        if (multi_output) {
            // Emit one diagnosis PER output-specific positive-blame node.
            // build_diag scopes each to its own output (targets from only
            // the output it feeds), so each output gets its own structural
            // fix attempt 鈥?the x虏 NEURON gets MULTIPLY-self, sin gets
            // TANH_SERIES, etc.
            for (const auto& attr : attribution) {
                if ((int)results.size() >= cfg_.max_failures_to_fix) break;
                if (attr.blame <= 0.0) continue;
                if (outputs_fed(attr.node_id) != 1) continue;
                results.push_back(build_diag(attr));
                Logger::info("Structural inability 鈥?output-specific bottleneck node "
                            + std::to_string(attr.node_id)
                            + " (blame=" + std::to_string(attr.blame) + ")");
            }
        }

        // Fallback (single-output or no output-specific node found).
        if (results.empty()) {
            const Graph::ErrorAttributionResult* global_best = nullptr;
            for (const auto& attr : attribution) {
                if (attr.blame <= 0.0) continue;
                if (!global_best || attr.blame > global_best->blame) global_best = &attr;
            }
            if (global_best) {
                results.push_back(build_diag(*global_best));
                Logger::info("Structural inability detected 鈥?surfacing bottleneck node "
                            + std::to_string(global_best->node_id)
                            + " (blame=" + std::to_string(global_best->blame) + ")");
            }
        }
    }

    Logger::verbose("diag results: " + std::to_string(results.size()) + " failures");

    return results;
}

// ============================================================================
// compute_targets 鈥?backward-pass target propagation for one failing node
// ============================================================================
void EvolutionEngine::compute_targets(FailureDiagnosis& diag, uint64_t output_node_id) {
    // Collect INPUT node IDs
    std::vector<uint64_t> input_node_ids;
    for (const auto& node : graph_->get_nodes()) {
        if (node->get_type() == NodeType::INPUT) input_node_ids.push_back(node->get_id());
    }

    // Collect upstream nodes (nodes that feed into diag.failing_node transitively)
    std::vector<uint64_t> upstream_ids = graph_->get_ancestors(diag.failing_node);

    // Collect all nodes in the graph for local_inputs (any node could be upstream)
    // PERF: cap the sample count for target estimation. This runs on EVERY
    // plateau-triggered structural cycle (per-epoch after patience), and was
    // measured at ~100s/cycle on 800-sample datasets (I.47.23: 68s gaps).
    // 200 strided samples give the same quality estimates at 4x less cost.
    const size_t n_total = training_data_.samples.size();
    const size_t n_cap = static_cast<size_t>(config::COMPUTE_TARGETS_MAX_SAMPLES);
    const size_t stride = (n_total > n_cap) ? (n_total / n_cap) : 1;
    for (size_t s_idx = 0; s_idx < n_total; s_idx += stride) {
        const auto& sample = training_data_.samples[s_idx];
        // Set inputs 鈥?use input_data_to_graph_ to translate CSV column IDs
        // (keys of sample.inputs) to graph node IDs. The old code iterated
        // graph node IDs and looked them up directly in sample.inputs, which
        // only worked by coincidence when node IDs matched column IDs.
        for (auto nid : input_node_ids) {
            graph_->set_input_value(nid, 0.0);
        }
        for (const auto& kv : sample.inputs) {
            auto map_it = input_data_to_graph_.find(kv.first);
            if (map_it != input_data_to_graph_.end()) {
                graph_->set_input_value(map_it->second, kv.second);
            }
        }
        graph_->execute();

        // Read the OUTPUT target for this sample. sample.targets is keyed by
        // the CSV DATA id (the key used in SampleIODesc.targets), NOT by the
        // graph node id 鈥?translate the graph output_node_id back to its data
        // id before lookup. Without this, the lookup misses and target
        // defaults to 0.0 for EVERY sample, silently starving the complexity
        // profiler of signal (it profiled current_out / noise instead of the
        // true target function).
        uint64_t target_data_key = output_node_id;
        for (const auto& kv : output_data_to_graph_) {
            if (kv.second == output_node_id) { target_data_key = kv.first; break; }
        }
        auto tit     = sample.targets.find(target_data_key);
        Value target = (tit != sample.targets.end()) ? tit->second : 0.0;

        // Profiling target: the raw OUTPUT target for this sample.
        //
        // The previous scheme back-propagated the output error to the
        // bottleneck via target_val = current_out - error/importance. That is
        // the mathematically-ideal *bottleneck* output, but it is numerically
        // fragile: when the downstream OUTPUT has a large learned scale S,
        // importance == S and target_val collapses to (target-bias)/S, which
        // (a) loses ~log10(S) digits to cancellation and (b) leaves var_r ~ 0
        // so the complexity profiler has no signal 鈥?interaction targets like
        // y=x0*x1 were then mis-read as linear/constant.
        //
        // For DETECTION (does some input feature g(inputs) explain the
        // residual?), the raw output target is the cleanest, best-scaled
        // signal: it is exactly the function we are trying to fit, in output
        // units, with no cancellation. Downstream consumers (classify_failure,
        // compute_complexity_profile, search_blackboard, IFELSE threshold) all
        // operate on "the value to fit/split", which the output target is.
        Value current_out = graph_->get_any_node_output(diag.failing_node);
        (void)current_out;  // retained for potential debug; not used below
        Value target_val = target;
        // Clamp only as a safety net against pathological target magnitudes.
        target_val = std::max(-config::TARGET_PROP_CLAMP,
                              std::min(config::TARGET_PROP_CLAMP, target_val));

        diag.targets.push_back(target_val);

        // Record local inputs: outputs of all upstream nodes
        std::unordered_map<uint64_t, Value> local_map;
        for (auto uid : upstream_ids) {
            local_map[uid] = graph_->get_any_node_output(uid);
        }
        // Also add INPUT nodes' raw values 鈥?translate CSV column IDs to
        // graph node IDs via input_data_to_graph_ (same correct pattern).
        for (auto nid : input_node_ids) {
            local_map[nid] = 0.0;
        }
        for (const auto& kv : sample.inputs) {
            auto map_it = input_data_to_graph_.find(kv.first);
            if (map_it != input_data_to_graph_.end()) {
                local_map[map_it->second] = kv.second;
            }
        }
        diag.local_inputs.push_back(std::move(local_map));
    }
}

// ============================================================================
// pearson_correlation 鈥?Pearson r between two vectors
// ============================================================================
double EvolutionEngine::pearson_correlation(const std::vector<Value>& a,
                                            const std::vector<Value>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    size_t n = a.size();

    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < n; ++i) { mean_a += a[i]; mean_b += b[i]; }
    mean_a /= n; mean_b /= n;

    double cov = 0.0, var_a = 0.0, var_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double da = a[i] - mean_a, db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    double denom = std::sqrt(var_a * var_b);
    return denom > config::GRADIENT_ZERO_THRESHOLD ? cov / denom : 0.0;
}

// ============================================================================
// search_blackboard 鈥?correlation search across all graph signals
// ============================================================================
std::vector<EvolutionEngine::BlackboardSignal> EvolutionEngine::search_blackboard(
    const FailureDiagnosis& diag) {
    std::vector<BlackboardSignal> results;

    if (diag.targets.empty()) return results;

    // The target vector is what we're trying to correlate against
    const auto& targets = diag.targets;

    // Check every signal in the blackboard registry
    std::vector<std::pair<Value, BlackboardSignal>> scored;
    for (const auto& kv : blackboard_registry_) {
        uint64_t nid = kv.first;
        if (nid == diag.failing_node) continue;  // skip self

        // Skip OUTPUT nodes (we can't reconnect an output to itself)
        const Node* nd = graph_->get_node(nid);
        if (nd && nd->get_type() == NodeType::OUTPUT) continue;

        // Compute Pearson r between this signal and the targets
        if (kv.second.size() < 2) continue;
        double corr = pearson_correlation(kv.second, targets);

        BlackboardSignal sig;
        sig.node_id     = nid;
        sig.correlation = std::abs(corr);  // use absolute correlation for ranking
        sig.is_input    = (nd && nd->get_type() == NodeType::INPUT);

        scored.emplace_back(sig.correlation, std::move(sig));
    }

    // Sort by absolute correlation descending, take top candidates
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int limit = std::min(cfg_.blackboard_max_candidates, (int)scored.size());
    for (int i = 0; i < limit; ++i) {
        results.push_back(std::move(scored[i].second));
    }

    return results;
}

// ============================================================================
// classify_failure 鈥?analyze mini-dataset to determine failure nature
// ============================================================================
EvolutionEngine::FailureType EvolutionEngine::classify_failure(
    const FailureDiagnosis& diag) const {

    const auto& targets = diag.targets;
    if (targets.empty()) return FailureType::UNKNOWN;

    // --- Check 1: BOOLEAN_BOUNDARY 鈥?targets cluster into distinct groups ---
    {
        std::vector<Value> sorted = targets;
        std::sort(sorted.begin(), sorted.end());
        double total_range = sorted.back() - sorted.front();
        if (total_range > config::GRADIENT_ZERO_THRESHOLD) {
            // Find the largest gap between adjacent sorted targets
            double max_gap = 0.0;
            size_t gap_idx = 0;
            for (size_t i = 1; i < sorted.size(); ++i) {
                double gap = sorted[i] - sorted[i - 1];
                if (gap > max_gap) {
                    max_gap = gap;
                    gap_idx = i;
                }
            }
            double gap_ratio = max_gap / total_range;

            // If gap is large, split the data at the gap and check cluster quality
            if (gap_ratio > config::BOOLEAN_GAP_RATIO_THRESHOLD && gap_idx > 0 && gap_idx < sorted.size()) {
                Value threshold = (sorted[gap_idx - 1] + sorted[gap_idx]) * 0.5;
                // Count samples in each cluster
                double mean_lo = 0.0, mean_hi = 0.0;
                size_t cnt_lo = 0, cnt_hi = 0;
                for (auto v : targets) {
                    if (v <= threshold) { mean_lo += v; cnt_lo++; }
                    else { mean_hi += v; cnt_hi++; }
                }
                if (cnt_lo > 1 && cnt_hi > 1) {
                    mean_lo /= cnt_lo;
                    mean_hi /= cnt_hi;
                    // Between-cluster variance vs within-cluster
                    double between = cnt_lo * cnt_hi * (mean_hi - mean_lo) * (mean_hi - mean_lo)
                                     / (cnt_lo + cnt_hi);
                    double within_lo = 0.0, within_hi = 0.0;
                    for (auto v : targets) {
                        if (v <= threshold) within_lo += (v - mean_lo) * (v - mean_lo);
                        else                 within_hi += (v - mean_hi) * (v - mean_hi);
                    }
                    double within = (within_lo + within_hi) / (cnt_lo + cnt_hi);
                    if (within > config::GRADIENT_ZERO_THRESHOLD && between / within > config::BOOLEAN_BETWEEN_WITHIN_RATIO) {
                        return FailureType::BOOLEAN_BOUNDARY;
                    }
                }
            }
        }
    }

    // --- Check 2: LINEAR_OFFSET 鈥?fit linear model, check R虏 ---
    if (diag.local_inputs.size() == targets.size() && targets.size() >= 3) {
        // Robustness: if all targets are identical, it's a constant (UNKNOWN)
        double t_var = 0.0, t_mean = 0.0;
        for (auto v : targets) t_mean += v;
        t_mean /= targets.size();
        for (auto v : targets) t_var += (v - t_mean) * (v - t_mean);
        if (t_var < config::CONSTANT_OUTPUT_VARIANCE_THRESHOLD) {
            return FailureType::UNKNOWN;
        }

        // For the linear fit, use all upstream node outputs as features.
        // Build feature matrix column by column.
        size_t N = targets.size();
        // Collect all upstream node IDs that appear in any local_inputs map
        std::vector<uint64_t> feature_ids;
        for (const auto& li : diag.local_inputs) {
            for (const auto& kv : li) {
                if (std::find(feature_ids.begin(), feature_ids.end(), kv.first) == feature_ids.end()) {
                    feature_ids.push_back(kv.first);
                }
            }
        }
        // Cap features to avoid overfitting on small datasets
        if (feature_ids.size() > config::LINEAR_FIT_MAX_FEATURES) feature_ids.resize(config::LINEAR_FIT_MAX_FEATURES);

        size_t F = feature_ids.size();
        if (F == 0) return FailureType::NON_LINEAR_CURVE;

        // Solve least squares: (X^T X) 尾 = X^T y
        // X is N脳(F+1) with a constant-1 column for bias
        // Simple closed-form using normal equations.

        // Compute X^T X
        std::vector<double> xtx((F + 1) * (F + 1), 0.0);
        std::vector<double> xty(F + 1, 0.0);

        for (size_t i = 0; i < N; ++i) {
            // bias term (column 0)
            xtx[0] += 1.0;
            xty[0] += targets[i];

            for (size_t j = 0; j < F; ++j) {
                auto it = diag.local_inputs[i].find(feature_ids[j]);
                Value x_j = (it != diag.local_inputs[i].end()) ? it->second : 0.0;

                xtx[0 + (j + 1) * (F + 1)] += x_j;              // row 0, col j+1
                xtx[(j + 1) + 0 * (F + 1)] += x_j;              // row j+1, col 0
                xty[j + 1] += x_j * targets[i];

                for (size_t k = 0; k < F; ++k) {
                    auto it_k = diag.local_inputs[i].find(feature_ids[k]);
                    Value x_k = (it_k != diag.local_inputs[i].end()) ? it_k->second : 0.0;
                    xtx[(j + 1) + (k + 1) * (F + 1)] += x_j * x_k;
                }
            }
        }

        // Gaussian elimination to solve for 尾
        std::vector<double> beta = xty;  // starts as RHS
        std::vector<double> Atx = xtx;   // working copy

        for (size_t col = 0; col <= F; ++col) {
            // Pivot: find max in column
            size_t max_row = col;
            double max_val = std::abs(Atx[col + col * (F + 1)]);
            for (size_t row = col + 1; row <= F; ++row) {
                double v = std::abs(Atx[row + col * (F + 1)]);
                if (v > max_val) { max_val = v; max_row = row; }
            }
            if (max_val < config::GAUSS_PIVOT_TOLERANCE) continue;  // singular 鈥?stop

            // Swap rows
            if (max_row != col) {
                for (size_t c = 0; c <= F; ++c)
                    std::swap(Atx[col + c * (F + 1)], Atx[max_row + c * (F + 1)]);
                std::swap(beta[col], beta[max_row]);
            }

            // Eliminate below
            double diag = Atx[col + col * (F + 1)];
            for (size_t row = col + 1; row <= F; ++row) {
                double factor = Atx[row + col * (F + 1)] / diag;
                for (size_t c = col; c <= F; ++c)
                    Atx[row + c * (F + 1)] -= factor * Atx[col + c * (F + 1)];
                beta[row] -= factor * beta[col];
            }
        }

        // Back-substitution
        for (int col = (int)F; col >= 0; --col) {
            double sum = 0.0;
            for (size_t c = col + 1; c <= F; ++c)
                sum += Atx[col + c * (F + 1)] * beta[c];
            if (std::abs(Atx[col + col * (F + 1)]) > config::GAUSS_PIVOT_TOLERANCE)
                beta[col] = (beta[col] - sum) / Atx[col + col * (F + 1)];
            else
                beta[col] = 0.0;
        }

        // Compute R虏
        double ss_total = 0.0, ss_residual = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double pred = beta[0];  // bias
            for (size_t j = 0; j < F; ++j) {
                auto it = diag.local_inputs[i].find(feature_ids[j]);
                pred += beta[j + 1] * ((it != diag.local_inputs[i].end()) ? it->second : 0.0);
            }
            double err = targets[i] - pred;
            ss_residual += err * err;
            ss_total += (targets[i] - t_mean) * (targets[i] - t_mean);
        }

        double r2 = ss_total > config::GRADIENT_ZERO_THRESHOLD ? 1.0 - ss_residual / ss_total : 0.0;
        if (r2 > config::LINEAR_FIT_R2_THRESHOLD) {
            return FailureType::LINEAR_OFFSET;
        }
    }

    if (targets.size() < 3) return FailureType::UNKNOWN;
    return FailureType::NON_LINEAR_CURVE;
}

// ============================================================================
// compute_complexity_profile 鈥?fingerprint the residual at the bottleneck node
// ============================================================================
// See evolution.h for the full design rationale. Key idea: classify_failure()
// already does a linear fit and returns one of three buckets; this method goes
// deeper and returns the *shape* of the residual (polynomial coeffs, pairwise
// interaction, boundedness, sharp-boundary) so the candidate generator can
// pick a structural template that matches the shape rather than just the bucket.
// ============================================================================
EvolutionEngine::ComplexityProfile
EvolutionEngine::compute_complexity_profile(const FailureDiagnosis& diag) const {
    ComplexityProfile prof;

    const auto& targets = diag.targets;
    if (targets.empty() || diag.local_inputs.empty()) return prof;

    // Identify INPUT node IDs from graph_ (const iteration only)
    std::vector<uint64_t> input_ids;
    for (const auto& node : graph_->get_nodes()) {
        if (node->get_type() == NodeType::INPUT) input_ids.push_back(node->get_id());
    }

    const size_t N = targets.size();
    const size_t F = input_ids.size();
    prof.num_inputs  = F;
    prof.num_samples = N;
    if (F == 0 || N < 4) return prof;

    // Build X[N 脳 F] using INPUT features only
    std::vector<std::vector<double>> X(N, std::vector<double>(F, 0.0));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < F; ++j) {
            auto it = diag.local_inputs[i].find(input_ids[j]);
            if (it != diag.local_inputs[i].end()) X[i][j] = it->second;
        }
    }

    // --- Residual stats ---
    double sum = 0.0;
    for (auto v : targets) sum += v;
    prof.mean_r = sum / N;
    double var = 0.0;
    for (auto v : targets) var += (v - prof.mean_r) * (v - prof.mean_r);
    var /= N;
    prof.var_r = var;
    prof.min_r = *std::min_element(targets.begin(), targets.end());
    prof.max_r = *std::max_element(targets.begin(), targets.end());
    prof.bound_ratio = (var > 1e-12) ? (prof.max_r - prof.min_r) / std::sqrt(var) : 0.0;
    prof.bounded = (prof.bound_ratio > 0.0 && prof.bound_ratio < config::PROFILE_BOUNDED_RATIO_MAX);

    // --- Degree-2 polynomial fit via Gaussian elimination ---
    // Feature layout: [1, x_1..x_F, x_1虏..x_F虏, x_1路x_2, x_1路x_3, ..., x_{F-1}路x_F]
    size_t n_cross = F * (F - 1) / 2;
    size_t P = 1 + 2 * F + n_cross;
    if (N >= P + 2) {
        std::vector<std::vector<double>> Phi(N, std::vector<double>(P, 0.0));
        for (size_t i = 0; i < N; ++i) {
            Phi[i][0] = 1.0;
            for (size_t j = 0; j < F; ++j) {
                Phi[i][1 + j]     = X[i][j];
                Phi[i][1 + F + j] = X[i][j] * X[i][j];
            }
            size_t idx = 1 + 2 * F;
            for (size_t a = 0; a < F; ++a) {
                for (size_t b = a + 1; b < F; ++b) {
                    Phi[i][idx++] = X[i][a] * X[i][b];
                }
            }
        }

        // Normal equations: (Phi^T Phi) 尾 = Phi^T y
        std::vector<double> AtA(P * P, 0.0);
        std::vector<double> Aty(P, 0.0);
        for (size_t i = 0; i < N; ++i) {
            for (size_t r = 0; r < P; ++r) {
                Aty[r] += Phi[i][r] * targets[i];
                for (size_t c = 0; c < P; ++c) {
                    AtA[r + c * P] += Phi[i][r] * Phi[i][c];
                }
            }
        }

        // Gaussian elimination with partial pivoting
        std::vector<double> beta = Aty;
        std::vector<double> M    = AtA;
        for (size_t col = 0; col < P; ++col) {
            size_t max_row = col;
            double max_val = std::abs(M[col + col * P]);
            for (size_t row = col + 1; row < P; ++row) {
                double v = std::abs(M[row + col * P]);
                if (v > max_val) { max_val = v; max_row = row; }
            }
            if (max_val < config::GAUSS_PIVOT_TOLERANCE) continue;
            if (max_row != col) {
                for (size_t c = 0; c < P; ++c)
                    std::swap(M[col + c * P], M[max_row + c * P]);
                std::swap(beta[col], beta[max_row]);
            }
            double d = M[col + col * P];
            for (size_t row = col + 1; row < P; ++row) {
                double f = M[row + col * P] / d;
                for (size_t c = col; c < P; ++c)
                    M[row + c * P] -= f * M[col + c * P];
                beta[row] -= f * beta[col];
            }
        }
        for (int col = (int)P - 1; col >= 0; --col) {
            double s = 0.0;
            for (size_t c = col + 1; c < P; ++c) s += M[col + c * P] * beta[c];
            if (std::abs(M[col + col * P]) > config::GAUSS_PIVOT_TOLERANCE)
                beta[col] = (beta[col] - s) / M[col + col * P];
            else
                beta[col] = 0.0;
        }

        // R虏
        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double pred = 0.0;
            for (size_t r = 0; r < P; ++r) pred += beta[r] * Phi[i][r];
            ss_res += (targets[i] - pred) * (targets[i] - pred);
            ss_tot += (targets[i] - prof.mean_r) * (targets[i] - prof.mean_r);
        }
        prof.poly_r2     = (ss_tot > 1e-12) ? 1.0 - ss_res / ss_tot : 0.0;
        prof.poly_coeffs = std::move(beta);

        // Track the dominant non-bias polynomial term. Compound-hypothesis
        // detection uses this to check whether the strongest term is a
        // cross-term (x_i路x_j interaction) vs a linear term.
        for (size_t i = 1; i < prof.poly_coeffs.size(); ++i) {
            double v = std::abs(prof.poly_coeffs[i]);
            if (v > std::abs(prof.max_coef_value)) {
                prof.max_coef_value = prof.poly_coeffs[i];
                prof.max_coef_index = i;
            }
        }

        // Decode the dominant non-bias term into a concrete (a,b) input pair.
        // Layout: [0]=bias, [1..F]=lin x_i, [F+1..2F]=x_i^2,
        // [2F+1..]=x_i*x_j (i<j). Used by MULTIPLY source selection 鈥?robust
        // where Sobol is noisy on low-variance residuals.
        if (F >= 1 && prof.poly_coeffs.size() >= 2 * F + 1
            && prof.max_coef_index >= 1 + F) {
            double max_linear = 0.0;
            for (size_t i = 0; i < F; ++i) {
                max_linear = std::max(max_linear, std::abs(prof.poly_coeffs[1 + i]));
            }
            double dom = std::abs(prof.max_coef_value);
            bool dominant = (max_linear > 1e-9)
                          ? (dom / max_linear >= config::PROFILE_INTERACTION_RATIO)
                          : (dom > 1e-3);
            if (dominant) {
                prof.interaction_dominant = true;
                size_t idx = prof.max_coef_index;
                if (idx < 1 + 2 * F) {
                    size_t i = idx - (1 + F);
                    prof.interact_a = i;
                    prof.interact_b = i;  // x_i^2 -> self-product
                } else {
                    size_t k = idx - (1 + 2 * F);
                    bool found = false;
                    for (size_t a = 0; a < F && !found; ++a) {
                        for (size_t b = a + 1; b < F && !found; ++b) {
                            if (k == 0) { prof.interact_a = a; prof.interact_b = b; found = true; }
                            else { --k; }
                        }
                    }
                    if (!found) { prof.interact_a = 0; prof.interact_b = (F > 1 ? 1 : 0); }
                }
            }
        }
    }

    // --- Sobol pairwise interaction index (max over input pairs) ---
    // V_int(x_a,x_b) = Var(E[r|xa,xb]) - Var(E[r|xa]) - Var(E[r|xb])
    // Continuous inputs are discretized into K bins per axis.
    if (F >= 2 && var > 1e-12) {
        const int K = config::PROFILE_SOBOL_BINS;
        double best_pair_strength = -1.0;
        size_t best_a = 0, best_b = 1;
        for (size_t a = 0; a < F; ++a) {
            for (size_t b = a + 1; b < F; ++b) {
                double min_a = X[0][a], max_a = X[0][a];
                double min_b = X[0][b], max_b = X[0][b];
                for (size_t i = 1; i < N; ++i) {
                    if (X[i][a] < min_a) min_a = X[i][a];
                    if (X[i][a] > max_a) max_a = X[i][a];
                    if (X[i][b] < min_b) min_b = X[i][b];
                    if (X[i][b] > max_b) max_b = X[i][b];
                }
                double range_a = max_a - min_a, range_b = max_b - min_b;
                if (range_a < 1e-12 || range_b < 1e-12) continue;

                auto bin_idx = [&](double v, double mn, double rg) -> int {
                    int idx = (int)((v - mn) / rg * K);
                    if (idx < 0) idx = 0;
                    if (idx >= K) idx = K - 1;
                    return idx;
                };

                std::vector<std::vector<double>> sum_ab(K, std::vector<double>(K, 0.0));
                std::vector<std::vector<int>>    cnt_ab(K, std::vector<int>(K, 0));
                std::vector<double> sum_a(K, 0.0), sum_b(K, 0.0);
                std::vector<int>    cnt_a(K, 0),   cnt_b(K, 0);
                for (size_t i = 0; i < N; ++i) {
                    int ia = bin_idx(X[i][a], min_a, range_a);
                    int ib = bin_idx(X[i][b], min_b, range_b);
                    sum_ab[ia][ib] += targets[i]; cnt_ab[ia][ib]++;
                    sum_a[ia]      += targets[i]; cnt_a[ia]++;
                    sum_b[ib]      += targets[i]; cnt_b[ib]++;
                }

                double var_rab = 0.0, var_ra = 0.0, var_rb = 0.0;
                for (size_t i = 0; i < N; ++i) {
                    int ia = bin_idx(X[i][a], min_a, range_a);
                    int ib = bin_idx(X[i][b], min_b, range_b);
                    double e_ab = (cnt_ab[ia][ib] > 0) ? sum_ab[ia][ib] / cnt_ab[ia][ib] : prof.mean_r;
                    double e_a  = (cnt_a[ia]      > 0) ? sum_a[ia]      / cnt_a[ia]      : prof.mean_r;
                    double e_b  = (cnt_b[ib]      > 0) ? sum_b[ib]      / cnt_b[ib]      : prof.mean_r;
                    var_rab += (e_ab - prof.mean_r) * (e_ab - prof.mean_r);
                    var_ra  += (e_a  - prof.mean_r) * (e_a  - prof.mean_r);
                    var_rb  += (e_b  - prof.mean_r) * (e_b  - prof.mean_r);
                }
                var_rab /= N; var_ra /= N; var_rb /= N;

                double V_int = var_rab - var_ra - var_rb;
                double pair_strength = (var > 1e-12) ? V_int / var : 0.0;
                if (pair_strength > best_pair_strength) {
                    best_pair_strength = pair_strength;
                    best_a = a;
                    best_b = b;
                }
            }
        }
        prof.sobol_pairwise = best_pair_strength;
        prof.sobol_pair_a   = best_a;
        prof.sobol_pair_b   = best_b;
        prof.high_pairwise_interaction =
            (best_pair_strength > config::PROFILE_PAIRWISE_INTERACTION_MIN);
    }

    // --- Lipschitz estimate per input axis ---
    // Sort by x_j, take max(|螖r|/|螖x_j|) over adjacent pairs.
    prof.lipschitz_per_axis.assign(F, 0.0);
    for (size_t j = 0; j < F; ++j) {
        std::vector<std::pair<double, double>> xv(N);
        for (size_t i = 0; i < N; ++i) xv[i] = {X[i][j], targets[i]};
        std::sort(xv.begin(), xv.end());
        double L = 0.0;
        for (size_t i = 1; i < N; ++i) {
            double dx = xv[i].first - xv[i - 1].first;
            if (dx > 1e-9) {
                double dr = xv[i].second - xv[i - 1].second;
                double r  = std::abs(dr) / dx;
                if (r > L) L = r;
            }
        }
        prof.lipschitz_per_axis[j] = L;
        if (L > prof.lipschitz_max) prof.lipschitz_max = L;
    }
    prof.sharp_boundary =
        (prof.lipschitz_max > config::PROFILE_SHARP_BOUNDARY_LIPSCHITZ);

    // --- Sign-quadrant means for first two inputs (F鈮?) ---
    // Order: (++, +-, -+, --)
    if (F >= 2) {
        std::vector<double> qsum(4, 0.0);
        std::vector<int>    qcnt(4, 0);
        for (size_t i = 0; i < N; ++i) {
            int idx = (X[i][0] >= 0.0 ? 0 : 2) + (X[i][1] >= 0.0 ? 0 : 1);
            qsum[idx] += targets[i];
            qcnt[idx]++;
        }
        prof.quadrant_means.assign(4, prof.mean_r);
        for (int q = 0; q < 4; ++q)
            if (qcnt[q] > 0) prof.quadrant_means[q] = qsum[q] / qcnt[q];
    }

    return prof;
}

// ----------------------------------------------------------------------------
// format_profile 鈥?single-line human-readable summary
// ----------------------------------------------------------------------------
std::string EvolutionEngine::format_profile(const ComplexityProfile& p) const {
    if (p.num_inputs == 0) return "[PROFILE] (empty 鈥?no INPUTs / too few samples)";

    std::ostringstream oss;
    oss << "[PROFILE] F=" << p.num_inputs
        << " N=" << p.num_samples
        << " var_r=" << std::fixed << std::setprecision(4) << p.var_r
        << " bound_ratio=" << p.bound_ratio
        << " poly_r2=" << p.poly_r2;

    // Polynomial coefficients (degree-2): show only the cross-term and largest
    // |coeff| among non-bias terms 鈥?the full vector would flood the log.
    if (!p.poly_coeffs.empty()) {
        size_t F = p.num_inputs;
        size_t cross_off = 1 + 2 * F;
        if (F >= 2 && p.poly_coeffs.size() >= cross_off + F * (F - 1) / 2) {
            oss << " a12=" << p.poly_coeffs[cross_off];
        }
        double best = 0.0; size_t best_i = 0;
        for (size_t i = 1; i < p.poly_coeffs.size(); ++i) {
            if (std::abs(p.poly_coeffs[i]) > std::abs(best)) {
                best = p.poly_coeffs[i]; best_i = i;
            }
        }
        oss << " max_coef[i=" << best_i << "]=" << best;
    }

    if (p.num_inputs >= 2) {
        oss << " sobol_pair(" << p.sobol_pair_a << "," << p.sobol_pair_b
            << ")=" << p.sobol_pairwise;
        if (!p.quadrant_means.empty()) {
            oss << " quad=[";
            for (int q = 0; q < 4; ++q) {
                if (q) oss << ",";
                oss << std::fixed << std::setprecision(3) << p.quadrant_means[q];
            }
            oss << "]";
        }
    }

    oss << " L_max=" << p.lipschitz_max;
    oss << " | flags:";
    if (p.high_pairwise_interaction) oss << " PAIRWISE";
    if (p.bounded)                   oss << " BOUNDED";
    if (p.sharp_boundary)            oss << " SHARP";
    if (!p.high_pairwise_interaction && !p.bounded && !p.sharp_boundary) oss << " none";
    return oss.str();
}

// ============================================================================
// form_hypothesis 鈥?choose a structural modification strategy
// ============================================================================
EvolutionEngine::Hypothesis EvolutionEngine::form_hypothesis(
    const FailureDiagnosis& diag,
    const std::vector<BlackboardSignal>& blackboard) {

    Hypothesis hyp;

    // Strategy 1: CONTEXT_WIRE 鈥?if a strong correlated signal exists
    if (!blackboard.empty() && blackboard[0].correlation > config::CONTEXT_WIRE_CORR_TRIGGER) {
        hyp.type                = Hypothesis::CONTEXT_WIRE;
        hyp.wire_source_node    = blackboard[0].node_id;
        return hyp;
    }

    // Strategy 2: IFELSE_BOUNDARY_SPLIT 鈥?if the failing node isn't constant
    if (!diag.is_constant_output && diag.targets.size() >= 2) {
        // Find median of targets as split threshold
        std::vector<Value> sorted = diag.targets;
        std::sort(sorted.begin(), sorted.end());
        hyp.type              = Hypothesis::IFELSE_BOUNDARY_SPLIT;
        hyp.split_threshold   = sorted[sorted.size() / 2];

        // Choose a condition source from blackboard signals (prefer INPUT)
        if (!blackboard.empty()) {
            for (const auto& bs : blackboard) {
                if (bs.is_input) {
                    hyp.condition_source_node = bs.node_id;
                    break;
                }
            }
            if (hyp.condition_source_node == 0) {
                hyp.condition_source_node = blackboard[0].node_id;
            }
        } else {
            // Fallback: create a CONSTANT as threshold source
            // (will be handled in apply_shadow_routing)
            hyp.condition_source_node = 0; // signal to create one
        }
        return hyp;
    }

    // Strategy 3: NEURON_TANH_INJECTION 鈥?default fallback
    hyp.type = Hypothesis::NEURON_TANH_INJECTION;
    return hyp;
}

// ============================================================================
// M5.7 zero-plateau edge detection — see constants.h for gating rationale
// ============================================================================
std::vector<std::pair<Value, bool>> EvolutionEngine::detect_zero_plateau_edges(
    uint64_t cond_graph_id, const Graph& g) const {
    std::vector<std::pair<Value, bool>> result;
    // Single-output graphs only (v1)
    size_t out_count = 0;
    uint64_t out_gid = 0;
    for (const auto& n : g.get_nodes()) {
        if (n->get_type() == NodeType::OUTPUT) {
            ++out_count;
            out_gid = n->get_id();
        }
    }
    if (out_count != 1 || training_data_.samples.empty()) return result;
    // samples are keyed by DATA keys, not graph node ids — invert both maps
    uint64_t out_key = 0; bool have_out_key = false;
    for (const auto& kv : output_data_to_graph_) {
        if (kv.second == out_gid) { out_key = kv.first; have_out_key = true; break; }
    }
    uint64_t cond_key = 0; bool have_cond_key = false;
    for (const auto& kv : input_data_to_graph_) {
        if (kv.second == cond_graph_id) { cond_key = kv.first; have_cond_key = true; break; }
    }
    if (!have_out_key || !have_cond_key) return result;

    std::vector<std::pair<Value, Value>> xy;
    xy.reserve(training_data_.samples.size());
    for (const auto& s : training_data_.samples) {
        auto xi = s.inputs.find(cond_key);
        auto yi = s.targets.find(out_key);
        if (xi != s.inputs.end() && yi != s.targets.end()) {
            xy.emplace_back(xi->second, yi->second);
        }
    }
    if (xy.size() < 16) return result;
    std::sort(xy.begin(), xy.end(),
              [](const std::pair<Value, Value>& a, const std::pair<Value, Value>& b) {
                  return a.first < b.first;
              });
    Value y_min = xy.front().second, y_max = y_min;
    Value x_min = xy.front().first, x_max = x_min;
    for (const auto& pr : xy) {
        y_min = std::min(y_min, pr.second); y_max = std::max(y_max, pr.second);
        x_min = std::min(x_min, pr.first);  x_max = std::max(x_max, pr.first);
    }
    const Value y_range = y_max - y_min;
    if (y_range <= 1e-9) return result;
    const Value flat_eps = y_range * config::ZERO_PLATEAU_FLAT_EPS;
    const size_t min_run = std::max<size_t>(
        static_cast<size_t>(config::ZERO_PLATEAU_MIN_RUN), xy.size() / 20);

    std::vector<std::pair<Value, bool>> edges;
    size_t in_flat = 0, flat_total = 0;
    Value run_min = 0, run_max = 0;
    bool have_run = false;
    auto flush_run = [&](size_t run_end) {
        if (in_flat >= min_run) {
            flat_total += in_flat;
            const size_t run_start = run_end - in_flat;
            if (run_start > 0) {
                // LEFT edge of the run: flat lies ABOVE this threshold
                edges.emplace_back(0.5 * (xy[run_start - 1].first + xy[run_start].first), true);
            }
            if (run_end < xy.size()) {
                // RIGHT edge: flat lies BELOW this threshold
                edges.emplace_back(0.5 * (xy[run_end - 1].first + xy[run_end].first), false);
            }
        }
    };
    for (size_t i = 0; i < xy.size(); ++i) {
        const Value y = xy[i].second;
        if (!have_run) {
            run_min = y; run_max = y; have_run = true; in_flat = 1;
        } else if (std::max(run_max, y) - std::min(run_min, y) <= flat_eps) {
            run_max = std::max(run_max, y);
            run_min = std::min(run_min, y);
            ++in_flat;
        } else {
            flush_run(i);
            run_min = y; run_max = y; in_flat = 1;
        }
    }
    flush_run(xy.size());
    const double frac = static_cast<double>(flat_total) / static_cast<double>(xy.size());
    if (frac < config::ZERO_PLATEAU_MIN_FRACTION || edges.empty()) return result;

    // Exclude thresholds already committed in the graph (sequential
    // commits must find the NEXT boundary — same margin logic as the
    // set-guided split's v2 exclusion).
    std::vector<Value> committed_thr;
    for (const auto& n : g.get_nodes()) {
        const std::string& nm = n->get_name();
        if ((nm == "ifelse_threshold" || nm == "mux_threshold")
            && n->get_type() == NodeType::CONSTANT) {
            committed_thr.push_back(static_cast<const ConstantNode*>(n.get())->get_value());
        }
    }
    const Value excl_margin = (x_max - x_min) * 0.05;
    for (const auto& e : edges) {
        bool near_committed = false;
        for (Value ct : committed_thr) {
            if (std::abs(e.first - ct) <= excl_margin) { near_committed = true; break; }
        }
        if (!near_committed) {
            result.push_back(e);
            // Cap 3: measured optimum — cap 6 builds deeper chains (K=6)
            // but settles worse (stripes20 0.207 vs 0.225 at cap 3; the
            // M5.4 settling-cadence limit punishes chain depth).
            if (result.size() >= 3) break;
        }
    }
    Logger::info("Zero-plateau edges: " + std::to_string(result.size()) + " (flat fraction "
                + std::to_string(frac) + ")");
    return result;
}

// ============================================================================
// generate_candidates 鈥?produce scored, ranked hypotheses for retry loop
// ============================================================================
std::vector<EvolutionEngine::Hypothesis> EvolutionEngine::generate_candidates(
    const FailureDiagnosis& diag,
    const std::vector<BlackboardSignal>& blackboard,
    FailureType ftype,
    const ComplexityProfile& profile) {

    struct ScoredHyp {
        Hypothesis hyp;
        double    score;
    };
    std::vector<ScoredHyp> candidates;
    // M1.1: failure-library penalties (populated in the library-prior block)
    std::unordered_map<int, double> family_penalty;

    // M5.4 piecewise signature (function scope — multiple emission blocks
    // read it): residual alternates sharply and repeatedly along one axis
    // => multi-boundary problem => boundary family (guided IFELSE/MUX)
    // outranks capacity families (stacks, sin). Stripes20 lesson.
    bool piecewise_signature = false;
    {
        if (diag.targets.size() >= 32 && !diag.local_inputs.empty()) {
            std::vector<std::pair<Value, Value>> vr;
            vr.reserve(diag.targets.size());
            for (size_t i = 0; i < diag.targets.size() && i < diag.local_inputs.size(); ++i) {
                if (!diag.local_inputs[i].empty()) {
                    vr.emplace_back(diag.local_inputs[i].begin()->second, diag.targets[i]);
                }
            }
            if (vr.size() >= 32) {
                std::sort(vr.begin(), vr.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                int sc = 0;
                for (size_t i = 1; i < vr.size(); ++i) {
                    if ((vr[i].second > 0.0) != (vr[i-1].second > 0.0)) ++sc;
                }
                piecewise_signature = (sc >= 10)
                    && profile.bounded
                    && profile.lipschitz_max > config::PROFILE_SHARP_BOUNDARY_LIPSCHITZ;
            }
        }
        if (piecewise_signature) {
            Logger::info("Piecewise signature detected (" + std::to_string(static_cast<int>(profile.num_inputs))
                        + "-input) — boundary family prioritized");
        }
    }

    // Detect binary classification targets ONCE 鈥?used by both MULTIPLY
    // (to suppress the plateau-fallback for binary problems, where
    // BOOLEAN_COMPOSE is the right tool) and BOOLEAN_COMPOSE (which only
    // fires for binary problems).
    bool is_binary = false;
    if (!training_data_.samples.empty()) {
        std::vector<Value> labels;
        labels.reserve(training_data_.samples.size());
        for (const auto& s : training_data_.samples) {
            if (!s.targets.empty()) labels.push_back(s.targets.begin()->second);
        }
        if (labels.size() >= 4) {
            std::sort(labels.begin(), labels.end());
            Value total_range = labels.back() - labels.front();
            if (total_range > config::BINARY_LABEL_MIN_RANGE) {
                Value max_gap = 0.0;
                size_t gap_idx = 0;
                for (size_t i = 1; i < labels.size(); ++i) {
                    Value g = labels[i] - labels[i - 1];
                    if (g > max_gap) { max_gap = g; gap_idx = i; }
                }
                if (max_gap / total_range > config::BINARY_LABEL_GAP_RATIO &&
                    gap_idx >= 2 && gap_idx <= labels.size() - 2) {
                    Value lo_mean = 0, hi_mean = 0;
                    for (size_t i = 0; i < gap_idx; ++i) lo_mean += labels[i];
                    for (size_t i = gap_idx; i < labels.size(); ++i) hi_mean += labels[i];
                    lo_mean /= gap_idx;
                    hi_mean /= (labels.size() - gap_idx);
                    Value within = 0;
                    for (size_t i = 0; i < gap_idx; ++i) within += (labels[i] - lo_mean) * (labels[i] - lo_mean);
                    for (size_t i = gap_idx; i < labels.size(); ++i) within += (labels[i] - hi_mean) * (labels[i] - hi_mean);
                    within /= labels.size();
                    if (within < config::BINARY_LABEL_MAX_WITHIN_VAR) is_binary = true;
                }
            }
        }
    }

    // Count INPUT signals visible in the blackboard (regardless of
    // correlation magnitude). Used by MULTIPLY plateau-fallback.
    int input_signal_count = 0;
    for (const auto& sig : blackboard) {
        if (sig.is_input) ++input_signal_count;
    }

    // --- Candidate 1: CONTEXT_WIRE ---
    if (!blackboard.empty()) {
        Hypothesis cw;
        cw.type             = Hypothesis::CONTEXT_WIRE;
        cw.wire_source_node = blackboard[0].node_id;
        double score = blackboard[0].correlation;
        // Boost for LINEAR_OFFSET, mild boost for BOOLEAN_BOUNDARY
        if (ftype == FailureType::LINEAR_OFFSET)      score = std::max(score, config::SCORE_CW_LINEAR_BOOST);
        else if (ftype == FailureType::BOOLEAN_BOUNDARY) score = std::max(score, config::SCORE_CW_BOOLEAN_BOOST);
        candidates.push_back({std::move(cw), score});
    }

    // --- Candidate 2: IFELSE_BOUNDARY_SPLIT ---
    if (!diag.is_constant_output && diag.targets.size() >= 2) {
        Hypothesis ibs;
        ibs.type = Hypothesis::IFELSE_BOUNDARY_SPLIT;

        // Choose condition source FIRST (prefer INPUT) 鈥?needed to compute
        // the threshold in the correct value space.
        if (!blackboard.empty()) {
            for (const auto& bs : blackboard) {
                if (bs.is_input) {
                    ibs.condition_source_node = bs.node_id;
                    break;
                }
            }
            if (ibs.condition_source_node == 0) {
                ibs.condition_source_node = blackboard[0].node_id;
            }
        } else {
            ibs.condition_source_node = 0;
        }

        // Threshold = median of CONDITION SOURCE values, not targets.
        // IFELSE compares condition_source > threshold, so the threshold
        // must be in the condition source's space (e.g., x 鈭?[-5,5]).
        // Using the target median places the boundary at the wrong position
        // (e.g., median of y 鈭?[-10,3] 鈮?-3.5 instead of x boundary at 0).
        bool threshold_found = false;
        bool zero_plateau_hit = false;
        std::vector<std::pair<Value, bool>> zp_edges;   // (thr, mask_above), capped
        if (ibs.condition_source_node != 0) {
            // SET-GUIDED SPLIT: derive the guard region {x | x <= t} from
            // where the residual changes character, not from the value
            // median. Pairs (condition value, residual) are index-aligned
            // in diag (same strided sample set). Falls back to median when
            // pairing is unavailable or the split finds no reduction.
            std::vector<std::pair<Value, Value>> vx;
            vx.reserve(diag.targets.size());
            for (size_t i = 0; i < diag.targets.size() && i < diag.local_inputs.size(); ++i) {
                auto it = diag.local_inputs[i].find(ibs.condition_source_node);
                if (it != diag.local_inputs[i].end()) {
                    vx.emplace_back(it->second, diag.targets[i]);
                }
            }
            // v2: exclude neighborhoods of committed thresholds (the graph's
            // ifelse_threshold/mux_threshold constants) so sequential
            // commits find the NEXT boundary.
            std::vector<Value> committed_thr;
            Value cond_min = 0.0, cond_max = 0.0; bool cond_ranged = false;
            if (!vx.empty()) {
                cond_min = vx.front().first; cond_max = vx.front().first;
                for (auto& pr : vx) {
                    cond_min = std::min(cond_min, pr.first);
                    cond_max = std::max(cond_max, pr.first);
                }
                cond_ranged = true;
            }
            for (const auto& n : graph_->get_nodes()) {
                const std::string& nm = n->get_name();
                if ((nm == "ifelse_threshold" || nm == "mux_threshold")
                    && n->get_type() == NodeType::CONSTANT) {
                    committed_thr.push_back(static_cast<const ConstantNode*>(n.get())->get_value());
                }
            }
            Value excl_margin = cond_ranged ? (cond_max - cond_min) * 0.05 : 0.0;
            // M5.7 ZERO-PLATEAU detection: see detect_zero_plateau_edges.
            // Label-space flat-run edges override the residual-space CART
            // split when present (windowed/dead-zone targets).
            zp_edges = detect_zero_plateau_edges(ibs.condition_source_node, *graph_);
            if (!zp_edges.empty()) {
                ibs.split_threshold = zp_edges[0].first;
                ibs.mask_above = zp_edges[0].second;
                threshold_found = true;
                zero_plateau_hit = true;
                ibs.structural_evidence = true;
                Logger::info("Zero-plateau boundary: thr=" + std::to_string(zp_edges[0].first)
                            + " mask_" + (zp_edges[0].second ? "above" : "below")
                            + " (" + std::to_string(zp_edges.size()) + " edges)");
            }
            Value guided_thr = 0.0;
            double reduction = zero_plateau_hit
                ? 0.0
                : error_weighted_split(vx, guided_thr, committed_thr, excl_margin);
            if (!zero_plateau_hit && reduction > 0.0) {
                ibs.split_threshold = guided_thr;
                threshold_found = true;
                Logger::info("IFELSE set-guided split: thr=" + std::to_string(guided_thr)
                            + " (SSE reduction " + std::to_string(reduction)
                            + " over " + std::to_string(vx.size()) + " paired samples)");
            } else if (!zero_plateau_hit) {
                auto reg_it = blackboard_registry_.find(ibs.condition_source_node);
                if (reg_it != blackboard_registry_.end() && reg_it->second.size() >= 2) {
                    std::vector<Value> src_sorted = reg_it->second;
                    std::sort(src_sorted.begin(), src_sorted.end());
                    ibs.split_threshold = src_sorted[src_sorted.size() / 2];
                    threshold_found = true;
                }
            }
        }
        if (!threshold_found) {
            std::vector<Value> sorted_t = diag.targets;
            std::sort(sorted_t.begin(), sorted_t.end());
            ibs.split_threshold = sorted_t[sorted_t.size() / 2];
        }

        // Compute gap-ratio from sorted targets (scoring only)
        std::vector<Value> sorted = diag.targets;
        std::sort(sorted.begin(), sorted.end());
        double gap_ratio = 0.0;
        if (sorted.back() - sorted.front() > config::GRADIENT_ZERO_THRESHOLD) {
            double max_gap = 0.0;
            for (size_t i = 1; i < sorted.size(); ++i) {
                max_gap = std::max(max_gap, sorted[i] - sorted[i - 1]);
            }
            gap_ratio = max_gap / (sorted.back() - sorted.front());
        }

        // Base score from gap ratio; boost for BOOLEAN_BOUNDARY
        double score = config::SCORE_IFELSE_BASE + config::SCORE_IFELSE_GAP_WEIGHT * gap_ratio;
        if (ftype == FailureType::BOOLEAN_BOUNDARY)      score = std::max(score, config::SCORE_IFELSE_BOOLEAN_BOOST);
        // M5.4: piecewise regime — the boundary family owns this residual
        if (piecewise_signature)                            score = std::max(score, 0.94);
        // M5.7: label-space plateau edges are near-exact boundaries
        if (zero_plateau_hit)                               score = std::max(score, config::SCORE_IFELSE_PLATEAU_BOOST);

        // Emit the PRESERVE variant (both-branches) as a sibling candidate
        // in the piecewise regime: for striped residuals both sides carry
        // structure, so separating them beats masking one away.
        // v3 MULTI-SPLIT: K boundaries at once (greedy tree) — single
        // splits move ~1/K of the loss, below the strict gate; the atomic
        // tree gives validation the full reduction to judge.
        if ((piecewise_signature || zero_plateau_hit) && threshold_found) {
            // M5.7: in the plateau regime the label-space edges ARE the
            // boundaries — skip CART re-derivation (which re-finds the
            // same edges noisily) and use them directly.
            std::vector<Value> ks;
            if (zero_plateau_hit && zp_edges.size() >= 2) {
                for (const auto& ze : zp_edges) ks.push_back(ze.first);
            } else {
            // Build the (input, residual) pairs for multi-split.
            std::vector<std::pair<Value, Value>> vx_m;
            for (size_t i = 0; i < diag.targets.size() && i < diag.local_inputs.size(); ++i) {
                if (!diag.local_inputs[i].empty()) {
                    auto it = diag.local_inputs[i].find(ibs.condition_source_node);
                    if (it == diag.local_inputs[i].end()) {
                        vx_m.emplace_back(diag.local_inputs[i].begin()->second, diag.targets[i]);
                    } else {
                        vx_m.emplace_back(it->second, diag.targets[i]);
                    }
                }
            }
            ks = error_weighted_multi_split(vx_m, 4);
            }
            Logger::verbose("[MULTI-DBG] vx_m.size()=" + std::to_string(ks.size())
                          + " -> K found=" + std::to_string(ks.size()));
            if (ks.size() >= 2) {
                Hypothesis ibm = ibs;
                ibm.type = Hypothesis::IFELSE_PRESERVE;
                ibm.compound_K = static_cast<int>(ks.size());
                // Store the extra thresholds... hypothesis struct has one
                // split_threshold. Reuse: first threshold in the field, the
                // rest re-derived by routing from the same pairs? No —
                // deterministic re-derivation is fragile. Simplest correct:
                // pack into sin_freq_init unused fields is ugly. Instead:
                // routing recomputes the SAME multi-split (same data, same
                // deterministic helper, margin excludes committed graph
                // thresholds — matching because emission excluded them too).
                ibm.split_threshold = ks[0];
                candidates.push_back({std::move(ibm), 0.96});   // above single-split
                Logger::info("Candidate emitted: IFELSE_PRESERVE multi-split K="
                            + std::to_string(ks.size()) + " (first thr="
                            + std::to_string(ks[0]) + ")");
            }
            Hypothesis ibp = ibs;   // single-split variant stays available
            ibp.type = Hypothesis::IFELSE_PRESERVE;
            ibp.compound_K = 1;
            candidates.push_back({std::move(ibp), 0.95});   // top of boundary family
            Logger::info("Candidate emitted: IFELSE_PRESERVE (both-branches split, thr="
                        + std::to_string(ibs.split_threshold) + ")");
        }
        else if (ftype == FailureType::LINEAR_OFFSET)     score = std::max(score, config::SCORE_IFELSE_LINEAR_BOOST);

        // Family-switch fatigue (stripes20 lesson): when MANY IFELSE commits
        // have already landed (count them in the graph) the incremental-
        // assembly path is stalling — one-boundary-per-commit can't reach a
        // 20-region target before the budget ends. Demote IFELSE so other
        // families (stacks for region boundaries, SIN for periodic stripes)
        // get validated. Heuristic: each 5 committed IFELSE nodes cost 0.1.
        {
            int ifelse_count = 0;
            for (const auto& n : graph_->get_nodes()) {
                if (n->get_type() == NodeType::IFELSE) ++ifelse_count;
            }
            if (ifelse_count >= 5) {
                double fatigue = 0.1 * static_cast<double>((ifelse_count - 5) / 5 + 1);
                score = std::max(0.05, score - fatigue);
                Logger::verbose("IFELSE fatigue: " + std::to_string(ifelse_count)
                               + " committed, score -" + std::to_string(fatigue));
            }
        }
        uint64_t ibs_cond_src_cache = ibs.condition_source_node;
        candidates.push_back({std::move(ibs), score});

        // M5.7: additional plateau edges as separate boundary candidates —
        // a windowed function needs BOTH edges; each gets its own shadow.
        for (size_t zei = 1; zei < zp_edges.size(); ++zei) {
            Hypothesis ibs2;
            ibs2.type = Hypothesis::IFELSE_BOUNDARY_SPLIT;
            ibs2.condition_source_node = ibs_cond_src_cache;
            ibs2.split_threshold = zp_edges[zei].first;
            ibs2.mask_above = zp_edges[zei].second;
            ibs2.structural_evidence = true;
            candidates.push_back({std::move(ibs2), score});
            Logger::info("Zero-plateau sibling boundary: thr="
                        + std::to_string(zp_edges[zei].first)
                        + " mask_" + (zp_edges[zei].second ? "above" : "below"));
        }
    }

    // --- Candidate 3: NEURON_TANH_INJECTION 鈥?always viable fallback ---
    {
        Hypothesis nti;
        nti.type = Hypothesis::NEURON_TANH_INJECTION;
        double score = config::SCORE_NTI_BASE;
        // Boost for NON_LINEAR_CURVE 鈥?it's the best strategy here
        if (ftype == FailureType::NON_LINEAR_CURVE) score = config::SCORE_NTI_NONLINEAR_BOOST;
        candidates.push_back({std::move(nti), score});
    }

    // --- Candidate 3b: DEEP_INSERTION 鈥?residual depth (hierarchical features) ---
    {
        Node* fn = graph_->get_node(diag.failing_node);
        bool failing_is_trainable = fn && (fn->get_type() == NodeType::NEURON
                                           || fn->get_type() == NodeType::LINEAR);
        if (failing_is_trainable) {
            Hypothesis di;
            di.type = Hypothesis::DEEP_INSERTION;
            double score = config::SCORE_DEEP_INSERTION;
            if (ftype == FailureType::NON_LINEAR_CURVE)
                score = std::max(score, config::SCORE_DEEP_INSERTION * 1.4);
            candidates.push_back({std::move(di), score});
        }

        // --- Candidate 3c: MULTI_LAYER_STACK 鈥?2-layer MLP for hard boundaries ---
        // Injects K parallel hidden neurons + a combining neuron, replacing
        // the single failing neuron with a proper hidden layer. Needed for
        // problems like spirals/checkerboard where no single-layer architecture
        // can capture the decision boundary. Emitted when the profile shows
        // high complexity (sharp boundaries, high variance ratio).
        if (failing_is_trainable) {
            // Count graph INPUT nodes to limit cost
            int graph_input_count = 0;
            for (const auto& n : graph_->get_nodes()) {
                if (n->get_type() == NodeType::INPUT) ++graph_input_count;
            }
            bool high_complexity = profile.lipschitz_max > config::PROFILE_SHARP_BOUNDARY_LIPSCHITZ
                                 || (ftype == FailureType::NON_LINEAR_CURVE
                                     && profile.var_r > 0.3
                                     && profile.poly_r2 < 0.9);
             // Stack-dominance gate (I.32.8 lesson): when the degree-2 fit
             // fails with a dominant cross-term, the residual signature is
             // product/quotient-shaped, NOT curve-capacity-shaped. Stacks
             // still get emitted (generic capacity), but the extreme-
             // lipschitz boost is suppressed so feature hypotheses
             // (DIVIDE_PRODUCT, MULTIPLY) can win the validation race.
             bool product_signature = profile.poly_r2 < 0.95
                                   && (profile.interaction_dominant
                                       || profile.max_coef_index >= 1 + profile.num_inputs);
             // Memory-signature gate (NARMA-30 lesson): in sequence mode with
             // few inputs, plateau means "missing temporal structure" — the
             // recurrent family (MULTI_TAP, SELF_WIRE) owns that domain,
             // whether or not recurrence exists yet. Suppress the stack
             // boost and demote stacks below recurrence scores.
             bool memory_signature = cfg_.sequence_mode && profile.num_inputs <= 3;
             if (high_complexity && graph_input_count <= config::MULTI_LAYER_STACK_MAX_INPUTS) {
                 Hypothesis mls;
                 mls.type = Hypothesis::MULTI_LAYER_STACK;
                 mls.compound_K = config::MULTI_LAYER_STACK_K;
                 double score = config::SCORE_MULTI_LAYER_STACK;
                 if (profile.lipschitz_max > config::MULTI_LAYER_STACK_BOOST_LIPSCHITZ
                     && !product_signature && !memory_signature && !piecewise_signature) {
                     score = 0.95;  // extreme complexity → rank above everything
                     mls.compound_K = config::MULTI_LAYER_STACK_K_MAX;
                 }
                // In the memory-signature regime, actively DEMOTE stacks below
                // recurrence hypotheses so MULTI_TAP/DELAY_LINE get validated
                // first (narma30 lesson: sin-product at 0.96 also outranked
                // the memory family and won every race while the model
                // stayed at mean prediction).
                if (memory_signature) {
                    score = std::min(score, config::SCORE_DELAY_LINE - 0.06);
                }
                // Piecewise regime: stacks cannot represent many sharp
                // boundaries — demote below the boundary family.
                if (piecewise_signature) {
                    score = std::min(score, 0.45);
                }
                candidates.push_back({std::move(mls), score});
                Logger::info("Candidate emitted: MULTI_LAYER_STACK (lipschitz="
                            + std::to_string(profile.lipschitz_max)
                            + " var_r=" + std::to_string(profile.var_r) + ")");
            }
        }

        // --- Candidate 3d: PATCH_POOLING 鈥?coarse convolutional prior ---
        // --- Candidate 3e: MUX_INJECTION — select between two signals ---
        // MUX(cond, a, b): piecewise/regime targets (stripes, cliffs,
        // switching behavior). Condition = top blackboard INPUT thresholded
        // via GREATER (MUX's execute uses !=0 truthiness; raw continuous
        // values would misroute); branches a,b = next two signals. Zero-
        // gain NEURON on the output gives identity start.
        if (failing_is_trainable && blackboard.size() >= 3) {
            // Condition source: prefer an INPUT (raw feature, thresholdable);
            // fall back to any blackboard signal. Branches: next two signals
            // (INPUTs first, then neurons — stripes20 has ONE input but a
            // rich blackboard of internal features after the first commits).
            std::vector<uint64_t> cond_pool, branch_pool;
            for (const auto& bs : blackboard) {
                if (bs.is_input) cond_pool.push_back(bs.node_id);
                branch_pool.push_back(bs.node_id);
            }
            uint64_t cond = 0;
            if (!cond_pool.empty()) cond = cond_pool[0];
            else cond = blackboard[0].node_id;
            // branches: first two blackboard signals that aren't the cond
            std::vector<uint64_t> srcs;
            for (uint64_t s : branch_pool) {
                if (s != cond) srcs.push_back(s);
                if (srcs.size() >= 2) break;
            }
            if (srcs.size() >= 2) {
                // SET-GUIDED threshold: same CART-style split as IFELSE —
                // the MUX selection boundary comes from where the residual
                // variance partitions, not the value median.
                Value thr = 0.0;
                {
                    std::vector<std::pair<Value, Value>> vx;
                    vx.reserve(diag.targets.size());
                    for (size_t i2 = 0; i2 < diag.targets.size() && i2 < diag.local_inputs.size(); ++i2) {
                        auto it = diag.local_inputs[i2].find(cond);
                        if (it != diag.local_inputs[i2].end()) {
                            vx.emplace_back(it->second, diag.targets[i2]);
                        }
                    }
                    double reduction = error_weighted_split(vx, thr);
                    if (reduction <= 0.0) {
                        auto rit = blackboard_registry_.find(cond);
                        if (rit != blackboard_registry_.end() && rit->second.size() >= 2) {
                            std::vector<Value> sorted_v = rit->second;
                            std::sort(sorted_v.begin(), sorted_v.end());
                            thr = sorted_v[sorted_v.size() / 2];
                        }
                    }
                }
                Hypothesis mx;
                mx.type = Hypothesis::MUX_INJECTION;
                mx.multiply_source_a = srcs[0];   // branch a
                mx.multiply_source_b = srcs[1];   // branch b
                mx.bool_source_a    = cond;       // condition source
                mx.split_threshold  = thr;
                double mx_score = config::SCORE_MUX_INJECTION;
                if (piecewise_signature) {
                    mx_score = config::SCORE_MUX_INJECTION_BOOST + 0.02;
                } else if (ftype == FailureType::BOOLEAN_BOUNDARY
                    || profile.lipschitz_max > config::PROFILE_SHARP_BOUNDARY_LIPSCHITZ) {
                    mx_score = config::SCORE_MUX_INJECTION_BOOST;
                }
                candidates.push_back({std::move(mx), mx_score});
                Logger::info("Candidate emitted: MUX_INJECTION (cond=node="
                            + std::to_string(cond) + " thr=" + std::to_string(thr)
                            + " a=node=" + std::to_string(srcs[0])
                            + " b=node=" + std::to_string(srcs[1]) + ")");
            }
        }
        // For image-like input layouts (input count a perfect square 鈮?        // PATCH_POOL_MIN_SIDE虏, or 3脳square for pixel-interleaved RGB),
        // --- M2.1: EMBED_TRUNK — shared dense trunk for many-output tasks ---
        // Language-model signature: MANY outputs (>= 8) at plateau. The
        // complexity profile was built for regression residuals and reads
        // flat on LM loss curves, so gate on the output count instead.
        // One trunk per graph (dedup on trunk node name).
        {
            size_t out_count = 0;
            bool has_trunk = false;
            for (const auto& n : graph_->get_nodes()) {
                if (n->get_type() == NodeType::OUTPUT) ++out_count;
                else if (n->get_name() == "embed_trunk_combine") has_trunk = true;
            }
            if (out_count >= 8 && !has_trunk) {
                Hypothesis et;
                et.type = Hypothesis::EMBED_TRUNK;
                et.compound_K = config::EMBED_TRUNK_K;
                candidates.push_back({std::move(et), config::SCORE_EMBED_TRUNK});
                Logger::info("Candidate emitted: EMBED_TRUNK (K="
                            + std::to_string(config::EMBED_TRUNK_K)
                            + ", " + std::to_string(out_count) + " outputs)");
                // M2.3 step 2: the attention head as a SIBLING candidate —
                // dense mixing (EMBED) and retrieval (ATTENTION) compete in
                // validation; the induction probe measured dense mixing at
                // chance (7-8% vs 6.25%), so tasks needing retrieval should
                // pick this one. Emitted regardless of has_trunk state: a
                // graph can carry both a dense trunk and a retrieval head.
            }
            // Attention head emission (dedup: one per graph by name).
            {
                bool has_attn = false;
                size_t in_count = 0;
                for (const auto& n : graph_->get_nodes()) {
                    if (n->get_type() == NodeType::ATTENTION) has_attn = true;
                    else if (n->get_type() == NodeType::INPUT) ++in_count;
                }
                if (out_count >= 8 && in_count >= 4 && !has_attn) {
                    Hypothesis at;
                    at.type = Hypothesis::ATTENTION_MIX;
                    // M2.3 evidence-class: once emitted, the head's table
                    // is SEEDED from the trunk's learned embeddings (when a
                    // one-hot trunk exists) — its structure is derived from
                    // measurement, not sampled. The evidence flag activates
                    // the neutral-tolerance commit gate + race advantage
                    // (the t22 playbook: both budget and seed alone measured
                    // insufficient — the head loses validation races to
                    // micro-gain commits it will eventually dominate).
                    at.structural_evidence = true;
                    candidates.push_back({std::move(at), config::SCORE_EMBED_TRUNK - 0.01});
                    Logger::info("Candidate emitted: ATTENTION_MIX ("
                                + std::to_string(in_count) + " slots, V="
                                + std::to_string(out_count) + ")");
                }
            }
        }

        // inject one average-pool LINEAR node per patch_size虏 block.
        // Uniform 1/k weights = exact block mean; SGD refines them into
        // learned filters. Pooled features wire into the failing node at
        // zero-init (identity start). Skipped when pool nodes already
        // exist (dedup 鈥?one pooling layer per graph).
        if (failing_is_trainable) {
            int img_inputs = 0;
            bool has_pool = false;
            for (const auto& n : graph_->get_nodes()) {
                if (n->get_type() == NodeType::INPUT) ++img_inputs;
                else if (n->get_name().rfind("pool_", 0) == 0) has_pool = true;
            }
            int channels = 1;
            int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(img_inputs))));
            if (side * side != img_inputs) {
                // Try pixel-interleaved RGB: input count == 3路side虏
                int s3 = static_cast<int>(std::lround(std::sqrt(img_inputs / 3.0)));
                if (3 * s3 * s3 == img_inputs) { side = s3; channels = 3; }
            }
            bool image_like = !has_pool
                           && side >= config::PATCH_POOL_MIN_SIDE
                           && channels * side * side == img_inputs
                           && side % config::PATCH_POOL_PATCH_SIZE == 0;
            if (image_like) {
                Hypothesis pp;
                pp.type = Hypothesis::PATCH_POOLING;
                pp.compound_K = config::PATCH_POOL_PATCH_SIZE;
                candidates.push_back({std::move(pp), config::SCORE_PATCH_POOLING});
                Logger::info("Candidate emitted: PATCH_POOLING (" + std::to_string(side)
                            + "x" + std::to_string(side) + " image"
                            + (channels == 3 ? ", RGB" : "")
                            + ", " + std::to_string(config::PATCH_POOL_PATCH_SIZE) + "px patches)");
            }
        }
    }

    // --- Candidate 4: MULTIPLY_INJECTION 鈥?interaction / polynomial features ---
    // Adds a product feature A*B (or A*A = A虏 if only one strong signal).
    // Useful for: x*y interactions, x虏 curvature, quadratic surfaces,
    // building blocks for circular boundaries (x虏+y虏 via repeated commits).
    if (!blackboard.empty()) {
        Hypothesis mi;
        mi.type = Hypothesis::MULTIPLY_INJECTION;

        // Interaction-aware source selection.
        //
        // For an interaction target (y 鈮?x_i路x_j or x_i^2), each marginal
        // input has ~zero Pearson correlation with the target (positives and
        // negatives cancel), so the blackboard 鈥?ranked by single-feature
        // |r| 鈥?orders the relevant inputs arbitrarily. Two robust signals
        // identify the right pair:
        //   1. interaction_dominant 鈥?the degree-2 fit's dominant non-bias
        //      term is a square/cross term, decoded to interact_a/interact_b.
        //      Robust even on low-variance residuals where Sobol is noisy
        //      (e.g. d4: 12 inputs, y=x0*x1).
        //   2. high_pairwise_interaction 鈥?Sobol pairwise index exceeds
        //      threshold (sobol_pair_a/b).
        // Prefer (1); fall back to (2); else blackboard order below.
        std::vector<uint64_t> graph_input_ids;
        for (const auto& node : graph_->get_nodes()) {
            if (node->get_type() == NodeType::INPUT) graph_input_ids.push_back(node->get_id());
        }
        bool sobol_pair_used = false;
        if (profile.interaction_dominant && !graph_input_ids.empty()) {
            size_t idx_a = std::min(profile.interact_a, graph_input_ids.size() - 1);
            size_t idx_b = std::min(profile.interact_b, graph_input_ids.size() - 1);
            // NOTE: do NOT force idx_a != idx_b here. A dominant square term
            // (x_i^2) is a GENUINE self-product signal 鈥?overriding it to a
            // cross-product picks the wrong feature (regressed t31). The
            // self-product is supported by MULTIPLY_INJECTION routing.
            mi.multiply_source_a = graph_input_ids[idx_a];
            mi.multiply_source_b = graph_input_ids[idx_b];
            sobol_pair_used = true;
            Logger::info("MULTIPLY sources from poly fit: input="
                         + std::to_string(mi.multiply_source_a)
                         + (idx_a == idx_b ? "^2" : ("*input=" + std::to_string(mi.multiply_source_b)))
                         + " (max_coef_idx=" + std::to_string(profile.max_coef_index) + ")");
        } else if (profile.high_pairwise_interaction && graph_input_ids.size() >= 2) {
            size_t idx_a = std::min(profile.sobol_pair_a, graph_input_ids.size() - 1);
            size_t idx_b = std::min(profile.sobol_pair_b, graph_input_ids.size() - 1);
            if (idx_a == idx_b) idx_b = (idx_a + 1) % graph_input_ids.size();
            mi.multiply_source_a = graph_input_ids[idx_a];
            mi.multiply_source_b = graph_input_ids[idx_b];
            sobol_pair_used = true;
            Logger::info("MULTIPLY sources from Sobol pair: input="
                         + std::to_string(mi.multiply_source_a) + "*input="
                         + std::to_string(mi.multiply_source_b)
                         + " (sobol_pairwise=" + std::to_string(profile.sobol_pairwise) + ")");
        }

        // Structural prior: prefer INPUT nodes as MULTIPLY sources.
        // The blackboard ranks by |correlation|, but for products, raw
        // inputs (x1*x2) are almost always more meaningful than products
        // of intermediate signals (e.g., neuron*neuron). After the first
        // NEURON commit, the NEURON's output typically dominates the
        // blackboard by correlation, but multiplying two NEURON outputs
        // is rarely what the problem needs 鈥?we want raw feature crosses.
        // Fall back to top-2-by-correlation only if fewer than 2 INPUTs
        // are present in the blackboard.
        if (!sobol_pair_used) {
            std::vector<uint64_t> input_signals;
            for (const auto& bs : blackboard) {
                if (bs.is_input) input_signals.push_back(bs.node_id);
                if (input_signals.size() >= 2) break;
            }
            if (input_signals.size() >= 2) {
                mi.multiply_source_a = input_signals[0];
                mi.multiply_source_b = input_signals[1];
            } else if (input_signals.size() == 1) {
                // One INPUT available 鈥?pair it with the top-correlated signal
                // (which may be another INPUT or a useful intermediate).
                mi.multiply_source_a = input_signals[0];
                mi.multiply_source_b = (blackboard[0].node_id != input_signals[0])
                                     ? blackboard[0].node_id
                                     : (blackboard.size() >= 2 ? blackboard[1].node_id
                                                               : blackboard[0].node_id);
            } else {
                // No INPUT signals in blackboard 鈥?fall back to top-2 correlation
                mi.multiply_source_a = blackboard[0].node_id;
                if (blackboard.size() >= 2) {
                    mi.multiply_source_b = blackboard[1].node_id;
                } else {
                    mi.multiply_source_b = blackboard[0].node_id;  // x * x
                }
            }
        }

        double score = config::SCORE_MULTIPLY_BASE;  // baseline 鈥?experimental, ranks BELOW proven NEURON_TANH_INJECTION (0.1)
        // Boost for NON_LINEAR_CURVE 鈥?products capture curvature that
        // a single tanh neuron struggles with (e.g., x虏 outside [-1,1])
        if (ftype == FailureType::NON_LINEAR_CURVE) score = std::max(score, config::SCORE_MULTIPLY_NONLINEAR_BOOST);
        // Boost for LINEAR_OFFSET 鈥?many interaction-term problems
        // (y=x1*x2) get misclassified as LINEAR_OFFSET because the local
        // linear fit at the failing node looks okay-ish. The missing
        // feature is genuinely a product, not another linear signal.
        if (ftype == FailureType::LINEAR_OFFSET) score = std::max(score, config::SCORE_MULTIPLY_NONLINEAR_BOOST);
        // Boost when both top signals correlate strongly AND the target is
        // continuous (interaction-term signal). Skip for BOOLEAN_BOUNDARY 鈥?        // products don't solve binary classification (e.g., XOR needs
        // x1+x2-2路x1路x2, not just x1路x2).
        if (ftype != FailureType::BOOLEAN_BOUNDARY && blackboard.size() >= 2) {
            double r1 = std::abs(blackboard[0].correlation);
            double r2 = std::abs(blackboard[1].correlation);
            if (r1 > config::MULTIPLY_CORR_TRIGGER && r2 > config::MULTIPLY_CORR_TRIGGER) score = std::max(score, config::SCORE_MULTIPLY_INTERACTION_BOOST);
        }

        // Plateau fallback: symmetric polynomial features (y=x虏 on symmetric
        // input, y=x1路x2 over 4 quadrants, y=x1虏+x2虏) have ~0 Pearson r
        // with the target because positive and negative contributions cancel,
        // so the correlation-trigger above never fires 鈥?but the INPUT
        // signals themselves are clearly visible in the blackboard. When
        // 鈮? INPUT signals exist and the problem isn't binary classification
        // (where BOOLEAN_COMPOSE is the right tool), the missing feature is
        // very likely a product. Boost above CONTEXT_WIRE (0.85) so MULTIPLY
        // gets tried before another linear wire commit bloats the graph.
        if (!is_binary && ftype != FailureType::BOOLEAN_BOUNDARY && input_signal_count >= 2) {
            score = std::max(score, config::SCORE_MULTIPLY_PLATEAU_FALLBACK);
        }

        // M7.6(c): quadrant-means interaction gate. The profile computes
        // per-quadrant target means (top-2 inputs' 4 quadrants) but the
        // emission never read them — "means differ strongly across
        // quadrants" is the natural 2-input interaction signal: an
        // interaction-only target (y ~ x1*x2) has near-zero marginal
        // structure but strongly asymmetric quadrant means. Boost the
        // product family when the quadrant contrast is large relative to
        // the residual's spread.
        if (profile.quadrant_means.size() == 4 && profile.num_inputs >= 2
            && !is_binary) {
            Value qmin = profile.quadrant_means[0], qmax = qmin;
            for (Value q : profile.quadrant_means) {
                qmin = std::min(qmin, q);
                qmax = std::max(qmax, q);
            }
            Value spread = std::max(profile.var_r > 1e-12
                                        ? std::sqrt(profile.var_r) * 2.0
                                        : static_cast<Value>(1e-6),
                                    static_cast<Value>(1e-6));
            Value contrast = (qmax - qmin) / spread;
            if (contrast > static_cast<Value>(0.5)) {
                score = std::max(score, config::SCORE_MULTIPLY_PLATEAU_FALLBACK);
                Logger::info("Quadrant interaction gate: contrast "
                            + std::to_string(contrast) + " boosts MULTIPLY");
            }
        }

        // Dedup: if the graph already has a MULTIPLY node with the same
        // source pair, try the OTHER input's self-product. This fixes t23
        // (y=x0虏+x1虏) where the poly fit keeps recommending the same
        // feature after the first commit.
        auto multiply_exists = [&](uint64_t a, uint64_t b) -> bool {
            for (const auto& conn_pair : graph_->get_connections()) {
                if (conn_pair.dst_node == 0) continue;
                const Node* n = graph_->get_node(conn_pair.dst_node);
                if (!n || n->get_type() != NodeType::MULTIPLY) continue;
                // Check if this MULTIPLY's inputs match (a,b) in either order
                uint64_t in0 = 0, in1 = 0;
                for (const auto& c2 : graph_->get_connections()) {
                    if (c2.dst_node == n->get_id()) {
                        if (c2.dst_port == 0) in0 = c2.src_node;
                        else if (c2.dst_port == 1) in1 = c2.src_node;
                    }
                }
                if ((in0 == a && in1 == b) || (in0 == b && in1 == a)) return true;
            }
            return false;
        };

        if (multiply_exists(mi.multiply_source_a, mi.multiply_source_b)) {
            // Try the other input's self-product
            bool found_alt = false;
            for (uint64_t gid : graph_input_ids) {
                if (gid != mi.multiply_source_a && gid != mi.multiply_source_b) {
                    if (!multiply_exists(gid, gid)) {
                        mi.multiply_source_a = gid;
                        mi.multiply_source_b = gid;
                        found_alt = true;
                        Logger::info("MULTIPLY dedup: switching to input=" + std::to_string(gid) + "^2");
                        break;
                    }
                }
            }
            // Also try swapping if only one input exists (self was used, try cross)
            if (!found_alt && graph_input_ids.size() >= 2) {
                for (uint64_t g1 : graph_input_ids) {
                    for (uint64_t g2 : graph_input_ids) {
                        if (g1 >= g2) continue;  // avoid duplicates
                        if (!multiply_exists(g1, g2)) {
                            mi.multiply_source_a = g1;
                            mi.multiply_source_b = g2;
                            found_alt = true;
                            Logger::info("MULTIPLY dedup: switching to input=" + std::to_string(g1)
                                        + "*input=" + std::to_string(g2));
                            break;
                        }
                    }
                    if (found_alt) break;
                }
            }
        }

        // DIVIDE_INJECTION: quotient features a/b and b/a with the SAME
        // sources the MULTIPLY candidate selected. Ratio targets (m/V,
        // (u+v)/(1+uv)) look identical to product targets in the poly fit
        // (both are non-polynomial), so both operator hypotheses are
        // emitted adjacently and validation arbitrates.
        {
            uint64_t da = mi.multiply_source_a;
            uint64_t db = mi.multiply_source_b;
            if (da != 0 && db != 0 && da != db) {
                // Reverse map: graph node id 鈫?data key, for reading raw
                // training-space denominator values.
                std::unordered_map<uint64_t, uint64_t> graph_to_data;
                for (const auto& kv : input_data_to_graph_) {
                    graph_to_data[kv.second] = kv.first;
                }
                auto da_it = graph_to_data.find(da);
                auto db_it = graph_to_data.find(db);
                bool have_vals = (da_it != graph_to_data.end() && db_it != graph_to_data.end());

                // Denominator safety: min |values| must clear
                // max(ABS_FLOOR, MIN_REL 脳 std) over the training set.
                auto denom_safe = [&](uint64_t den_data_key) -> bool {
                    if (!have_vals) return false;
                    double mn = 1e18, sq = 0.0, sum = 0.0;
                    int n = 0;
                    for (const auto& s : training_data_.samples) {
                        auto it = s.inputs.find(den_data_key);
                        if (it == s.inputs.end()) continue;
                        double v = it->second;
                        mn = std::min(mn, std::abs(v));
                        sum += v; sq += v * v; ++n;
                    }
                    if (n < 4 || mn > 1e17) return false;
                    double mean = sum / n;
                    double stdev = std::sqrt(std::max(0.0, sq / n - mean * mean));
                    return mn >= std::max(config::DIVIDE_DENOMINATOR_ABS_FLOOR,
                                         config::DIVIDE_DENOMINATOR_MIN_REL * stdev);
                };

                // Dedup: skip an ordering already present in the graph.
                auto divide_exists = [&](uint64_t num, uint64_t den) -> bool {
                    for (const auto& c : graph_->get_connections()) {
                        const Node* n = graph_->get_node(c.dst_node);
                        if (!n || n->get_type() != NodeType::DIVIDE) continue;
                        uint64_t in0 = 0, in1 = 0;
                        for (const auto& c2 : graph_->get_connections()) {
                            if (c2.dst_node == c.dst_node) {
                                if (c2.dst_port == 0) in0 = c2.src_node;
                                else if (c2.dst_port == 1) in1 = c2.src_node;
                            }
                        }
                        if ((in0 == num && in1 == den) || (in0 == den && in1 == num)) return true;
                    }
                    return false;
                };

                // Demote divide-family ONLY when a degree-2 fit actually
                // explains the residual (poly_r2 >= 0.95). When the fit
                // fails (I.32.8 q虏a虏/c鲁: 0.937 鈥?needs 1/c鲁), a quotient is
                // precisely what's missing; LINEAR_OFFSET classification is
                // then an artifact of the strong linear numerator term.
                double dscore = config::SCORE_DIVIDE_BOOST;
                if (ftype != FailureType::NON_LINEAR_CURVE
                    && ftype != FailureType::LINEAR_OFFSET) {
                    dscore = config::SCORE_DIVIDE_BASE;
                } else if (ftype == FailureType::LINEAR_OFFSET
                           && profile.poly_r2 >= 0.95) {
                    dscore = config::SCORE_DIVIDE_BASE;
                }

                auto try_emit = [&](uint64_t num, uint64_t den) {
                    uint64_t den_key = graph_to_data[den];
                    if (!denom_safe(den_key)) {
                        Logger::verbose("DIVIDE candidate skipped: denominator near zero");
                        return;
                    }
                    if (divide_exists(num, den)) return;
                    Hypothesis dv;
                    dv.type = Hypothesis::DIVIDE_INJECTION;
                    dv.multiply_source_a = num;
                    dv.multiply_source_b = den;
                    candidates.push_back({std::move(dv), dscore});
                    Logger::info("Candidate emitted: DIVIDE_INJECTION (input="
                                + std::to_string(num) + "/input=" + std::to_string(den) + ")");
                };
                if (have_vals) {
                    try_emit(da, db);
                    try_emit(db, da);
                }

                // COMPOUND_DIVIDE_PRODUCT: pow(a*b, n)/pow(c, m) over ALL input
                // TRIPLES (pair x denominator). The profile's single pair guess
                // picked the wrong triple twice on I.32.8 ((a*c)^2/q^3 instead
                // of (q*a)^2/c^3); permutation is cheap and validation
                // arbitrates. Gated to 3..6 inputs; denom_safe per denominator.
                if (have_vals && profile.num_inputs >= 3
                    && profile.num_inputs <= 6) {
                    std::vector<std::pair<uint64_t, uint64_t>> inputs;
                    for (const auto& kv : input_data_to_graph_) {
                        if (graph_->get_node(kv.second)) inputs.push_back(kv);
                    }
                    std::vector<std::pair<int,int>> pows = {{1,1}};
                    if (profile.poly_r2 < 0.95) {
                        pows.push_back({2,3});
                        pows.push_back({2,1});
                        pows.push_back({1,2});
                    }
                    double cscore = config::SCORE_COMPOUND_DIVIDE_PRODUCT;
                    if (ftype != FailureType::NON_LINEAR_CURVE
                        && ftype != FailureType::LINEAR_OFFSET) {
                        cscore = config::SCORE_DIVIDE_BASE;
                    } else if (ftype == FailureType::LINEAR_OFFSET
                               && profile.poly_r2 >= 0.95) {
                        cscore = config::SCORE_DIVIDE_BASE;
                    }
                    for (size_t i = 0; i < inputs.size(); ++i) {
                        for (size_t j = i + 1; j < inputs.size(); ++j) {
                            for (size_t k = 0; k < inputs.size(); ++k) {
                                if (k == i || k == j) continue;
                                if (!denom_safe(inputs[k].first)) continue;
                                uint64_t pa_ = inputs[i].second;
                                uint64_t pb_ = inputs[j].second;
                                uint64_t pc_ = inputs[k].second;
                                for (auto& npair : pows) {
                                    int np = npair.first, dp = npair.second;
                                    Hypothesis cdp;
                                    cdp.type = Hypothesis::COMPOUND_DIVIDE_PRODUCT;
                                    cdp.multiply_source_a = pa_;
                                    cdp.multiply_source_b = pb_;
                                    cdp.bool_source_a     = pc_;
                                    cdp.bool_source_b     = (static_cast<uint64_t>(np) << 8)
                                                          | static_cast<uint64_t>(dp);
                                    candidates.push_back({std::move(cdp), cscore});
                                }
                                Logger::info("Candidate emitted: COMPOUND_DIVIDE_PRODUCT (input="
                                            + std::to_string(pa_) + "*input=" + std::to_string(pb_)
                                            + ")^n/input=" + std::to_string(pc_) + "^m");
                            }
                        }
                    }
                }
            }
        }

        candidates.push_back({std::move(mi), score});
    }

    // --- Candidate 5: BOOLEAN_COMPOSE 鈥?parity / multi-condition logic ---
    // Composes two signals via XOR/AND/OR (with implicit thresholding).
    //
    // Detection: we look at the GLOBAL training labels, not the local node
    // targets. classify_failure() sees back-propagated continuous targets at
    // the failing node, so binary classification problems (XOR, AND, OR)
    // don't trigger FailureType::BOOLEAN_BOUNDARY at the local level. The
    // global labels are the reliable signal that this is a binary problem.
    //
    // Sources: accept ANY two blackboard signals (not just boolean-typed
    // nodes). apply_shadow_routing() will threshold continuous sources via
    // GREATER(src, threshold) before the boolean op, since XOR/AND/OR use
    // != 0.0 truthiness which is wrong for continuous values.
    if (blackboard.size() >= 2) {
        // is_binary was computed once at the top of generate_candidates —
        // on the FIRST output only. Many-class one-hot tasks (char-LM:
        // 65 outputs) look binary per-output, and boolean composition
        // crowds out smooth statistics (observed: 5-7 BOOLEAN_COMPOSE
        // commits per char-LM run, each useless). Gate: emit only when
        // the graph has ≤4 OUTPUT nodes (true boolean problems).
        size_t output_count = 0;
        for (const auto& n : graph_->get_nodes()) {
            if (n->get_type() == NodeType::OUTPUT) ++output_count;
        }
        if (is_binary && output_count <= 4) {
            Hypothesis bc;
            bc.type           = Hypothesis::BOOLEAN_COMPOSE;
            bc.bool_source_a  = blackboard[0].node_id;
            bc.bool_source_b  = blackboard[1].node_id;
            bc.bool_op        = NodeType::XOR;  // default 鈥?most expressive for parity
            bc.bool_threshold = 0.0;
            double score = config::SCORE_BOOLEAN_COMPOSE;  // strong score 鈥?this is the right tool for binary problems
            candidates.push_back({std::move(bc), score});
        }
    }

    // --- Candidate 6: COMPOUND_MULTIPLY_NEURON 鈥?sin(xy)-class composition ---
    // Fires when the complexity profile of the residual shows:
    //   (1) A dominant polynomial cross-term a_ij (structurally meaningful
    //       relative to the linear coefficients), indicating an x_i路x_j
    //       interaction is the missing feature; AND
    //   (2) The residual is BOUNDED (max-min range is small relative to
    //       stddev), indicating the missing function is a squashing curve
    //       (sin, tanh, sigmoid) applied to that product.
    //
    // Neither MULTIPLY_INJECTION alone (unbounded product, useless without a
    // nonlinearity 鈥?fails validation) nor NEURON_TANH_INJECTION alone (no
    // interaction feature to bind to) can solve this class. The composition
    // MULTIPLY 鈫?NEURON 鈫?TANH, validated as a single unit, can.
    //
    // Skip for binary classification (BOOLEAN_COMPOSE is the right tool).
    if (!is_binary && profile.num_inputs >= 2 && profile.bounded) {
        // Identify the dominant cross-term coefficient.
        // Layout: [0]=bias, [1..F]=linear, [F+1..2F]=squares, [2F+1..]=cross terms (i<j).
        // We already track max_coef_index and max_coef_value in the profile.
        size_t F = profile.num_inputs;
        size_t cross_terms_start = 2 * F + 1;
        bool cross_term_dominant = false;
        if (profile.max_coef_index >= cross_terms_start) {
            // Find the largest |linear| coefficient to compare against.
            double max_linear = 0.0;
            for (size_t i = 0; i < F; ++i) {
                double v = std::abs(profile.poly_coeffs[1 + i]);
                if (v > max_linear) max_linear = v;
            }
            if (max_linear > 1e-9) {
                double ratio = std::abs(profile.max_coef_value) / max_linear;
                if (ratio >= config::PROFILE_COMPOUND_CROSS_TERM_RATIO) {
                    cross_term_dominant = true;
                }
            } else if (std::abs(profile.max_coef_value) > 1e-3) {
                // No meaningful linear terms but a non-trivial cross-term.
                cross_term_dominant = true;
            }
        }

        if (cross_term_dominant) {
            // Pick the two RAW INPUT nodes identified by the Sobol pairwise
            // index as having the strongest interaction. The profile already
            // computed sobol_pair_a and sobol_pair_b 鈥?these are input INDICES
            // (0-based among INPUT nodes), not graph node IDs.
            //
            // CRITICAL: we must multiply RAW INPUTS (x_i, x_j), not intermediate
            // neurons. MULTIPLY(neuron_a, neuron_b) cannot learn sin(x路y)
            // because the intermediate outputs are not the product features
            // the target depends on.
            std::vector<uint64_t> graph_input_ids;
            for (const auto& node : graph_->get_nodes()) {
                if (node->get_type() == NodeType::INPUT) {
                    graph_input_ids.push_back(node->get_id());
                }
            }
            uint64_t src_a = 0, src_b = 0;
            if (graph_input_ids.size() >= 2) {
                size_t idx_a = std::min(profile.sobol_pair_a, graph_input_ids.size() - 1);
                size_t idx_b = std::min(profile.sobol_pair_b, graph_input_ids.size() - 1);
                if (idx_a == idx_b) {
                    // Avoid multiplying an input by itself 鈥?pick adjacent
                    idx_b = (idx_a + 1) % graph_input_ids.size();
                }
                src_a = graph_input_ids[idx_a];
                src_b = graph_input_ids[idx_b];
            }
            if (src_a != 0 && src_b != 0) {
                Hypothesis compound;
                compound.type              = Hypothesis::COMPOUND_MULTIPLY_NEURON;
                compound.multiply_source_a = src_a;
                compound.multiply_source_b = src_b;
                candidates.push_back({std::move(compound), config::SCORE_COMPOUND_MULTIPLY_NEURON});
                Logger::info("Compound candidate emitted: MULTIPLY(input="
                             + std::to_string(src_a) + ",input=" + std::to_string(src_b)
                             + ")鈫扤EURON鈫扵ANH (cross-term ratio triggered, sobol_pair="
                             + std::to_string(profile.sobol_pairwise) + ")");
            }
        }
    }

    // --- Candidate 6.1: COMPOUND_MULTIPLY_ABS 鈥?|x*y|-class composition ---
    // Emitted when interaction_dominant fires AND the target is sign-symmetric
    // (essentially non-negative, like |x*y|). The non-negativity gate is what
    // distinguishes a genuine |interaction| target from a plain signed product
    // (e.g. y = x*y*z on t31), so ABS is only attempted where it's the right
    // tool. Without this gate the candidate adds thrash to sign-varying
    // interaction tasks (t31 variance tripled) without helping them.
    if (!is_binary && profile.interaction_dominant && profile.num_inputs >= 2
        && profile.poly_r2 < 0.95                             // abs fold => poor poly fit; perfect fit (x虏) needs no abs
        && !diag.targets.empty()) {
        Value tmin = *std::min_element(diag.targets.begin(), diag.targets.end());
        Value tmax = *std::max_element(diag.targets.begin(), diag.targets.end());
        Value rng = tmax - tmin;
        bool sign_symmetric = (rng > 1e-9) ? (tmin >= -config::ABSMUL_NONNEG_FRACTION * rng)
                                           : (tmin >= -1e-6);
        if (sign_symmetric) {
            std::vector<uint64_t> graph_input_ids;
            for (const auto& node : graph_->get_nodes()) {
                if (node->get_type() == NodeType::INPUT) graph_input_ids.push_back(node->get_id());
            }
            if (graph_input_ids.size() >= 2) {
                size_t idx_a = std::min(profile.interact_a, graph_input_ids.size() - 1);
                size_t idx_b = std::min(profile.interact_b, graph_input_ids.size() - 1);
                if (idx_a == idx_b) idx_b = (idx_a + 1) % graph_input_ids.size();
                Hypothesis compound;
                compound.type              = Hypothesis::COMPOUND_MULTIPLY_ABS;
                compound.multiply_source_a = graph_input_ids[idx_a];
                compound.multiply_source_b = graph_input_ids[idx_b];
                candidates.push_back({std::move(compound), config::SCORE_COMPOUND_MULTIPLY_NEURON});
                Logger::info("Compound candidate emitted: MULTIPLY(input="
                             + std::to_string(graph_input_ids[idx_a]) + ",input="
                             + std::to_string(graph_input_ids[idx_b]) + ")鈫扐BS (sign-symmetric, interaction_dominant)");
            }
        }
    }

    // --- Candidate 6.5: COMPOUND_MULTIPLY3_NEURON 鈥?x*y*z-class 3-way product ---
    // Fires when 3+ inputs and the degree-2 polynomial fit is poor
    // (poly_r2 < threshold) because the missing feature is the 3-way product
    // x路y路z, which degree-2 can't represent. NOTE: not gated on `bounded` 鈥?    // raw products (y=x*y*z) are unbounded; only the sin(x*y*z) subclass is
    // bounded, and the NEURON鈫扵ANH handles squashing either way.
    //
    // Architecture: MULTIPLY(MULTIPLY(x,y),z) 鈫?NEURON 鈫?TANH 鈫?ADD
    // The nested MULTIPLY computes x路y路z, then NEURON鈫扵ANON learns the target.
    if (!is_binary && profile.num_inputs >= 3
        && profile.poly_r2 < config::COMPOUND3_POLY_R2_MAX
        && profile.var_r > config::COMPOUND3_VAR_MIN) {
        // Gather all raw INPUT node IDs.
        std::vector<uint64_t> graph_input_ids;
        for (const auto& node : graph_->get_nodes()) {
            if (node->get_type() == NodeType::INPUT) {
                graph_input_ids.push_back(node->get_id());
            }
        }
        if (graph_input_ids.size() >= 3) {
            // Pick the 3 inputs with highest Lipschitz (most active).
            // Lipschitz_per_axis is indexed by input order.
            std::vector<std::pair<double, size_t>> lip_idx;
            for (size_t i = 0; i < profile.lipschitz_per_axis.size()
                             && i < graph_input_ids.size(); ++i) {
                lip_idx.emplace_back(profile.lipschitz_per_axis[i], i);
            }
            std::sort(lip_idx.begin(), lip_idx.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            uint64_t src_a = graph_input_ids[lip_idx[0].second];
            uint64_t src_b = graph_input_ids[lip_idx[1].second];
            uint64_t src_c = graph_input_ids[lip_idx[2].second];
            Hypothesis compound3;
            compound3.type               = Hypothesis::COMPOUND_MULTIPLY3_NEURON;
            compound3.multiply_source_a  = src_a;
            compound3.multiply_source_b  = src_b;
            compound3.bool_source_a      = src_c;  // reuse bool_source_a for 3rd input
            candidates.push_back({std::move(compound3), config::SCORE_COMPOUND_MULTIPLY3_NEURON});
            Logger::info("Compound candidate emitted: MULTIPLY3(input="
                         + std::to_string(src_a) + "," + std::to_string(src_b)
                         + "," + std::to_string(src_c)
                         + ")鈫扤EURON鈫扵ANH (3-way product, poly_r2="
                         + std::to_string(profile.poly_r2) + ")");
        }
    }

    // --- Candidate 7a: SIN_INJECTION 鈥?sin(wx+b) for oscillating residuals ---
    // One SIN node = one sine component. For bounded smooth residuals that
    // OSCILLATE (multiple sign changes when sorted by input 鈥?distinguishes
    // sin(kx) from a step function which has only 1 sign change).
if (profile.bounded
    && profile.var_r > config::COMPOUND_TANH_VAR_MIN
    && profile.num_inputs >= 1
    && profile.num_inputs <= static_cast<size_t>(config::SIN_INJECTION_MAX_INPUTS)
        && !diag.targets.empty()) {
        // Count sign changes in targets sorted by first input axis.
        std::vector<std::pair<Value, Value>> sorted_xt;
        for (size_t i = 0; i < diag.targets.size() && i < diag.local_inputs.size(); ++i) {
            Value xv = 0;
            if (!diag.local_inputs[i].empty()) xv = diag.local_inputs[i].begin()->second;
            sorted_xt.push_back({xv, diag.targets[i]});
        }
        std::sort(sorted_xt.begin(), sorted_xt.end());
        int sign_changes = 0;
        for (size_t i = 1; i < sorted_xt.size(); ++i) {
            if ((sorted_xt[i].second > 0) != (sorted_xt[i-1].second > 0)) ++sign_changes;
        }
        if (sign_changes >= 3) {  // oscillating (not a step with 1 crossing)
        std::vector<uint64_t> graph_input_ids;
        for (const auto& node : graph_->get_nodes()) {
            if (node->get_type() == NodeType::INPUT) graph_input_ids.push_back(node->get_id());
        }
        if (!graph_input_ids.empty()) {
            size_t axis = 0;
            if (profile.lipschitz_per_axis.size() == graph_input_ids.size()) {
                double mx = -1;
                for (size_t i = 0; i < graph_input_ids.size(); ++i)
                    if (profile.lipschitz_per_axis[i] > mx) { mx = profile.lipschitz_per_axis[i]; axis = i; }
            }
            // Frequency estimate from sign changes: each half-period is one
            // sign change, so periods = sign_changes/2 and
            // w = 2蟺路periods / (x_max - x_min). x is in the graph's (normalized)
            // input space, so w is in the space the new NEURON weight lives in.
            // d2 (x mod 3 over normalized [0,12]): 8 sign changes, 4 periods,
            // range ~3.46 鈫?w 鈮?7.26 鈥?exactly the frequency SGD was too slow
            // to reach from zero-init.
            //
            // M5.7 TWO-PEAK decomposition (d3_two_sines): a single-frequency
            // residual has UNIFORM inter-crossing gaps; a sum of two sines
            // has BIMODAL gaps (short = high component, long = low). When
            // bimodal, emit BOTH component frequencies instead of one blend
            // (the blend fits neither component and fails validation).
            double freq_est = 0.0;
            double freq2_est = 0.0;
            bool two_peak = false;
            double two_peak_gap_ratio_ = 0.0;   // diagnostics
            // Gap-decomposition helper shared by residual- and label-space
            // analyses: returns (ok, w_lo, w_hi, ratio).
            auto gap_two_peak = [](const std::vector<std::pair<Value, Value>>& sxt,
                                   double& w_lo, double& w_hi, double& ratio) -> bool {
                std::vector<double> gaps;
                for (size_t i = 1; i < sxt.size(); ++i) {
                    if ((sxt[i].second > 0) != (sxt[i-1].second > 0)) {
                        double g = sxt[i].first - sxt[i-1].first;
                        if (g > 1e-9) gaps.push_back(g);
                    }
                }
                if (gaps.size() < 6) return false;
                std::sort(gaps.begin(), gaps.end());
                std::vector<double> lo(gaps.begin(), gaps.begin() + gaps.size() / 2);
                std::vector<double> hi(gaps.begin() + gaps.size() / 2, gaps.end());
                auto mean = [](const std::vector<double>& v) {
                    double s = 0; for (double x : v) s += x;
                    return v.empty() ? 0.0 : s / v.size();
                };
                auto cv = [&mean](const std::vector<double>& v) {
                    if (v.size() < 2) return 1e9;
                    double m = mean(v), s = 0;
                    for (double x : v) s += (x - m) * (x - m);
                    return std::sqrt(s / v.size()) / std::max(m, 1e-12);
                };
                double m_lo = mean(lo), m_hi = mean(hi);
                if (m_lo <= 1e-9 || m_hi <= 1e-9) return false;
                ratio = m_hi / m_lo;
                // Genuine two-frequency sums: component ratio in [1.8, 8]
                // with TIGHT clusters (CV < 0.8 each). Ratios far above 8
                // with a tiny-gap cluster are tangent-touch noise.
                if (!(ratio > 1.8 && ratio < 8.0
                      && cv(lo) < 0.8 && cv(hi) < 0.8
                      && lo.size() >= 3 && hi.size() >= 3)) return false;
                constexpr double kPi = 3.14159265358979323846;
                w_lo = kPi / m_hi;
                w_hi = kPi / m_lo;
                return w_lo >= 0.5 && w_hi <= 120.0;
            };
            {
                if (sorted_xt.size() >= 2) {
                    double x_lo = sorted_xt.front().first, x_hi = sorted_xt.back().first;
                    double x_range = x_hi - x_lo;
                    constexpr double kTwoPi = 6.28318530717958647692;
                    if (x_range > 1e-6) {
                        double periods = static_cast<double>(sign_changes) / 2.0;
                        freq_est = kTwoPi * periods / x_range;
                        if (freq_est < 0.5 || freq_est > 60.0) freq_est = 0.0;
                        // Residual-space two-peak (helper does the gating).
                        double w_lo = 0.0, w_hi = 0.0, ratio = 0.0;
                        if (gap_two_peak(sorted_xt, w_lo, w_hi, ratio)) {
                            freq_est = w_lo;
                            freq2_est = w_hi;
                            two_peak = true;
                            two_peak_gap_ratio_ = ratio;
                        } else if (ratio > 0.0) {
                            two_peak_gap_ratio_ = ratio;
                        }
                    }
                }
            }
            uint64_t src_id = graph_input_ids[std::min(axis, graph_input_ids.size()-1)];

            // M7.5(c): a matched sin-family library entry's LEARNED
            // frequency beats a missing residual estimate (and is data
            // the sign-change analysis cannot see on distorted residuals).
            if (freq_est == 0.0 && library_freq_hint_ > 0.0) {
                freq_est = library_freq_hint_;
                Logger::info("SIN freq-init from library hint: w="
                            + std::to_string(freq_est));
            }

            // M5.7 v2: LABEL-SPACE two-peak fallback. Post-fit residuals
            // distort the crossing structure (tangent touches, the blend's
            // own crossings — measured ratios 60.7/10.5 on d3) while the
            // target's true components live in the RAW LABELS (ratio 2.12).
            // Same lesson as the zero-plateau detector: target structure
            // is label-space evidence. Single-output graphs (v1).
            if (!two_peak) {
                size_t out_count = 0;
                uint64_t out_gid = 0;
                for (const auto& n : graph_->get_nodes()) {
                    if (n->get_type() == NodeType::OUTPUT) {
                        ++out_count;
                        out_gid = n->get_id();
                    }
                }
                uint64_t src_key = 0; bool have_sk = false;
                for (const auto& kv : input_data_to_graph_) {
                    if (kv.second == src_id) { src_key = kv.first; have_sk = true; break; }
                }
                uint64_t out_key = 0; bool have_ok = false;
                for (const auto& kv : output_data_to_graph_) {
                    if (kv.second == out_gid) { out_key = kv.first; have_ok = true; break; }
                }
                if (out_count == 1 && have_sk && have_ok
                    && !training_data_.samples.empty()) {
                    std::vector<std::pair<Value, Value>> lxy;
                    for (const auto& s : training_data_.samples) {
                        auto xi = s.inputs.find(src_key);
                        auto yi = s.targets.find(out_key);
                        if (xi != s.inputs.end() && yi != s.targets.end()) {
                            lxy.emplace_back(xi->second, yi->second);
                        }
                    }
                    if (lxy.size() >= 16) {
                        std::sort(lxy.begin(), lxy.end(),
                                  [](const std::pair<Value, Value>& a,
                                     const std::pair<Value, Value>& b) {
                                      return a.first < b.first;
                                  });
                        double w_lo = 0.0, w_hi = 0.0, ratio = 0.0;
                        if (gap_two_peak(lxy, w_lo, w_hi, ratio)) {
                            freq_est = w_lo;
                            freq2_est = w_hi;
                            two_peak = true;
                            two_peak_gap_ratio_ = ratio;
                            Logger::info("Label-space two-peak: w_lo="
                                        + std::to_string(w_lo) + " w_hi="
                                        + std::to_string(w_hi) + " (ratio "
                                        + std::to_string(ratio) + ")");
                        }
                    }
                }
            }

            // Freq-init variant: starts at the estimated frequency; SGD only
            // refines amplitude/phase. Ranked slightly above the zero-init
            // fallback; validation gate decides which (if either) commits.
            if (freq_est > 0.0) {
                Hypothesis sinj_f;
                sinj_f.type = Hypothesis::SIN_INJECTION;
                sinj_f.multiply_source_a = src_id;
                sinj_f.sin_freq_init = freq_est;
                candidates.push_back({std::move(sinj_f), piecewise_signature ? 0.45 : (config::SCORE_COMPOUND_TANH_SERIES + 0.01)});
                Logger::info("Candidate emitted: SIN_INJECTION freq-init (w="
                            + std::to_string(freq_est) + ", " + std::to_string(sign_changes)
                            + " sign changes, gap-ratio "
                            + std::to_string(two_peak_gap_ratio_)  // set below
                            + (two_peak ? ", two-peak LOW" : "") + ")");
            }
            // M5.7: the second component of a two-frequency residual.
            if (two_peak && freq2_est > 0.0) {
                Hypothesis sinj_f2;
                sinj_f2.type = Hypothesis::SIN_INJECTION;
                sinj_f2.multiply_source_a = src_id;
                sinj_f2.sin_freq_init = freq2_est;
                candidates.push_back({std::move(sinj_f2), piecewise_signature ? 0.45 : (config::SCORE_COMPOUND_TANH_SERIES + 0.005)});
                Logger::info("Candidate emitted: SIN_INJECTION freq-init (w="
                            + std::to_string(freq2_est) + ", two-peak HIGH)");
            }
            // Zero-init variant (proven fallback; identity start).
            Hypothesis sinj;
            sinj.type = Hypothesis::SIN_INJECTION;
            sinj.multiply_source_a = src_id;
            candidates.push_back({std::move(sinj), piecewise_signature ? 0.45 : config::SCORE_COMPOUND_TANH_SERIES});
            Logger::info("Candidate emitted: SIN_INJECTION (bounded oscillating residual)");

            // --- COMPOUND_SIN_PRODUCT: MULTIPLY(a,b) 鈫?NEURON(freq) 鈫?SIN ---
            // For sin-of-product signatures: residual oscillates when sorted
            // by ANY single axis (sign_changes 鈮?3) AND the profile shows an
            // interaction (interaction_dominant or sobol pairwise) 鈥?the
            // oscillation lives in the PRODUCT space, which single-axis SIN
            // can't express. Sources: same interaction pair MULTIPLY uses.
            if (profile.num_inputs >= 2
                && (profile.interaction_dominant
                    || profile.high_pairwise_interaction)) {
                uint64_t pa = 0, pb = 0;
                if (profile.interaction_dominant) {
                    pa = graph_input_ids[std::min(profile.interact_a, graph_input_ids.size()-1)];
                    pb = graph_input_ids[std::min(profile.interact_b, graph_input_ids.size()-1)];
                } else {
                    pa = graph_input_ids[std::min(profile.sobol_pair_a, graph_input_ids.size()-1)];
                    pb = graph_input_ids[std::min(profile.sobol_pair_b, graph_input_ids.size()-1)];
                    if (pa == pb && graph_input_ids.size() >= 2)
                        pb = graph_input_ids[(profile.sobol_pair_a + 1) % graph_input_ids.size()];
                }
            if (pa != 0 && pb != 0 && pa != pb) {
                Hypothesis csp;
                csp.type = Hypothesis::COMPOUND_SIN_PRODUCT;
                csp.multiply_source_a = pa;
                csp.multiply_source_b = pb;
                // Reuse the single-axis frequency estimate as the inner
                // neuron init 鈥?SGD refines it into product-space units.
                csp.sin_freq_init = (freq_est > 0.0) ? freq_est : 1.0;
                candidates.push_back({std::move(csp), config::SCORE_COMPOUND_SIN_PRODUCT});
                Logger::info("Candidate emitted: COMPOUND_SIN_PRODUCT (input="
                            + std::to_string(pa) + "*input=" + std::to_string(pb)
                            + ", freq_init=" + std::to_string(csp.sin_freq_init) + ")");
            }
        }
    }
        }
    }

    // --- COMPOUND_SIN_PRODUCT, unbounded path ---
    // x路sin(x路y)-class targets (Korns F4): the residual includes a strong
    // LINEAR component (2.3路x0) so the bounded gate above never fires, but
    // the degree-2 fit leaves a non-polynomial remainder. Emit when:
    // poly_r2 < 0.95 (a pure x路y product would fit degree-2 exactly).
    // Sources: interaction evidence if present; else the top-2 Lipschitz
    // axes (oscillation concentrates local slope on the participating
    // inputs). Default freq init 1.0: product-space oscillation over
    // |p| 鈮?10 has O(1) frequency; SGD refines within the 3x budget.
    if (!profile.bounded
        && profile.num_inputs >= 2
        && profile.poly_r2 < 0.95) {
        std::vector<uint64_t> graph_input_ids;
        bool has_sinprod = false;
        for (const auto& node : graph_->get_nodes()) {
            if (node->get_type() == NodeType::INPUT) graph_input_ids.push_back(node->get_id());
            else if (node->get_name().rfind("sinprod_", 0) == 0) has_sinprod = true;
        }
        if (!has_sinprod && graph_input_ids.size() >= 2) {
            uint64_t pa = 0, pb = 0;
            double score = config::SCORE_COMPOUND_SIN_PRODUCT;
            if (profile.interaction_dominant) {
                pa = graph_input_ids[std::min(profile.interact_a, graph_input_ids.size()-1)];
                pb = graph_input_ids[std::min(profile.interact_b, graph_input_ids.size()-1)];
            } else if (profile.high_pairwise_interaction) {
                pa = graph_input_ids[std::min(profile.sobol_pair_a, graph_input_ids.size()-1)];
                pb = graph_input_ids[std::min(profile.sobol_pair_b, graph_input_ids.size()-1)];
                if (pa == pb)
                    pb = graph_input_ids[(profile.sobol_pair_a + 1) % graph_input_ids.size()];
            } else if (profile.lipschitz_max > config::PROFILE_SHARP_BOUNDARY_LIPSCHITZ
                       && profile.lipschitz_per_axis.size() == graph_input_ids.size()) {
                size_t a1 = 0, a2 = 1;
                if (profile.lipschitz_per_axis[a2] > profile.lipschitz_per_axis[a1])
                    std::swap(a1, a2);
                for (size_t i = 2; i < profile.lipschitz_per_axis.size(); ++i) {
                    if (profile.lipschitz_per_axis[i] > profile.lipschitz_per_axis[a1]) {
                        a2 = a1; a1 = i;
                    } else if (profile.lipschitz_per_axis[i] > profile.lipschitz_per_axis[a2]) {
                        a2 = i;
                    }
                }
                pa = graph_input_ids[a1];
                pb = graph_input_ids[a2];
                score = 0.90;  // speculative pair choice 鈥?below MULTIPLY fallback
            }
            if (pa != 0 && pb != 0 && pa != pb) {
                Hypothesis csp;
                csp.type = Hypothesis::COMPOUND_SIN_PRODUCT;
                csp.multiply_source_a = pa;
                csp.multiply_source_b = pb;
                csp.sin_freq_init = 1.0;
                // Memory-signature demotion (narma30 lesson): oscillation
                // hypotheses must not outrank the memory family when the
                // missing structure is temporal.
                if (cfg_.sequence_mode && profile.num_inputs <= 3) {
                    score = std::min(score, config::SCORE_DELAY_LINE - 0.05);
                }
                candidates.push_back({std::move(csp), score});
                Logger::info("Candidate emitted: COMPOUND_SIN_PRODUCT unbounded (input="
                            + std::to_string(pa) + "*input=" + std::to_string(pb)
                            + ", poly_r2=" + std::to_string(profile.poly_r2) + ")");
            }
        }
    }

    // --- Candidate 7: COMPOUND_TANH_SERIES 鈥?sin(x)-class smooth bounded ---
    // Triggered when residual is bounded + smooth + high-variance, but NOT
    // sharp-boundary (which would suggest IFELSE instead). This catches the
    // signature of "needs more curve capacity" that classify_failure()
    // misclassifies as LINEAR_OFFSET (the oscillation around zero inflates
// the linear-fit R虏).
//
// NOTE: an earlier relaxation (drop !sharp + lipschitz ceiling) let this fire
// for high-frequency sin like d3 鈥?selection-correct but net-negative: the
// K tanh chains can't converge fast enough in the shadow budget, so d3 ended
// up slightly worse (0.117 -> 0.122) and 7x more variable. The binding
// constraint there is convergence rate / expressivity (sin(11x) needs ~22
// bumps + heavy training), not selection. Keep the original smooth-residual
// gates until that deeper layer is addressed.
if (profile.bounded
    && !profile.sharp_boundary
    && profile.var_r > config::COMPOUND_TANH_VAR_MIN
    && profile.lipschitz_max < config::COMPOUND_TANH_LIPSCHITZ_MAX
    && profile.num_inputs >= 1) {
        // Pick the INPUT source on the axis with the highest Lipschitz
        // estimate (most "wiggly" direction).
        size_t axis = 0;
        double max_lip = -1.0;
        for (size_t i = 0; i < profile.lipschitz_per_axis.size() && i < profile.num_inputs; ++i) {
            if (profile.lipschitz_per_axis[i] > max_lip) {
                max_lip = profile.lipschitz_per_axis[i];
                axis = i;
            }
        }
        // Find the INPUT node id for this axis.
        uint64_t src = 0;
        size_t input_idx = 0;
        for (const auto& bs : blackboard) {
            if (bs.is_input) {
                if (input_idx == axis) { src = bs.node_id; break; }
                input_idx++;
            }
        }
        if (src == 0) {
            // Fallback: first INPUT signal in the blackboard.
            for (const auto& bs : blackboard) {
                if (bs.is_input) { src = bs.node_id; break; }
            }
        }
        if (src == 0) {
            // Last resort: look up INPUT nodes directly from the graph.
            // The blackboard is correlation-ranked and may exclude INPUTs
            // when their correlation with the residual is low (common after
            // a NEURON_TANH commit, where the input's linear correlation
            // with the oscillating residual is near zero).
            for (const auto& kv : input_data_to_graph_) {
                if (kv.second != 0) { src = kv.second; break; }
            }
        }
        if (src != 0) {
            Hypothesis series;
            series.type              = Hypothesis::COMPOUND_TANH_SERIES;
            series.multiply_source_a = src;  // reused as the single input source
            // Adaptive K: scale chain count with the residual's Lipschitz
            // estimate. More zero-crossings / sharper residual => more tanh
            // bumps needed. Floor at the default K (no regression on low-freq
            // tasks); ceiling at COMPOUND_TANH_SERIES_K_MAX.
            series.compound_K = static_cast<int>(std::ceil(profile.lipschitz_max));
            if (series.compound_K < config::COMPOUND_TANH_SERIES_K)
                series.compound_K = config::COMPOUND_TANH_SERIES_K;
            candidates.push_back({std::move(series), config::SCORE_COMPOUND_TANH_SERIES});
            Logger::info("Compound candidate emitted: TANH_SERIES(input="
                         + std::to_string(src) + ")脳"
                         + std::to_string(series.compound_K)
                         + " (bounded smooth residual, var_r="
                         + std::to_string(profile.var_r).substr(0, 6)
                         + ", L_max=" + std::to_string(profile.lipschitz_max).substr(0, 6) + ")");
        }
    }

    // --- Candidate 8: RECURRENT_SELF_WIRE 鈥?sequence/memory targets ---
    // Only attempted in sequence mode (cfg_.sequence_mode, set by --no-shuffle),
    // because memory-based targets need temporal order to carry state. Adds a
    // self-recurrent input to the failing NEURON; BPTT in train() grows it.
    // Validation rejects it where recurrence doesn't help (so it's safe to
    // emit speculatively in sequence mode).
    if (cfg_.sequence_mode) {
        const Node* fn = graph_->get_node(diag.failing_node);
        if (fn && (fn->get_type() == NodeType::NEURON
                   || fn->get_type() == NodeType::LINEAR)) {
            Hypothesis rec;
            rec.type = Hypothesis::RECURRENT_SELF_WIRE;
            candidates.push_back({std::move(rec), config::SCORE_RECURRENT_SELF_WIRE});
            Logger::info("Candidate emitted: RECURRENT_SELF_WIRE (sequence mode)");
        }

        // RECURRENT_XOR: for binary sequence tasks (running parity).
        // A NEURON can't compute XOR of (input, prev_state) 鈥?XOR needs
        // a dedicated node. Check if inputs are binary (2 distinct values).
        bool inputs_binary = true;
        {
            std::set<Value> distinct_vals;
            for (const auto& s : training_data_.samples) {
                for (const auto& kv : s.inputs) {
                    distinct_vals.insert(std::round(kv.second * 1000) / 1000.0);
                    if (distinct_vals.size() > 2) { inputs_binary = false; break; }
                }
                if (!inputs_binary) break;
            }
        }
        if (inputs_binary && fn && (fn->get_type() == NodeType::NEURON
                                    || fn->get_type() == NodeType::LINEAR)) {
            Hypothesis rxor;
            rxor.type = Hypothesis::RECURRENT_XOR;
            candidates.push_back({std::move(rxor), config::SCORE_RECURRENT_XOR});
            Logger::info("Candidate emitted: RECURRENT_XOR (binary sequence mode)");
        }

        // RECURRENT_MULTI_TAP: long-sequence memory (NARMA-30 collapsed to
        // mean prediction with single k=1 lines). K self-wires at delays
        // 1..K. Emitted in sequence mode for trainable failing nodes.
        {
            const Node* fn2 = graph_->get_node(diag.failing_node);
            if (fn2 && (fn2->get_type() == NodeType::NEURON
                        || fn2->get_type() == NodeType::LINEAR)) {
                Hypothesis rmt;
                rmt.type = Hypothesis::RECURRENT_MULTI_TAP;
                rmt.compound_K = config::RECURRENT_MULTI_TAP_K;
                candidates.push_back({std::move(rmt), config::SCORE_RECURRENT_MULTI_TAP});
                Logger::info("Candidate emitted: RECURRENT_MULTI_TAP (K="
                            + std::to_string(config::RECURRENT_MULTI_TAP_K) + ")");

                // DELAY_LINE: k delayed copies of the raw input u[t-1..t-k]
                // as features (narma10_lag proved lag features suffice).
                // Delay-line dedup: skip if the failing node already has
                // delayed input edges wired in.
                bool has_delay_line = false;
                for (const auto& c : graph_->get_connections()) {
                    if (c.dst_node == diag.failing_node && c.delay_taps > 0) {
                        has_delay_line = true;
                        break;
                    }
                }
                if (!has_delay_line) {
                    Hypothesis dl;
                    dl.type = Hypothesis::DELAY_LINE;
                    dl.compound_K = config::DELAY_LINE_K;
                    candidates.push_back({std::move(dl), config::SCORE_DELAY_LINE});
                    Logger::info("Candidate emitted: DELAY_LINE (k="
                                + std::to_string(config::DELAY_LINE_K) + " input lags)");
                }
            }
        }
    }

    // --- Candidate 9: PARITY_TREE 鈥?k-bit parity as ONE atomic XOR tree ---
    // Greedy structural search can't build parity incrementally: every
    // intermediate XOR(x_i, x_j) gives ZERO loss improvement over the
    // base rate, so single-step validation rejects each partial tree.
    // The tree XOR(x0, XOR(x1, ...)) is exact, so inject it whole and
    // let validation see the finished function.
    // Gate: binary inputs (鈮? distinct raw values), binary target,
    // 3..16 inputs, not sequence mode (RECURRENT_XOR owns that), no
    // existing parity root (dedup).
    if (!cfg_.sequence_mode && is_binary) {
        std::vector<uint64_t> graph_input_ids;
        bool has_parity_root = false;
        for (const auto& node : graph_->get_nodes()) {
            if (node->get_type() == NodeType::INPUT) graph_input_ids.push_back(node->get_id());
            else if (node->get_name() == "parity_root") has_parity_root = true;
        }
        int k = static_cast<int>(graph_input_ids.size());
        if (!has_parity_root
            && k >= config::PARITY_TREE_MIN_INPUTS
            && k <= config::PARITY_TREE_MAX_INPUTS) {
            // Verify all inputs are binary (raw values; binary features
            // skip normalization so they stay 0/1 or 卤1).
            bool inputs_binary = true;
            {
                std::set<Value> distinct;
                for (const auto& s : training_data_.samples) {
                    for (const auto& kv : s.inputs) {
                        distinct.insert(std::round(kv.second * 1000) / 1000.0);
                        if (distinct.size() > 2) { inputs_binary = false; break; }
                    }
                    if (!inputs_binary) break;
                }
            }
            if (inputs_binary) {
                Hypothesis pt;
                pt.type = Hypothesis::PARITY_TREE;
                candidates.push_back({std::move(pt), config::SCORE_PARITY_TREE});
                Logger::info("Candidate emitted: PARITY_TREE (" + std::to_string(k)
                            + "-bit, atomic XOR tree)");
            }
        }
    }

    // --- Library prior: if a loaded SubgraphLibrary has an entry whose
    // behavioral fingerprint matches the current needed-behavior (the target
    // function shape), either BOOST an already-emitted candidate's score or
    // INJECT a candidate the profile gates blocked. Injection only fires on
    // strong matches (distance < LIBRARY_INJECT_THRESHOLD) and the validation
    // gate still decides commit/reject 鈥?so it's safe to try.
    if (library_ && !library_->entries().empty() && !diag.targets.empty()) {
        std::vector<uint64_t> lib_input_ids;
        for (const auto& node : graph_->get_nodes())
            if (node->get_type() == NodeType::INPUT) lib_input_ids.push_back(node->get_id());
        if (!lib_input_ids.empty() && diag.local_inputs.size() == diag.targets.size()) {
            std::vector<std::vector<double>> X(diag.targets.size(),
                                                std::vector<double>(lib_input_ids.size()));
            for (size_t i = 0; i < diag.local_inputs.size(); ++i) {
                for (size_t j = 0; j < lib_input_ids.size(); ++j) {
                    auto it = diag.local_inputs[i].find(lib_input_ids[j]);
                    X[i][j] = (it != diag.local_inputs[i].end()) ? it->second : 0.0;
                }
            }
            BehavioralFingerprint needed = compute_fingerprint(X, diag.targets);

            // M1.1: expose this cycle's residual fingerprint to the shadow
            // loop (failure records) — valid for this generate_candidates call.
            current_cycle_fp_ = needed;
            current_cycle_fp_valid_ = true;

            // M1.1 READ SIDE: down-weight families that repeatedly failed
            // on similar residuals (loaded failure library). Penalty is
            // per-family: max over matching failures of
            // FAILURE_PENALTY * count_capped. Applied AFTER candidates are
            // built (below) to avoid restructuring every emission block.
            if (failure_library_ && !failure_library_->empty()) {
                for (const auto& fr : *failure_library_) {
                    if (fr.fingerprint.num_inputs != needed.num_inputs) continue;
                    double d = fingerprint_distance(needed, fr.fingerprint);
                    if (d < config::FAILURE_MATCH_RADIUS) {
                        // M1.5: records written under older family
                        // semantics describe candidates the current
                        // engine no longer emits — discount them so v3
                        // candidates aren't pre-penalized by v1 failures.
                        double w = (fr.version < config::FAILURE_FAMILY_VERSION)
                                 ? config::FAILURE_LEGACY_VERSION_DISCOUNT : 1.0;
                        family_penalty[fr.hyp_type] += config::FAILURE_PENALTY_UNIT * w;
                    }
                }
                // Cap per-family penalty so nothing is banned outright on
                // a couple of old failures — families stay reachable.
                for (auto& kv : family_penalty) {
                    if (kv.second > config::FAILURE_PENALTY_MAX) {
                        kv.second = config::FAILURE_PENALTY_MAX;
                    }
                }
            }

            // === Matcher guards (library-audit fixes) ===
            // G1: degenerate fingerprints refuse to match. When the needed-
            // behavior's variance ~ 0 (e.g. charLM residuals: all features
            // near-constant), the fingerprint vector is mostly zeros and
            // EVERY zero-variance task matches every other at distance~0
            // (observed: w1 -> narma10 at dist=0.000 — pure noise).
            bool degenerate = needed.var < 1e-8;
            // G2: self-echo. A task matching its OWN earlier library entry
            // injects what it already tried (hetero3: 65 self-injects, 1
            // commit). Retrieve matches EXCLUDING entries sourced from the
            // current task; cross-task transfer is the entire point.
            auto matches = library_->find_matches_excluding_self(
                needed, 1, current_task_name_);
            // M7.5(c): a matched sin-family entry that STORES its source
            // expression's numeric literals carries learned frequencies —
            // freq-init's missing data source. Stash for the SIN emission
            // block (used when the residual analysis yields no estimate,
            // or to refine one derived from a distorted residual).
            if (!matches.empty()
                && matches[0].distance < config::LIBRARY_MATCH_THRESHOLD) {
                const auto& me = library_->entry(matches[0].index);
                if ((me.pattern == "sin_chain" || me.pattern == "sin_component")
                    && !me.params.empty()) {
                    try {
                        double pf = std::stod(me.params);
                        if (pf >= 0.5 && pf <= 60.0) {
                            library_freq_hint_ = pf;
                            Logger::info("Library freq hint: w="
                                        + std::to_string(pf) + " (from "
                                        + me.source_task + ")");
                        }
                    } catch (...) {}
                }
            }
            if (!degenerate
                && !matches.empty()
                && matches[0].distance < config::LIBRARY_MATCH_THRESHOLD) {
                const auto& matched_entry = library_->entry(matches[0].index);
                const auto& mfp = matched_entry.fingerprint;

                // Use the CURRENT PROBLEM's profile (not the matched entry's)
                // to select the hypothesis type. The library's role is to say
                // "this behavior has been solved before" (fingerprint match) 鈥?                // a confidence signal + injection permission. The CURRENT
                // problem's own shape determines WHICH tool to use.
                double pr_range = profile.max_r - profile.min_r;
                bool pr_sign_sym = (pr_range > 1e-9) ?
                    (profile.min_r >= -config::ABSMUL_NONNEG_FRACTION * pr_range) : true;
                Hypothesis::Type suggested = Hypothesis::NEURON_TANH_INJECTION;
                if (profile.interaction_dominant && pr_sign_sym && profile.num_inputs >= 2)
                    suggested = Hypothesis::COMPOUND_MULTIPLY_ABS;
                else if (profile.interaction_dominant && profile.bounded)
                    suggested = Hypothesis::COMPOUND_MULTIPLY_NEURON;
                else if (profile.interaction_dominant)
                    suggested = Hypothesis::MULTIPLY_INJECTION;
                else if (profile.bounded && !profile.sharp_boundary)
                    suggested = Hypothesis::COMPOUND_TANH_SERIES;
                else if (profile.sharp_boundary)
                    suggested = Hypothesis::IFELSE_BOUNDARY_SPLIT;
                const char* hnames[] = {"NONE","IFELSE","NEURON_TANH","CONTEXT","MULTIPLY","BOOL","MUL_NEURON","TANH_SERIES","MUL3","MUL_ABS","RECURRENT","SIN_INJECT"};

                bool already_present = false;
                for (auto& sh : candidates) {
                    if (sh.hyp.type == suggested) {
                        // M7.5(b): pattern-aware boost strength. The matched
                        // entry's semantic tag says WHICH family solved this
                        // behavior before — a "sin_chain"/"sin_component"
                        // match strongly endorses SIN-family candidates;
                        // "product"/"abs_product" endorses MULTIPLY family.
                        // Generic +0.2 for untagged/unknown patterns.
                        double boost = config::LIBRARY_BOOST;
                        {
                            const std::string& pat =
                                library_->entry(matches[0].index).pattern;
                            if (pat == "sin_chain" || pat == "sin_component") {
                                if (suggested == Hypothesis::SIN_INJECTION
                                    || suggested == Hypothesis::COMPOUND_SIN_PRODUCT
                                    || suggested == Hypothesis::COMPOUND_TANH_SERIES) {
                                    boost = config::LIBRARY_BOOST * 2.0;
                                }
                            } else if (pat == "product" || pat == "abs_product") {
                                if (suggested == Hypothesis::MULTIPLY_INJECTION
                                    || suggested == Hypothesis::COMPOUND_MULTIPLY_NEURON
                                    || suggested == Hypothesis::COMPOUND_MULTIPLY_ABS) {
                                    boost = config::LIBRARY_BOOST * 2.0;
                                }
                            } else if (pat == "boundary") {
                                if (suggested == Hypothesis::IFELSE_BOUNDARY_SPLIT
                                    || suggested == Hypothesis::IFELSE_PRESERVE
                                    || suggested == Hypothesis::MUX_INJECTION) {
                                    boost = config::LIBRARY_BOOST * 2.0;
                                }
                            }
                        }
                        sh.score += boost;
                        already_present = true;
                        Logger::info("Library prior (dist="
                                     + std::to_string(matches[0].distance).substr(0,5)
                                     + " src=" + library_->entry(matches[0].index).source_task
                                     + " pattern=" + library_->entry(matches[0].index).pattern
                                     + ") boosts " + hnames[static_cast<int>(suggested)]
                                     + " by " + std::to_string(boost));
                        break;
                    }
                }

                // --- Injection: if the suggested type is NOT already emitted
                // and the match is strong, inject it 鈥?bypassing the profile
                // gates that would normally block it. Validation still decides.
                if (!already_present && matches[0].distance < config::LIBRARY_INJECT_THRESHOLD) {
                    Hypothesis h;
                    h.type = suggested;
                    // Set minimum fields for apply_shadow_routing.
                    if (suggested == Hypothesis::COMPOUND_TANH_SERIES) {
                        size_t axis = 0;
                        if (profile.lipschitz_per_axis.size() == lib_input_ids.size()) {
                            double mx = -1;
                            for (size_t i = 0; i < lib_input_ids.size(); ++i)
                                if (profile.lipschitz_per_axis[i] > mx) { mx = profile.lipschitz_per_axis[i]; axis = i; }
                        }
                        h.multiply_source_a = lib_input_ids[std::min(axis, lib_input_ids.size()-1)];
                    } else if (suggested == Hypothesis::MULTIPLY_INJECTION
                               || suggested == Hypothesis::COMPOUND_MULTIPLY_NEURON
                               || suggested == Hypothesis::COMPOUND_MULTIPLY_ABS) {
                        h.multiply_source_a = lib_input_ids[std::min(profile.interact_a, lib_input_ids.size()-1)];
                        h.multiply_source_b = lib_input_ids[std::min(profile.interact_b, lib_input_ids.size()-1)];
                    } else if (suggested == Hypothesis::IFELSE_BOUNDARY_SPLIT) {
                        if (!blackboard.empty()) {
                            for (const auto& bs : blackboard) if (bs.is_input) { h.condition_source_node = bs.node_id; break; }
                            if (!h.condition_source_node) h.condition_source_node = blackboard[0].node_id;
                            auto reg_it = blackboard_registry_.find(h.condition_source_node);
                            if (reg_it != blackboard_registry_.end() && reg_it->second.size() >= 2) {
                                std::vector<Value> s = reg_it->second;
                                std::sort(s.begin(), s.end());
                                h.split_threshold = s[s.size()/2];
                            }
                        }
                    } else if (suggested == Hypothesis::CONTEXT_WIRE) {
                        if (!blackboard.empty()) h.wire_source_node = blackboard[0].node_id;
                    }
                    // NEURON_TANH_INJECTION needs no special fields.
                    candidates.push_back({std::move(h), config::LIBRARY_BOOST});
                    Logger::info("Library INJECT (dist="
                                 + std::to_string(matches[0].distance).substr(0,5)
                                 + " src=" + library_->entry(matches[0].index).source_task
                                 + ") -> " + hnames[static_cast<int>(suggested)]
                                 + " [gate-bypass]");
                }
            }
        }
    }

    // M1.1: apply failure-library penalties just before sorting. This is
    // the single choke point through which every emitted candidate flows,
    // regardless of which emission block produced it. (family_penalty is
    // populated in the library-prior block when a failure library loaded;
    // empty otherwise.)
    if (!family_penalty.empty()) {
        for (auto& c : candidates) {
            auto pit = family_penalty.find(static_cast<int>(c.hyp.type));
            if (pit != family_penalty.end() && pit->second > 0.0) {
                double before = c.score;
                c.score = std::max(0.01, c.score - pit->second);
                Logger::verbose("Failure-prior: type="
                               + std::to_string(static_cast<int>(c.hyp.type))
                               + " score " + std::to_string(before)
                               + " -> " + std::to_string(c.score));
            }
        }
    }

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const ScoredHyp& a, const ScoredHyp& b) { return a.score > b.score; });

    std::vector<Hypothesis> result;
    for (auto& c : candidates) result.push_back(std::move(c.hyp));
    return result;
}

// ============================================================================
// apply_shadow_routing 鈥?clone graph; apply the structural modification
// ============================================================================
// ============================================================================
// compute_gain_init — least-squares scale for freshly-injected features
// ============================================================================
double EvolutionEngine::compute_gain_init(Graph& shadow,
                                          uint64_t feature_src,
                                          uint64_t failing_id,
                                          const FailureDiagnosis& diag) const {
    if (diag.local_inputs.empty() || diag.targets.empty()) return 0.0;
    size_t n = std::min(diag.local_inputs.size(), diag.targets.size());
    if (n < 4) return 0.0;

    // The residual target: what the failing node SHOULD output (targets)
    // minus what it currently outputs. We reconstruct current outputs by
    // executing the ORIGINAL (pre-injection) subgraph — approximated here
    // by executing the shadow with the gain at 0: the ADD passes the
    // failing node's output through unchanged, so reading the failing
    // node's output in the shadow equals its original value regardless.
    double sum_f = 0, sum_r = 0, sum_ff = 0, sum_fr = 0;
    size_t used = 0;
    for (size_t i = 0; i < n; ++i) {
        for (const auto& kv : diag.local_inputs[i]) {
            // local_inputs keys are upstream node ids — but only INPUT/
            // CONSTANT nodes accept set_input_value. Internal nodes are
            // recomputed by execute() anyway; skip them.
            uint64_t sid = kv.first;
            const Node* n = shadow.get_node(sid);
            if (!n) {
                auto it = input_data_to_graph_.find(sid);
                if (it != input_data_to_graph_.end()) {
                    sid = it->second;
                    n = shadow.get_node(sid);
                }
                if (!n) continue;
            }
            if (n->get_type() == NodeType::INPUT
                || n->get_type() == NodeType::CONSTANT) {
                shadow.set_input_value(sid, kv.second);
            }
        }
        // NOTE: const method — execute() mutates node caches. We work on a
        // const_cast'd reference; safe because this is a throwaway shadow.
        auto& sh = const_cast<Graph&>(shadow);
        sh.execute();
        Value f = shadow.get_any_node_output(feature_src);
        Value cur = shadow.get_any_node_output(failing_id);
        Value r = diag.targets[i] - cur;
        sum_f += f; sum_r += r; sum_ff += f * f; sum_fr += f * r;
        ++used;
    }
    if (used < 4) return 0.0;
    double mf = sum_f / used, mr = sum_r / used;
    double var_f = sum_ff / used - mf * mf;
    if (var_f < 1e-12) return 0.0;
    double cov = sum_fr / used - mf * mr;
    double w = cov / var_f;
    // Sanity clamp: |gain| ≤ 10 keeps tanh(bounded) features from exploding.
    if (w > 10.0) w = 10.0;
    if (w < -10.0) w = -10.0;
    return w;
}

std::unique_ptr<Graph> EvolutionEngine::apply_shadow_routing(
    const Hypothesis& hyp,
    const FailureDiagnosis& diag) {

    auto shadow = graph_->clone();
    if (!shadow) return nullptr;

    uint64_t failing_id = diag.failing_node;
    NodeType ftype = NodeType::CONSTANT;
    {
        const Node* fn = shadow->get_node(failing_id);
        if (!fn) return nullptr;
        ftype = fn->get_type();
    }

    // === Committed-chain protection invariant ===
    // Hypotheses that RE-ROUTE the failing node's output (the IFELSE and
    // PRESERVE families detach the node's outgoing connections) must not
    // run on a node whose output already feeds a committed boundary chain:
    // the re-route orphans the chain (observed: PRESERVE multi-splits
    // committed, then a later rank-2 IFELSE_BOUNDARY_SPLIT re-routed the
    // same node and the chain vanished from the final graph).
    // Detection: any connection src==failing_id whose dst is an IFELSE
    // node (preserve chains consume the node's output at input port 1).
    // Non-re-routing hypotheses (MULTIPLY, CONTEXT_WIRE attach new inputs;
    // DEEP_INSERTION adds alongside) are unaffected.
    auto reroutes_output = [&](const Hypothesis& h) -> bool {
        switch (h.type) {
            case Hypothesis::IFELSE_BOUNDARY_SPLIT:
            case Hypothesis::IFELSE_PRESERVE:
            case Hypothesis::MUX_INJECTION:
                return true;
            default:
                return false;
        }
    };
    if (reroutes_output(hyp)) {
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node != failing_id) continue;
            const Node* dst = shadow->get_node(c.dst_node);
            if (dst && dst->get_type() == NodeType::IFELSE) {
                Logger::verbose("  routing refused: node " + std::to_string(failing_id)
                              + " feeds a committed boundary chain — re-route would orphan it");
                return nullptr;
            }
        }
    }

    switch (hyp.type) {

    case Hypothesis::CONTEXT_WIRE: {
        // Wire the correlated Blackboard signal into the failing node
        uint64_t src_id = hyp.wire_source_node;
        if (src_id == 0 || !shadow->get_node(src_id)) break;

        const Node* dst_node = shadow->get_node(failing_id);
        if (!dst_node) break;

        // Find the next available input port (check which ports are connected)
        // For NeuronNode/LinearNode, use set_input_count to expand
        if (dst_node->get_type() == NodeType::NEURON
            || dst_node->get_type() == NodeType::LINEAR) {
            size_t current = dst_node->get_num_inputs();
            // Check if all current ports are connected, find first unconnected
            size_t target_port = current;
            for (size_t i = 0; i < current; ++i) {
                if (!dst_node->is_input_connected(i)) { target_port = i; break; }
            }
            // Expand input count if needed
            Node* dst_mut = shadow->get_node(failing_id);
            if (target_port >= static_cast<NeuronNode*>(dst_mut)->get_num_weights()) {
                static_cast<NeuronNode*>(dst_mut)->set_input_count(target_port + 1);
            }
            shadow->add_connection(src_id, 0, failing_id, target_port);
        } else if (dst_node->get_num_inputs() > 0) {
            // For non-NEURON nodes with fixed inputs, wire to port 0
            // (removing existing connection at port 0)
            const auto& conns = shadow->get_connections();
            for (const auto& c : conns) {
                if (c.dst_node == failing_id && c.dst_port == 0) {
                    shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
                    break;
                }
            }
            shadow->add_connection(src_id, 0, failing_id, 0);
        }
        break;
    }

    case Hypothesis::NEURON_TANH_INJECTION: {
        // When the failing node is a NEURON, add a PARALLEL neuron (not serial).
        //
        // WHY: The previous code inserted NEURON_B AFTER the existing NEURON_A,
        // creating tanh(w_b * tanh(w_a 路 x) + b_b). This serial chain is WORSE
        // than the original single neuron for XOR-like problems because:
        //   - It still has a single bottleneck through NEURON_A
        //   - Double tanh squashing causes vanishing gradients
        //   - XOR requires TWO parallel features, not a serial composition
        //
        // The fix creates a parallel architecture:
        //   INPUT(s) 鈫?NEURON_A 鈹€鈹€鈹?        //                          鈹溾啋 ADD 鈫?downstream (OUTPUT)
        //   INPUT(s) 鈫?NEURON_B 鈹€鈹€鈹?        //
        // NEURON_B gets the SAME input connections as NEURON_A. The ADD node
        // combines them. This gives a 2-unit hidden layer with a linear output
        // head 鈥?sufficient to solve XOR and approximate piecewise functions.
        //
        // For non-NEURON failing nodes, keep the old serial approach (insert a
        // NEURON+TANH chain to add non-linear capacity where none existed).

        if (ftype == NodeType::NEURON || ftype == NodeType::LINEAR) {
            // --- Parallel neuron injection ---
            uint64_t new_neuron = shadow->add_node(NodeType::NEURON, "parallel_neuron");

            // Copy ALL incoming connections from the failing node to the new one
            std::vector<Connection> failing_inputs;
            for (const auto& c : shadow->get_connections()) {
                if (c.dst_node == failing_id) {
                    failing_inputs.push_back(c);
                }
            }
            // Expand the new neuron's input count to match the failing node's
            Node* fn = shadow->get_node(failing_id);
            if (fn && (fn->get_type() == NodeType::NEURON
                       || fn->get_type() == NodeType::LINEAR)) {
                size_t n_inputs = static_cast<NeuronNode*>(fn)->get_num_weights();
                Node* nn = shadow->get_node(new_neuron);
                if (nn && nn->get_type() == NodeType::NEURON) {
                    static_cast<NeuronNode*>(nn)->set_input_count(n_inputs);
                }
            }
            // Wire the same sources into the new neuron (same ports)
            for (const auto& c : failing_inputs) {
                shadow->add_connection(c.src_node, c.src_port, new_neuron, c.dst_port);
            }

            // Collect outgoing connections from failing_node
            std::vector<Connection> outgoing;
            for (const auto& c : shadow->get_connections()) {
                if (c.src_node == failing_id) {
                    outgoing.push_back(c);
                }
            }
            // Remove old outgoing connections from failing_node
            for (const auto& c : outgoing) {
                shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
            }

            // Add ADD node to combine failing NEURON + new NEURON
            uint64_t add_id = shadow->add_node(NodeType::ADD, "neuron_combine");
            shadow->add_connection(failing_id,  0, add_id, 0);
            shadow->add_connection(new_neuron,  0, add_id, 1);

            // Re-route: ADD 鈫?original downstream
            for (const auto& c : outgoing) {
                shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
            }
        } else {
            // --- Serial NEURON+TANH injection for non-NEURON failing nodes ---
            uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "shadow_neuron");

            std::vector<Connection> outgoing;
            for (const auto& c : shadow->get_connections()) {
                if (c.src_node == failing_id) {
                    outgoing.push_back(c);
                }
            }
            for (const auto& c : outgoing) {
                shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
            }

            shadow->add_connection(failing_id, 0, neuron_id, 0);

            uint64_t tanh_id = shadow->add_node(NodeType::TANH, "shadow_tanh");
            shadow->add_connection(neuron_id, 0, tanh_id, 0);
            for (const auto& c : outgoing) {
                shadow->add_connection(tanh_id, 0, c.dst_node, c.dst_port);
            }
        }
        break;
    }

    case Hypothesis::IFELSE_BOUNDARY_SPLIT: {
        // Insert IFELSE to mask the failing node's output based on a domain split.
        //
        // The GREATER comparator must compare the RAW INPUT signal against a
        // threshold constant — NOT the failing node's (tanh-squashed) output.
        // This is the key fix: previously, GREATER(failing_node, condition_src)
        // split on the post-activation output (bounded [-1,1]), which could not
        // correctly partition the input domain.
        //
        // IFELSE semantics: input[0]=condition, input[1]=value.
        //   condition=true  → output[0]=value, output[1]=0
        //   condition=false → output[0]=0,       output[1]=value
        //
        // We connect output[1] (false branch) to downstream. This creates a
        // domain mask: "pass failing_node through when x ≤ threshold, output 0
        // when x > threshold." For cliff/piecewise tasks where one region should
        // be suppressed, this is the correct behavior.
        //
        // M5.7 EVIDENCE PATH (structural_evidence): the failing node may be
        // ONE OF SEVERAL parallel contributors to the OUTPUT (observed t22:
        // starter_neuron ∥ parallel_neuron → ADD → OUTPUT). Masking an
        // internal contributor zeroes only ITS share while the OUTPUT scale
        // was calibrated for the SUM — the "identity start" never exists and
        // the in-region forward is corrupted (pre-train loss 0.12–0.28 vs
        // baseline 0.009, measured via EVIDENCE-PRE). Instead: wrap the
        // FINAL signal (the OUTPUT's port-0 source) with a DIRECTIONAL mask
        // — mask_above (x>thr→0, passes false branch out[1]) or mask_below
        // (x≤thr→0, passes true branch out[0]), per plateau-edge direction.
        if (hyp.structural_evidence) {
            uint64_t out_id = 0;
            bool out_found = false;
            uint64_t final_src = 0;
            size_t final_src_port = 0;
            for (const auto& n : shadow->get_nodes()) {
                if (n->get_type() == NodeType::OUTPUT) {
                    out_id = n->get_id();
                    out_found = true;
                    break;
                }
            }
            if (out_found) {
                int incoming = 0;
                for (const auto& c : shadow->get_connections()) {
                    if (c.dst_node == out_id) {
                        ++incoming;
                        final_src = c.src_node;
                        final_src_port = c.src_port;
                    }
                }
            }
            if (out_found && final_src != 0) {
                uint64_t condition_src = hyp.condition_source_node;
                if (condition_src == 0 || !shadow->get_node(condition_src)) {
                    condition_src = failing_id;
                }
                uint64_t threshold_id = shadow->add_node(NodeType::CONSTANT,
                                                         "ifelse_threshold");
                Node* cnode = shadow->get_node(threshold_id);
                if (cnode && cnode->get_type() == NodeType::CONSTANT) {
                    static_cast<ConstantNode*>(cnode)->set_value(hyp.split_threshold);
                }
                uint64_t greater_id = shadow->add_node(NodeType::GREATER, "ifelse_cond");
                shadow->add_connection(condition_src, 0, greater_id, 0);
                shadow->add_connection(threshold_id, 0, greater_id, 1);
                uint64_t ifelse_id = shadow->add_node(NodeType::IFELSE, "shadow_ifelse");
                shadow->add_connection(greater_id, 0, ifelse_id, 0);
                shadow->add_connection(final_src, final_src_port, ifelse_id, 1);
                // mask_above passes the FALSE branch (x≤thr keeps value);
                // mask_below passes the TRUE branch (x>thr keeps value).
                size_t pass_port = hyp.mask_above ? 1 : 0;
                shadow->remove_connection(final_src, final_src_port, out_id, 0);
                shadow->add_connection(ifelse_id, pass_port, out_id, 0);
                Logger::info("  [EVIDENCE-BOUNDARY] wrapped OUTPUT source node="
                             + std::to_string(final_src) + " thr="
                             + std::to_string(hyp.split_threshold)
                             + " mask_" + (hyp.mask_above ? "above" : "below"));
                break;
            }
            // No single OUTPUT source (multi-input OUTPUT): fall through
            // to the legacy failing-node masking below.
        }

        uint64_t condition_src = hyp.condition_source_node;

        // Resolve condition_src: must be a node whose output represents the
        // input domain signal (preferably an INPUT node from the blackboard).
        if (condition_src == 0 || !shadow->get_node(condition_src)) {
            // No blackboard signal available 鈥?fall back to failing_node as
            // condition source (degraded but better than no split).
            condition_src = failing_id;
        }

        // Always create a threshold CONSTANT (separate from condition_src).
        uint64_t threshold_id = shadow->add_node(NodeType::CONSTANT, "ifelse_threshold");
        Node* cnode = shadow->get_node(threshold_id);
        if (cnode && cnode->get_type() == NodeType::CONSTANT) {
            static_cast<ConstantNode*>(cnode)->set_value(hyp.split_threshold);
        }

        uint64_t ifelse_id = shadow->add_node(NodeType::IFELSE, "shadow_ifelse");

        // Collect outgoing connections from failing_node
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) {
                outgoing.push_back(c);
            }
        }

        // GREATER: condition_src > threshold
        // (e.g., "is x > 5?" for the cliff boundary)
        uint64_t greater_id = shadow->add_node(NodeType::GREATER, "ifelse_cond");
        shadow->add_connection(condition_src, 0, greater_id, 0);
        shadow->add_connection(threshold_id,  0, greater_id, 1);

        // Wire GREATER 鈫?IFELSE(0) [condition], failing_node 鈫?IFELSE(1) [value]
        shadow->add_connection(greater_id, 0, ifelse_id, 0);
        shadow->add_connection(failing_id, 0, ifelse_id, 1);

        // Remove old outgoing connections from failing_node
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }

        // Re-route: IFELSE output[1] (false branch: x 鈮?threshold) 鈫?downstream.
        // When x > threshold (true), output[1]=0 鈫?downstream gets 0 (masked).
        // When x 鈮?threshold (false), output[1]=failing_node 鈫?downstream gets value.
        for (const auto& c : outgoing) {
            shadow->add_connection(ifelse_id, 1, c.dst_node, c.dst_port);
        }

        break;
    }

    case Hypothesis::MULTIPLY_INJECTION: {
        // Add a product feature A*B (or A虏 if B is missing/==A).
        //
        // Architecture:
        //   src_a 鈹€鈹€鈹?        //          MULTIPLY 鈹€鈹€鈫?[feature output]
        //   src_b 鈹€鈹€鈹?        //
        // Routing:
        //   - If failing_node is NEURON: extend its input count by 1 and
        //     wire MULTIPLY output as a new input. The neuron's existing
        //     weights give the product a learnable coefficient for free.
        //   - Otherwise: route MULTIPLY 鈫?fresh 1-input NEURON 鈫?ADD combiner
        //     alongside the failing node's output. The fresh neuron provides
        //     the learnable weight w路(A路B) + b.
        //
        // Why: many targets need interaction terms (x*y) or polynomial
        // features (x虏) that a linear combination of inputs cannot express.
        // A single NEURON with tanh can fit x虏 on [-1,1] but saturates
        // outside that range 鈥?explicit MULTIPLY generalizes correctly.

        uint64_t src_a = hyp.multiply_source_a;
        uint64_t src_b = hyp.multiply_source_b;
        if (src_a == 0 || !shadow->get_node(src_a)) break;
        if (src_b == 0) src_b = src_a;  // self-product 鈫?x虏
        if (!shadow->get_node(src_b)) break;

        uint64_t mul_id = shadow->add_node(NodeType::MULTIPLY, "feature_product");
        shadow->add_connection(src_a, 0, mul_id, 0);
        shadow->add_connection(src_b, 0, mul_id, 1);

        Node* dst = shadow->get_node(failing_id);
        if (dst && (dst->get_type() == NodeType::NEURON
                    || dst->get_type() == NodeType::LINEAR)) {
            // Extend neuron/linear input count and wire product as new input
            size_t new_port = static_cast<NeuronNode*>(dst)->get_num_weights();
            static_cast<NeuronNode*>(dst)->set_input_count(new_port + 1);
            shadow->add_connection(mul_id, 0, failing_id, new_port);
        } else {
            // 1-input NEURON + ADD combiner
            uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "product_neuron");
            Node* nn = shadow->get_node(neuron_id);
            if (nn && nn->get_type() == NodeType::NEURON) {
                static_cast<NeuronNode*>(nn)->set_input_count(1);
            }
            shadow->add_connection(mul_id, 0, neuron_id, 0);

            std::vector<Connection> outgoing;
            for (const auto& c : shadow->get_connections()) {
                if (c.src_node == failing_id) outgoing.push_back(c);
            }
            for (const auto& c : outgoing) {
                shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
            }

            uint64_t add_id = shadow->add_node(NodeType::ADD, "product_combine");
            shadow->add_connection(failing_id, 0, add_id, 0);
            shadow->add_connection(neuron_id, 0, add_id, 1);
            for (const auto& c : outgoing) {
                shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
            }
        }
        break;
    }

    case Hypothesis::DIVIDE_INJECTION: {
        // Quotient feature: DIVIDE(num, den) wired like MULTIPLY_INJECTION.
        // For NEURON/LINEAR failing nodes, the new input port's weight is
        // ZERO-INIT (identity start 鈥?the raw ratio can be large, so a
        // random weight would disrupt the trained graph). Otherwise route
        // through a fresh 1-input NEURON + ADD combiner.

        uint64_t num = hyp.multiply_source_a;
        uint64_t den = hyp.multiply_source_b;
        if (num == 0 || den == 0) break;
        if (!shadow->get_node(num) || !shadow->get_node(den)) break;

        uint64_t div_id = shadow->add_node(NodeType::DIVIDE, "feature_quotient");
        shadow->add_connection(num, 0, div_id, 0);
        shadow->add_connection(den, 0, div_id, 1);

        Node* dst = shadow->get_node(failing_id);
        if (dst && (dst->get_type() == NodeType::NEURON
                    || dst->get_type() == NodeType::LINEAR)) {
            auto* dn = static_cast<NeuronNode*>(dst);
            size_t new_port = dn->get_num_weights();
            dn->set_input_count(new_port + 1);
            // Least-squares gain init (was 0.0): the quotient starts at the
            // scale the residual wants.
            dn->set_weight(new_port, compute_gain_init(*shadow, div_id, failing_id, diag));
            shadow->add_connection(div_id, 0, failing_id, new_port);
        } else {
            uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "quotient_neuron");
            Node* nn = shadow->get_node(neuron_id);
            if (nn && nn->get_type() == NodeType::NEURON) {
                static_cast<NeuronNode*>(nn)->set_input_count(1);
                static_cast<NeuronNode*>(nn)->set_weight(0, 0.0);
                static_cast<NeuronNode*>(nn)->set_bias(0.0);
            }
            shadow->add_connection(div_id, 0, neuron_id, 0);
            if (nn && nn->get_type() == NodeType::NEURON) {
                static_cast<NeuronNode*>(nn)->set_weight(
                    0, compute_gain_init(*shadow, div_id, failing_id, diag));
            }

            std::vector<Connection> outgoing;
            for (const auto& c : shadow->get_connections()) {
                if (c.src_node == failing_id) outgoing.push_back(c);
            }
            for (const auto& c : outgoing) {
                shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
            }
            uint64_t add_id = shadow->add_node(NodeType::ADD, "quotient_combine");
            shadow->add_connection(failing_id, 0, add_id, 0);
            shadow->add_connection(neuron_id, 0, add_id, 1);
            for (const auto& c : outgoing) {
                shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
            }
        }
        break;
    }

    case Hypothesis::COMPOUND_SIN_PRODUCT: {
        // MULTIPLY(a,b) 鈫?NEURON(freq-init) 鈫?SIN 鈫?ADD
        //                                     鈫?        //   failing_node 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?        //
        // sin-of-product targets (x路sin(x路y), A路sin(kx)). The inner NEURON
        // starts at the estimated FREQUENCY (not zero 鈥?the point of this
        // hypothesis is a trained oscillator over the product; zero-init
        // would kill its gradient through sin(0)). SGD refines frequency
        // (weight) and phase (bias). ADD gives identity start for the
        // amplitude-free graph.

        uint64_t src_a = hyp.multiply_source_a;
        uint64_t src_b = hyp.multiply_source_b;
        if (src_a == 0 || !shadow->get_node(src_a)) break;
        if (src_b == 0) src_b = src_a;
        if (!shadow->get_node(src_b)) break;

        // 1. MULTIPLY: x_i 路 x_j
        uint64_t mul_id = shadow->add_node(NodeType::MULTIPLY, "sinprod_product");
        shadow->add_connection(src_a, 0, mul_id, 0);
        shadow->add_connection(src_b, 0, mul_id, 1);

        // 2. NEURON with frequency init
        uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "sinprod_neuron");
        Node* nn = shadow->get_node(neuron_id);
        if (nn && nn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(nn)->set_input_count(1);
            static_cast<NeuronNode*>(nn)->set_weight(0, hyp.sin_freq_init);
            static_cast<NeuronNode*>(nn)->set_bias(0.0);
        }
        shadow->add_connection(mul_id, 0, neuron_id, 0);

        // 3. SIN
        uint64_t sin_id = shadow->add_node(NodeType::SIN, "sinprod_sin");
        shadow->add_connection(neuron_id, 0, sin_id, 0);

        // 3b. Zero-gain NEURON: identity start. The freq-init sin begins
        // ACTIVE (full oscillation) 鈥?wiring it straight into ADD disrupts
        // the trained graph (shadow_train 35 vs baseline 4.5 on Korns F4).
        // A zero-weight neuron after the sin lets SGD grow the amplitude
        // while the frequency is already at its estimate.
        uint64_t gain_id = shadow->add_node(NodeType::NEURON, "sinprod_gain");
        Node* gn = shadow->get_node(gain_id);
        if (gn && gn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(gn)->set_input_count(1);
            static_cast<NeuronNode*>(gn)->set_weight(0, 0.0);
            static_cast<NeuronNode*>(gn)->set_bias(0.0);
        }
        shadow->add_connection(sin_id, 0, gain_id, 0);
        // Least-squares gain init: sin component starts at optimal scale
        // (was 0.0 — lost first-cycle validation races on Korns F4).
        if (gn && gn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(gn)->set_weight(
                0, compute_gain_init(*shadow, sin_id, failing_id, diag));
        }

        // 4. ADD combiner (identity start)
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) outgoing.push_back(c);
        }
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }
        uint64_t add_id = shadow->add_node(NodeType::ADD, "sinprod_combine");
        shadow->add_connection(failing_id, 0, add_id, 0);
        shadow->add_connection(gain_id,    0, add_id, 1);
        for (const auto& c : outgoing) {
            shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
        }
        break;
    }

    case Hypothesis::COMPOUND_DIVIDE_PRODUCT: {
        // DIVIDE(MULTIPLY(a,b), c) 鈫?NEURON(zero-gain) 鈫?ADD
        //                                              鈫?        //   failing_node 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?        //
        // q虏a虏/c鲁-class targets (Feynman I.32.8). The raw quotient can be
        // large, so a zero-gain output neuron gives identity start (same
        // pattern as COMPOUND_SIN_PRODUCT's fix).

        uint64_t a = hyp.multiply_source_a;
        uint64_t b = hyp.multiply_source_b;
        uint64_t c = hyp.bool_source_a;
        if (a == 0 || b == 0 || c == 0) break;
        if (!shadow->get_node(a) || !shadow->get_node(b) || !shadow->get_node(c)) break;

        // Decode power variant: bool_source_b packs (num_pow<<8)|den_pow.
        // Default 0 → (1,1). pow chain built via nested MULTIPLY.
        int np = 1, dp = 1;
        if (hyp.bool_source_b != 0) {
            np = static_cast<int>((hyp.bool_source_b >> 8) & 0xFF);
            dp = static_cast<int>(hyp.bool_source_b & 0xFF);
            if (np < 1 || np > 3) np = 1;
            if (dp < 1 || dp > 3) dp = 1;
        }
        // pow(base, k): k=1 → base; else nested MULTIPLY (base^2, then ^3)
        auto build_pow = [&](uint64_t base, int k, const char* tag) -> uint64_t {
            uint64_t cur = base;
            for (int i = 2; i <= k; ++i) {
                uint64_t p = shadow->add_node(NodeType::MULTIPLY,
                                              std::string(tag) + "_pow" + std::to_string(i));
                shadow->add_connection(cur, 0, p, 0);
                shadow->add_connection(base, 0, p, 1);
                cur = p;
            }
            return cur;
        };

        uint64_t num_base = shadow->add_node(NodeType::MULTIPLY, "divprod_num");
        shadow->add_connection(a, 0, num_base, 0);
        shadow->add_connection(b, 0, num_base, 1);
        uint64_t num_powed = build_pow(num_base, np, "divprod_n");

        uint64_t den_powed = build_pow(c, dp, "divprod_d");

        uint64_t div_id = shadow->add_node(NodeType::DIVIDE, "divprod_quotient");
        shadow->add_connection(num_powed, 0, div_id, 0);
        shadow->add_connection(den_powed, 0, div_id, 1);

        // Gain node — LINEAR (identity), NOT NEURON: the quotient-of-
        // products is UNBOUNDED; a tanh gain squashes exactly the scale
        // information the ratio carries (measured on I.32.8: the exact
        // form (x0*x1)^2/x2^3 committed but routed through
        // tanh(0.021*form) — saturated at R2 0.976 with the structure
        // already correct). Least-squares init keeps the first-cycle
        // competitiveness.
        uint64_t gain_id = shadow->add_node(NodeType::LINEAR, "divprod_gain");
        Node* gn = shadow->get_node(gain_id);
        if (gn && (gn->get_type() == NodeType::NEURON
                   || gn->get_type() == NodeType::LINEAR)) {
            static_cast<NeuronNode*>(gn)->set_input_count(1);
            static_cast<NeuronNode*>(gn)->set_weight(0, 0.0);
            static_cast<NeuronNode*>(gn)->set_bias(0.0);
        }
        shadow->add_connection(div_id, 0, gain_id, 0);
        if (gn && gn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(gn)->set_weight(
                0, compute_gain_init(*shadow, div_id, failing_id, diag));
        }

        std::vector<Connection> outgoing;
        for (const auto& conn : shadow->get_connections()) {
            if (conn.src_node == failing_id) outgoing.push_back(conn);
        }
        for (const auto& conn : outgoing) {
            shadow->remove_connection(conn.src_node, conn.src_port, conn.dst_node, conn.dst_port);
        }
        uint64_t add_id = shadow->add_node(NodeType::ADD, "divprod_combine");
        shadow->add_connection(failing_id, 0, add_id, 0);
        shadow->add_connection(gain_id,    0, add_id, 1);
        for (const auto& conn : outgoing) {
            shadow->add_connection(add_id, 0, conn.dst_node, conn.dst_port);
        }
        break;
    }

    case Hypothesis::COMPOUND_MULTIPLY_NEURON: {
        // Compound composition: MULTIPLY(src_a, src_b) 鈫?NEURON 鈫?TANH 鈫?ADD
        //                                                     鈫?        //   failing_node 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?        //
        // Why a compound: MULTIPLY alone yields an unbounded product that
        // validation rejects (raw x路y is not a useful feature for sin(x路y)).
        // NEURON_TANH alone has no interaction feature to bind to. The
        // composition MULTIPLY 鈫?NEURON 鈫?TANH, validated atomically, can
        // learn sin(x路y) 鈮?tanh(w路(x路y) + b) 鈥?a bounded function of the
        // product. This case is emitted by generate_candidates when the
        // complexity profile shows a dominant polynomial cross-term AND a
        // bounded residual.

        uint64_t src_a = hyp.multiply_source_a;
        uint64_t src_b = hyp.multiply_source_b;
        if (src_a == 0 || !shadow->get_node(src_a)) break;
        if (src_b == 0) src_b = src_a;
        if (!shadow->get_node(src_b)) break;

        // 1. MULTIPLY node: x_i 路 x_j interaction feature.
        uint64_t mul_id = shadow->add_node(NodeType::MULTIPLY, "compound_product");
        shadow->add_connection(src_a, 0, mul_id, 0);
        shadow->add_connection(src_b, 0, mul_id, 1);

        // 2. NEURON: provides learnable weight + bias on the product.
        //    ZERO-INIT (weight=0, bias=0) so the new chain contributes
        //    nothing initially: TANH(0路x路y + 0) = TANH(0) = 0, and
        //    ADD(old, 0) = old. The compound starts as identity 鈥?SGD then
        //    grows the chain's contribution as it learns. Without this,
        //    random Xavier init produces TANH(random) 鈮?0, which disrupts
        //    the well-trained output and causes catastrophic regression
        //    (shadow_train jumps from 0.089 to 0.35 every cycle).
        uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "compound_neuron");
        Node* nn = shadow->get_node(neuron_id);
        if (nn && nn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(nn)->set_input_count(1);
            static_cast<NeuronNode*>(nn)->set_weight(0, 0.0);
            static_cast<NeuronNode*>(nn)->set_bias(0.0);
        }
        shadow->add_connection(mul_id, 0, neuron_id, 0);

        // 3. TANH: bounded squashing nonlinearity 鈥?this is what makes the
        //    composition useful for sin/tanh/sigmoid of product targets.
        //    Without it, the NEURON output is unbounded and validation
        //    rejects the commit just like bare MULTIPLY.
        uint64_t tanh_id = shadow->add_node(NodeType::TANH, "compound_tanh");
        shadow->add_connection(neuron_id, 0, tanh_id, 0);

        // 4. ADD combiner: failing_node's output + new tanh term.
        //    This preserves the existing contribution while adding the
        //    new bounded interaction feature in parallel.
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) outgoing.push_back(c);
        }
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }

        uint64_t add_id = shadow->add_node(NodeType::ADD, "compound_combine");
        shadow->add_connection(failing_id, 0, add_id, 0);
        shadow->add_connection(tanh_id,   0, add_id, 1);
        for (const auto& c : outgoing) {
            shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
        }
        break;
    }

    case Hypothesis::COMPOUND_MULTIPLY_ABS: {
        // |x*y|-class composition: MULTIPLY(a,b) -> ABS -> NEURON(zero-init) -> ADD.
        //
        // ABS(p) = p * sign(p), where sign(p) = 2*GREATER(p,0) - 1  in {-1,+1}.
        //
        // Emitted whenever interaction_dominant fires (a cross-term x_i*x_j is
        // the dominant missing feature). For sign-symmetric targets like |x*y|,
        // bare MULTIPLY is wrong-signed half the time and gets rejected by
        // single-step validation; ABS is necessary but useless alone. This
        // compound validates MULTIPLY->ABS as a unit. The zero-init NEURON gates
        // the abs contribution (TANH(0*|p|+0)=0 -> identity start), so SGD can
        // grow it without catastrophically disrupting the trained graph.

        uint64_t src_a = hyp.multiply_source_a;
        uint64_t src_b = hyp.multiply_source_b;
        if (src_a == 0 || !shadow->get_node(src_a)) break;
        if (src_b == 0) src_b = src_a;
        if (!shadow->get_node(src_b)) break;

        // 1. MULTIPLY: x_i * x_j
        uint64_t mul_id = shadow->add_node(NodeType::MULTIPLY, "absmul_product");
        shadow->add_connection(src_a, 0, mul_id, 0);
        shadow->add_connection(src_b, 0, mul_id, 1);

        // 2. ABS(product) = product * sign(product)
        auto add_const = [&](const char* name, Value v) -> uint64_t {
            uint64_t id = shadow->add_node(NodeType::CONSTANT, name);
            Node* cn = shadow->get_node(id);
            if (cn && cn->get_type() == NodeType::CONSTANT)
                static_cast<ConstantNode*>(cn)->set_value(v);
            return id;
        };
        uint64_t zero_c = add_const("absmul_zero", 0.0);
        uint64_t g_id = shadow->add_node(NodeType::GREATER, "absmul_gt0");
        shadow->add_connection(mul_id, 0, g_id, 0);
        shadow->add_connection(zero_c, 0, g_id, 1);

        uint64_t two_c = add_const("absmul_two", 2.0);
        uint64_t t2_id = shadow->add_node(NodeType::MULTIPLY, "absmul_2g");
        shadow->add_connection(two_c, 0, t2_id, 0);
        shadow->add_connection(g_id,  0, t2_id, 1);

        uint64_t one_c = add_const("absmul_one", 1.0);
        uint64_t sign_id = shadow->add_node(NodeType::SUBTRACT, "absmul_sign");
        shadow->add_connection(t2_id, 0, sign_id, 0);
        shadow->add_connection(one_c, 0, sign_id, 1);

        uint64_t abs_id = shadow->add_node(NodeType::MULTIPLY, "absmul_abs");
        shadow->add_connection(mul_id,  0, abs_id, 0);
        shadow->add_connection(sign_id, 0, abs_id, 1);

        // 3. Zero-init NEURON gates the abs term (identity start).
        uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "absmul_neuron");
        Node* nn = shadow->get_node(neuron_id);
        if (nn && nn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(nn)->set_input_count(1);
            static_cast<NeuronNode*>(nn)->set_weight(0, 0.0);
            static_cast<NeuronNode*>(nn)->set_bias(0.0);
        }
        shadow->add_connection(abs_id, 0, neuron_id, 0);

        // 4. ADD combiner: failing_node + new gated abs term.
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) outgoing.push_back(c);
        }
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }
        uint64_t add_id = shadow->add_node(NodeType::ADD, "absmul_combine");
        shadow->add_connection(failing_id, 0, add_id, 0);
        shadow->add_connection(neuron_id, 0, add_id, 1);
        for (const auto& c : outgoing) {
            shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
        }
        break;
    }

    case Hypothesis::COMPOUND_MULTIPLY3_NEURON: {
        // Three-way compound: MULTIPLY(MULTIPLY(a,b),c) 鈫?NEURON 鈫?TANH 鈫?ADD
        //                                                          鈫?        //   failing_node 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?        //
        // For sin(x路y路z)-class functions where no single pair dominates.
        // The nested MULTIPLY computes the 3-way product, then NEURON鈫扵ANH
        // learns sin of it. Same zero-init identity start as MULTIPLY_NEURON.

        uint64_t src_a = hyp.multiply_source_a;
        uint64_t src_b = hyp.multiply_source_b;
        uint64_t src_c = hyp.bool_source_a;  // reused for 3rd input
        if (src_a == 0 || !shadow->get_node(src_a)) break;
        if (src_b == 0 || !shadow->get_node(src_b)) break;
        if (src_c == 0 || !shadow->get_node(src_c)) break;

        // 1. Nested MULTIPLY: (x路y)路z
        uint64_t mul_ab = shadow->add_node(NodeType::MULTIPLY, "compound3_product_ab");
        shadow->add_connection(src_a, 0, mul_ab, 0);
        shadow->add_connection(src_b, 0, mul_ab, 1);
        uint64_t mul_abc = shadow->add_node(NodeType::MULTIPLY, "compound3_product_abc");
        shadow->add_connection(mul_ab, 0, mul_abc, 0);
        shadow->add_connection(src_c, 0, mul_abc, 1);

        // 2. NEURON (zero-init): TANH(0路xyz + 0) = 0 鈫?identity start.
        uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "compound3_neuron");
        Node* nn = shadow->get_node(neuron_id);
        if (nn && nn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(nn)->set_input_count(1);
            static_cast<NeuronNode*>(nn)->set_weight(0, 0.0);
            static_cast<NeuronNode*>(nn)->set_bias(0.0);
        }
        shadow->add_connection(mul_abc, 0, neuron_id, 0);

        // 3. TANH: bounded squashing nonlinearity for sin(xyz).
        uint64_t tanh_id = shadow->add_node(NodeType::TANH, "compound3_tanh");
        shadow->add_connection(neuron_id, 0, tanh_id, 0);

        // 4. ADD combiner: failing_node + new tanh term.
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) outgoing.push_back(c);
        }
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }
        uint64_t add_id = shadow->add_node(NodeType::ADD, "compound3_combine");
        shadow->add_connection(failing_id, 0, add_id, 0);
        shadow->add_connection(tanh_id,   0, add_id, 1);
        for (const auto& c : outgoing) {
            shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
        }
        break;
    }

    case Hypothesis::COMPOUND_TANH_SERIES: {
        // Compound series of NEURON鈫扵ANH chains, summed via ADD alongside
        // the failing node. Designed for smooth bounded residuals that need
        // more curve capacity than a single tanh can provide (e.g. sin(x)
        // over multiple periods).
        //
        // Architecture (K = COMPOUND_TANH_SERIES_K):
        //
        //                      鈹屸攢鈫?NEURON 鈹€鈫?TANH 鈹€鈹?        //   INPUT 鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈫?NEURON 鈹€鈫?TANH 鈹€鈹尖攢鈫?ADD 鈹€鈹€鈫?ADD 鈹€鈹€鈫?ADD 鈹€鈹€鈹?        //             鈹?       鈹斺攢鈫?NEURON 鈹€鈫?TANH 鈹€鈹?                        鈹?        //             鈹?                                                  鈫?        //   failing_node 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈫?ADD (chain)
        //                                                                鈹?        //                                                                鈫?        //                                                          (original outgoing)
        //
        // Rationale: a sum of K tanh bumps can approximate K zeros of a
        // smooth function. sin(x) over [-5, 5] has 3 zero crossings,
        // needing ~3 parallel tanh chains. A single NEURON_TANH_INJECTION
        // gets stuck at ~0.29 because it can only fit one bump.

        uint64_t src = hyp.multiply_source_a;  // reusing this field for the input source
        if (src == 0 || !shadow->get_node(src)) break;

        const int K = (hyp.compound_K > 0)
                      ? std::min(hyp.compound_K, config::COMPOUND_TANH_SERIES_K_MAX)
                      : config::COMPOUND_TANH_SERIES_K;
        std::vector<uint64_t> tanh_ids;
        for (int k = 0; k < K; ++k) {
            uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "tanh_series_neuron");
            Node* nn = shadow->get_node(neuron_id);
            if (nn && nn->get_type() == NodeType::NEURON) {
                static_cast<NeuronNode*>(nn)->set_input_count(1);
                // Zero-init ALL chains with tiny deterministic symmetry breaking.
                // Each chain k gets weight = epsilon * (k - K/2 + 0.5), so the
                // initial TANH output 鈮?epsilon路x 鈮?0 鈥?near identity start.
                // Without this, Xavier-init chains produce TANH(卤1路x) 鈮?卤1
                // which catastrophically disrupts the trained graph.
                constexpr double TANH_SERIES_EPS = 1e-3;
                double wk = TANH_SERIES_EPS * (static_cast<double>(k)
                             - static_cast<double>(K) / 2.0 + 0.5);
                static_cast<NeuronNode*>(nn)->set_weight(0, wk);
                static_cast<NeuronNode*>(nn)->set_bias(0.0);
            }
            shadow->add_connection(src, 0, neuron_id, 0);

            uint64_t tanh_id = shadow->add_node(NodeType::TANH, "tanh_series_tanh");
            shadow->add_connection(neuron_id, 0, tanh_id, 0);
            tanh_ids.push_back(tanh_id);
        }

        // Reroute failing_node's outgoing edges through an ADD chain that
        // sums the failing_node output with all K tanh outputs.
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) outgoing.push_back(c);
        }
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }

        // Build left-folded ADD tree: (((failing + tanh[0]) + tanh[1]) + tanh[2])
        uint64_t acc = failing_id;
        for (int k = 0; k < K; ++k) {
            uint64_t add_id = shadow->add_node(NodeType::ADD, "tanh_series_add");
            shadow->add_connection(acc,         0, add_id, 0);
            shadow->add_connection(tanh_ids[k], 0, add_id, 1);
            acc = add_id;
        }
        for (const auto& c : outgoing) {
            shadow->add_connection(acc, 0, c.dst_node, c.dst_port);
        }
        break;
    }

    case Hypothesis::RECURRENT_SELF_WIRE: {
        // Add a self-recurrent input to the failing NEURON so it can carry
        // state across timesteps (sequence/memory targets like running parity).
        // BPTT (teacher-forced k=1, already implemented in train()) grows the
        // recurrent weight from a zero init (identity start -> the new input
        // contributes nothing until SGD engages it). add_connection() flags
        // the self-edge as recurrent (cycle detection), and execute() routes
        // it through delay_buffer.
        Node* fn = shadow->get_node(failing_id);
        if (!fn || (fn->get_type() != NodeType::NEURON
                    && fn->get_type() != NodeType::LINEAR)) break;
        auto* nn = static_cast<NeuronNode*>(fn);
        size_t new_port = nn->get_num_weights();   // current count == index of next port
        nn->set_input_count(new_port + 1);
        nn->set_weight(new_port, 0.0);             // zero-init recurrent weight
        // Self-recurrent: own output[0] -> new input port. add_connection
        // detects the cycle (node is its own ancestor) and sets is_recurrent.
        shadow->add_connection(failing_id, 0, failing_id, new_port);
        break;
    }

    case Hypothesis::ATTENTION_MIX: {
        // M2.3 step 2: previous-token attention head. W = the graph's
        // INPUT slots in creation order (the last is the query char);
        // vocab = OUTPUT count. V-port readout: one scalar per class,
        // wired like EMBED_TRUNK — per-OUTPUT ADD (each output keeps its
        // trained path and ADDS its attention logit; identity start).
        // (v1's single-failing-node wiring could not carry V classes —
        // the probe showed the head losing with a structurally
        // insufficient readout.)
        std::vector<uint64_t> in_ids;
        std::vector<uint64_t> output_ids;
        size_t out_count = 0;
        for (const auto& n : shadow->get_nodes()) {
            if (n->get_type() == NodeType::INPUT) in_ids.push_back(n->get_id());
            else if (n->get_type() == NodeType::OUTPUT) {
                ++out_count;
                output_ids.push_back(n->get_id());
            }
        }
        if (in_ids.size() < 4 || out_count < 8) break;
        uint64_t att_id = shadow->add_node(NodeType::ATTENTION, "attention_head");
        auto* an = dynamic_cast<AttentionNode*>(shadow->get_node(att_id));
        if (an) {
            an->set_vocab(out_count);
            // M2.3 table-init from the committed ONE-HOT trunk: its hidden
            // neurons learned per-(slot, code) weights; the slot-AVERAGE per
            // code is a natural embedding seed — retrieval starts from
            // learned embeddings instead of random (the probes measured
            // random-init heads losing every validation cycle). Only when a
            // one-hot trunk exists (embed_onehot_* nodes); input port of
            // trunk neuron h for (slot i, code c) is i*V + c.
            std::vector<NeuronNode*> hnodes;
            bool has_onehot_trunk = false;
            for (const auto& n : shadow->get_nodes()) {
                if (n->get_name().rfind("embed_trunk_h", 0) == 0
                    && n->get_type() == NodeType::NEURON) {
                    hnodes.push_back(static_cast<NeuronNode*>(n.get()));
                } else if (n->get_name().rfind("embed_onehot_", 0) == 0) {
                    has_onehot_trunk = true;
                }
            }
            if (has_onehot_trunk && hnodes.size() >= an->get_dim()) {
                const size_t V = out_count;
                const size_t nslots = in_ids.size();
                int seeded = 0;
                for (size_t c = 0; c < V; ++c) {
                    for (size_t d = 0; d < an->get_dim() && d < hnodes.size(); ++d) {
                        Value sum = 0;
                        size_t cnt = 0;
                        for (size_t i = 0; i < nslots; ++i) {
                            size_t port = i * V + c;
                            if (port < hnodes[d]->get_num_weights()) {
                                sum += hnodes[d]->get_weight(port);
                                ++cnt;
                            }
                        }
                        if (cnt > 0) {
                            an->set_table(c, d, sum / static_cast<Value>(cnt));
                            ++seeded;
                        }
                    }
                }
                if (seeded > 0) {
                    Logger::info("  [ATTENTION] table seeded from trunk ("
                                + std::to_string(seeded) + " entries)");
                }
            }
        }
        for (size_t i = 0; i < in_ids.size(); ++i) {
            shadow->add_connection(in_ids[i], 0, att_id, i);
        }
        // Per-OUTPUT ADDs: output ordinal c reads head port c. Collect
        // ids FIRST (add_node invalidates refs — the EMBED lesson).
        for (size_t c = 0; c < output_ids.size(); ++c) {
            uint64_t oid = output_ids[c];
            std::vector<Connection> outs_in;
            for (const auto& conn : shadow->get_connections()) {
                if (conn.dst_node == oid) outs_in.push_back(conn);
            }
            for (const auto& conn : outs_in) {
                shadow->remove_connection(conn.src_node, conn.src_port,
                                          conn.dst_node, conn.dst_port);
            }
            uint64_t add_id = shadow->add_node(
                NodeType::ADD, "attention_add" + std::to_string(oid));
            for (const auto& conn : outs_in) {
                shadow->add_connection(conn.src_node, conn.src_port, add_id, 0);
            }
            shadow->add_connection(att_id, c, add_id, 1);
            shadow->add_connection(add_id, 0, oid, 0);
        }
        Logger::info("  [ATTENTION] built V-port head over "
                    + std::to_string(in_ids.size()) + " slots (V="
                    + std::to_string(out_count) + ")");
        break;
    }

    case Hypothesis::EMBED_TRUNK: {
        // M2.1 shared dense trunk for many-output tasks (charLM):
        //
        //   INPUT_0 .. INPUT_F ──→ K hidden NEURONs (Xavier) ──→ combine(zero) ──┐
        //                                                                              ├ per-OUTPUT ADD
        //   OUTPUT_c's current input ─────────────────────────────────────────────────┘
        //
        // Every OUTPUT keeps its trained LINEAR path and ADDS the trunk
        // feature (identity start via zero-init combine). SGD then learns
        // which outputs use the shared representation. This is the standard
        // embedding-trunk architecture assembled from existing parts.

        // Collect all INPUT node ids in creation order
        std::vector<uint64_t> in_ids;
        for (const auto& n : shadow->get_nodes()) {
            if (n->get_type() == NodeType::INPUT) in_ids.push_back(n->get_id());
        }
        if (in_ids.empty()) break;

        int K = (hyp.compound_K > 0) ? hyp.compound_K : config::EMBED_TRUNK_K;

        // M2.3 step 1: ONE-HOT expansion per input slot. The raw char
        // codes fed to NEURONs give tanh(w*code) — every symbol on a LINE
        // in embedding space (the linear-embedding limitation measured by
        // the induction probe: 8.01% vs 6.25% chance). Expanding each
        // code to a V-dim one-hot makes the trunk NEURONs TRUE per-symbol
        // embeddings. V = OUTPUT count (input codes share the output
        // alphabet: 0..V-1). Gated: only when the input values look like
        // categorical codes (all in [0, V]) — continuous inputs keep the
        // direct path.
        size_t V_embed = 0;
        for (const auto& n : shadow->get_nodes()) {
            if (n->get_type() == NodeType::OUTPUT) ++V_embed;
        }
        std::vector<uint64_t> trunk_sources;   // what hidden NEURONs read
        std::vector<size_t> src_ports;         // output port on each source
        bool use_onehot = false;
        if (V_embed >= 8 && in_ids.size() >= 4) {
            // Verify code-likeness on the training data.
            bool codes_ok = true;
            uint64_t probe_key = 0; bool have_pk = false;
            for (const auto& kv : input_data_to_graph_) {
                if (kv.second == in_ids[0]) { probe_key = kv.first; have_pk = true; break; }
            }
            if (have_pk && !training_data_.samples.empty()) {
                size_t checked = 0;
                for (const auto& s : training_data_.samples) {
                    auto it = s.inputs.find(probe_key);
                    if (it == s.inputs.end()) continue;
                    Value cv = it->second;
                    if (cv < -0.5 || cv > static_cast<Value>(V_embed) - 0.5
                        || std::abs(cv - std::round(cv)) > 1e-3) {
                        codes_ok = false;
                        Logger::info("  [ONEHOT-GATE] rejected: value "
                                    + std::to_string(cv) + " at V="
                                    + std::to_string(V_embed));
                        break;
                    }
                    if (++checked >= 500) break;
                }
            } else {
                codes_ok = false;
                Logger::info("  [ONEHOT-GATE] rejected: probe key "
                            + std::string(have_pk ? "found" : "MISSING")
                            + ", samples "
                            + std::to_string(training_data_.samples.size()));
            }
            use_onehot = codes_ok;
        }
        if (use_onehot) {
            for (size_t i = 0; i < in_ids.size(); ++i) {
                uint64_t oh_id = shadow->add_node(NodeType::ONEHOT,
                                                  "embed_onehot_" + std::to_string(i));
                Node* oh = shadow->get_node(oh_id);
                if (auto* ohn = dynamic_cast<OneHotNode*>(oh)) {
                    ohn->set_vocab(V_embed);
                }
                shadow->add_connection(in_ids[i], 0, oh_id, 0);
                for (size_t d = 0; d < V_embed; ++d) {
                    trunk_sources.push_back(oh_id);
                    src_ports.push_back(d);
                }
            }
            Logger::info("  [EMBED_TRUNK] one-hot expansion: "
                        + std::to_string(in_ids.size()) + " slots x V="
                        + std::to_string(V_embed));
        } else {
            trunk_sources = in_ids;
            src_ports.assign(in_ids.size(), 0);
        }

        // K hidden neurons over ALL inputs
        std::vector<uint64_t> hidden;
        for (int k = 0; k < K; ++k) {
            uint64_t nid = shadow->add_node(NodeType::NEURON,
                                            "embed_trunk_h" + std::to_string(k));
            Node* nn = shadow->get_node(nid);
            if (nn && nn->get_type() == NodeType::NEURON) {
                static_cast<NeuronNode*>(nn)->set_input_count(trunk_sources.size());
            }
            for (size_t j = 0; j < trunk_sources.size(); ++j) {
                shadow->add_connection(trunk_sources[j], src_ports[j], nid, j);
            }
            hidden.push_back(nid);
        }

        // Zero-init combining neuron (identity start)
        uint64_t combine_id = shadow->add_node(NodeType::NEURON, "embed_trunk_combine");
        Node* cn = shadow->get_node(combine_id);
        if (cn && cn->get_type() == NodeType::NEURON) {
            auto* cnp = static_cast<NeuronNode*>(cn);
            cnp->set_input_count(hidden.size());
            for (size_t k = 0; k < hidden.size(); ++k) {
                cnp->set_weight(k, 0.0);
            }
            cnp->set_bias(0.0);
        }
        for (size_t k = 0; k < hidden.size(); ++k) {
            shadow->add_connection(hidden[k], 0, combine_id, k);
        }

        // Rewire every OUTPUT: current input + trunk feature via per-OUTPUT ADD
        // NOTE: collect output ids FIRST — the loop body calls add_node,
        // which appends to the nodes vector and invalidates any reference
        // into it (use-after-realloc; crashed deterministically on charLM's
        // 65 outputs).
        std::vector<uint64_t> output_ids;
        for (const auto& n : shadow->get_nodes()) {
            if (n->get_type() == NodeType::OUTPUT) output_ids.push_back(n->get_id());
        }
        for (uint64_t oid : output_ids) {
            // Save the OUTPUT's existing incoming connections
            std::vector<Connection> outs_in;
            for (const auto& c : shadow->get_connections()) {
                if (c.dst_node == oid) outs_in.push_back(c);
            }
            for (const auto& c : outs_in) {
                shadow->remove_connection(c.src_node, c.src_port,
                                         c.dst_node, c.dst_port);
            }
            uint64_t add_id = shadow->add_node(NodeType::ADD,
                                               "embed_trunk_add" + std::to_string(oid));
            // original input -> ADD port 0
            for (const auto& c : outs_in) {
                shadow->add_connection(c.src_node, c.src_port, add_id, 0);
            }
            // trunk -> ADD port 1
            shadow->add_connection(combine_id, 0, add_id, 1);
            // ADD -> OUTPUT port 0
            shadow->add_connection(add_id, 0, oid, 0);
        }
        Logger::info("  [EMBED_TRUNK] built K=" + std::to_string(K)
                    + " trunk over " + std::to_string(in_ids.size()) + " inputs");
        break;
    }

    case Hypothesis::IFELSE_PRESERVE: {
        // Both-branches boundary split (stripes20-class):
        //
        //   failing_node ──→ IFELSE(cond) out[0] (true side:  x > thr) ──→ downstream
        //        └────────→ IFELSE(cond) out[1] (false side: x <= thr) ──→ downstream
        //
        // Unlike IFELSE_BOUNDARY_SPLIT (which masks the FALSE side to 0 —
        // designed for cliffs where one region should be suppressed),
        // PRESERVE routes the failing node's output through BOTH branches:
        // each side keeps its trained value, only SEPARATED. Downstream
        // nodes can then learn side-specific corrections. For striped
        // residuals (both sides equally structured) the masked variant
        // regresses training by discarding half the fit.
        //
        // v3 multi-split (compound_K > 1): apply K splits as a chain —
        // each split separates the FALSE output of the previous IFELSE
        // further (nested tree): IFELSE_1 splits x<=t1; IFELSE_2 splits
        // its false side at t2 < t1; etc. Deterministic threshold
        // re-derivation from the SAME diag pairs + committed-margin
        // exclusion (same helper, same inputs => same output as emission).

        uint64_t condition_src = hyp.condition_source_node;
        if (condition_src == 0 || !shadow->get_node(condition_src)) {
            condition_src = failing_id;
        }

        int K = std::max(1, hyp.compound_K);
        // Threshold derivation FIRST (shared by the evidence and generic
        // chain builders). M5.7: measured-evidence candidates use the
        // label-space plateau edges (same helper, same committed-exclusion
        // as emission — the structure validated is the structure emitted).
        // The nested chain requires DESCENDING thresholds: level k splits
        // its FALSE side (x <= e_{k-1}) at e_k < e_{k-1}, so each TRUE
        // branch is the band (e_k, e_{k-1}]. Plateau edges arrive
        // ASCENDING (left-to-right scan) — feeding them unsorted makes
        // every nested band EMPTY (chain collapses; observed as persistent
        // 0.27 pre-train divergence).
        std::vector<Value> ks;
        if (hyp.structural_evidence) {
            for (const auto& ze :
                 detect_zero_plateau_edges(hyp.condition_source_node, *shadow)) {
                ks.push_back(ze.first);
            }
            std::sort(ks.begin(), ks.end(), std::greater<Value>());
            if (static_cast<int>(ks.size()) > K) ks.resize(static_cast<size_t>(K));
        }
        // M7.6(b) GENERALIZED — closed-form region leaves for evidence
        // chains. The generic chain separates regions but every band
        // carries the SAME signal S(x): without per-region learnable
        // leaves it cannot represent piecewise structure, and SGD on the
        // shared gates degrades (measured: chains 0.238 pre-train ->
        // 0.2499 post-train). Instead: each band's true side feeds a
        // 1-input NEURON leaf initialized w=1, b = per-band residual
        // mean (label mean minus the wrapped signal's mean, adjusted
        // for the OUTPUT scale/bias) — the chain becomes a regression
        // tree AT ROUTING TIME. Exact for piecewise-constant targets
        // (stripes20); identity-plus-shift for mixed windows (t22).
        if (K > 1 && hyp.structural_evidence && ks.size() >= 2) {
            // Find OUTPUT and its port-0 source (the final signal).
            uint64_t out_id = 0, s_src = 0;
            size_t s_port = 0;
            for (const auto& n : shadow->get_nodes()) {
                if (n->get_type() == NodeType::OUTPUT) { out_id = n->get_id(); break; }
            }
            if (out_id != 0) {
                for (const auto& c : shadow->get_connections()) {
                    if (c.dst_node == out_id && c.dst_port == 0) {
                        s_src = c.src_node; s_port = c.src_port; break;
                    }
                }
            }
            if (out_id != 0 && s_src != 0) {
                // Per-band label/signal stats from the raw training data
                // (execute the UNMODIFIED shadow clone per sample).
                const size_t nb = ks.size() + 1;
                std::vector<Value> sum_y(nb, 0.0), sum_s(nb, 0.0);
                std::vector<size_t> cnt(nb, 0);
                Value out_scale = 1.0, out_bias = 0.0;
                if (Node* on = shadow->get_node(out_id);
                    on && on->get_type() == NodeType::OUTPUT) {
                    out_scale = static_cast<OutputNode*>(on)->get_scale();
                    out_bias = static_cast<OutputNode*>(on)->get_bias();
                    if (std::abs(out_scale) < 1e-6) out_scale = 1.0;
                }
                {
                    uint64_t cond_key = 0; bool have_ck = false;
                    for (const auto& kv : input_data_to_graph_) {
                        if (kv.second == condition_src) { cond_key = kv.first; have_ck = true; break; }
                    }
                    uint64_t out_key = 0; bool have_ok = false;
                    for (const auto& kv : output_data_to_graph_) {
                        if (kv.second == out_id) { out_key = kv.first; have_ok = true; break; }
                    }
                    if (have_ck && have_ok) {
                        for (const auto& smp : training_data_.samples) {
                            auto xi = smp.inputs.find(cond_key);
                            auto yi = smp.targets.find(out_key);
                            if (xi == smp.inputs.end() || yi == smp.targets.end()) continue;
                            // band index: ks is DESCENDING; band 0 = x > ks[0],
                            // band k = (ks[k], ks[k-1]], band K = x <= ks[K-1].
                            size_t b = 0;
                            while (b < ks.size() && xi->second <= ks[b]) ++b;
                            for (const auto& kv : smp.inputs) {
                                auto git = input_data_to_graph_.find(kv.first);
                                if (git != input_data_to_graph_.end()) {
                                    shadow->set_input_value(git->second, kv.second);
                                }
                            }
                            shadow->execute();
                            sum_y[b] += yi->second;
                            if (const Node* sn = shadow->get_node(s_src)) {
                                sum_s[b] += sn->get_output(s_port);
                            }
                            ++cnt[b];
                            shadow->reset_recurrent_state();
                        }
                    }
                }
                // Build the chain with leaves.
                uint64_t prev_false_src = s_src;
                uint64_t accum = 0;   // chained ADD accumulator over leaves
                auto join_leaf = [&](uint64_t feed_node, size_t feed_port, size_t band) {
                    uint64_t leaf = shadow->add_node(NodeType::NEURON,
                                                     "region_leaf");
                    Node* ln = shadow->get_node(leaf);
                    if (ln && ln->get_type() == NodeType::NEURON) {
                        auto* lnn = static_cast<NeuronNode*>(ln);
                        lnn->set_input_count(1);
                        if (cnt[band] > 0) {
                            Value mean_y = sum_y[band] / static_cast<Value>(cnt[band]);
                            Value mean_s = sum_s[band] / static_cast<Value>(cnt[band]);
                            lnn->set_weight(0, 1.0);
                            lnn->set_bias((mean_y - out_bias) / out_scale - mean_s);
                        } else {
                            lnn->set_weight(0, 1.0);
                            lnn->set_bias(0.0);
                        }
                    }
                    shadow->add_connection(feed_node, feed_port, leaf, 0);
                    if (accum == 0) {
                        accum = leaf;
                    } else if (Node* an = shadow->get_node(accum);
                               an && an->get_type() == NodeType::NEURON) {
                        // previous accumulator is a leaf NEURON: make an ADD
                        uint64_t add_id = shadow->add_node(NodeType::ADD, "region_add");
                        shadow->add_connection(accum, 0, add_id, 0);
                        shadow->add_connection(leaf, 0, add_id, 1);
                        accum = add_id;
                    } else {
                        uint64_t add_id = shadow->add_node(NodeType::ADD, "region_add");
                        // reconnect: prior accumulator feeds new ADD port 0
                        std::vector<Connection> ex;
                        for (const auto& c : shadow->get_connections()) {
                            if (c.src_node == accum) ex.push_back(c);
                        }
                        for (const auto& c : ex) {
                            shadow->remove_connection(c.src_node, c.src_port,
                                                      c.dst_node, c.dst_port);
                            shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
                        }
                        shadow->add_connection(accum, 0, add_id, 0);
                        shadow->add_connection(leaf, 0, add_id, 1);
                        accum = add_id;
                    }
                };
                for (size_t k = 0; k < ks.size(); ++k) {
                    uint64_t threshold_id = shadow->add_node(NodeType::CONSTANT,
                                                             "ifelse_threshold");
                    if (Node* cn = shadow->get_node(threshold_id);
                        cn && cn->get_type() == NodeType::CONSTANT) {
                        static_cast<ConstantNode*>(cn)->set_value(ks[k]);
                    }
                    uint64_t greater_id = shadow->add_node(NodeType::GREATER,
                                                           "preserve_cond");
                    shadow->add_connection(condition_src, 0, greater_id, 0);
                    shadow->add_connection(threshold_id, 0, greater_id, 1);
                    uint64_t ifelse_id = shadow->add_node(NodeType::IFELSE,
                                                          "preserve_ifelse");
                    shadow->add_connection(greater_id, 0, ifelse_id, 0);
                    shadow->add_connection(prev_false_src, 0, ifelse_id, 1);
                    join_leaf(ifelse_id, 0, k);          // true side = band k
                    prev_false_src = ifelse_id;
                }
                join_leaf(prev_false_src, 1, ks.size());  // deepest false band
                // Rewire: accumulator -> OUTPUT (replacing S -> OUTPUT).
                shadow->remove_connection(s_src, s_port, out_id, 0);
                shadow->add_connection(accum, 0, out_id, 0);
                Logger::info("  [EVIDENCE-CHAIN] K=" + std::to_string(ks.size())
                             + " region leaves (closed-form init), bands n="
                             + std::to_string(nb));
                break;
            }
            // No OUTPUT/source found: fall through to the generic chain.
        }

        if (K > 1) {
            // CART re-derivation for ordinary candidates (evidence ks
            // were already derived above; the evidence builder returned
            // before reaching here).
            std::vector<std::pair<Value, Value>> vx_m;
            if (ks.size() < 2) {
                for (size_t i = 0; i < diag.targets.size() && i < diag.local_inputs.size(); ++i) {
                    if (!diag.local_inputs[i].empty()) {
                        auto it = diag.local_inputs[i].find(hyp.condition_source_node);
                        if (it == diag.local_inputs[i].end()) {
                            vx_m.emplace_back(diag.local_inputs[i].begin()->second, diag.targets[i]);
                        } else {
                            vx_m.emplace_back(it->second, diag.targets[i]);
                        }
                    }
                }
                ks = error_weighted_multi_split(vx_m, K);
            }
            if (ks.size() >= 2) {
                // Chain: each IFELSE's FALSE output feeds the next IFELSE.
                uint64_t prev_false_src = failing_id;
                uint64_t first_downstream = 0;
                size_t first_downstream_port = 0;
                uint64_t preserve_add = 0;
                for (size_t k = 0; k < ks.size(); ++k) {
                    uint64_t threshold_id = shadow->add_node(NodeType::CONSTANT, "ifelse_threshold");
                    Node* cnode = shadow->get_node(threshold_id);
                    if (cnode && cnode->get_type() == NodeType::CONSTANT) {
                        static_cast<ConstantNode*>(cnode)->set_value(ks[k]);
                    }
                    uint64_t ifelse_id = shadow->add_node(NodeType::IFELSE, "preserve_ifelse");
                    shadow->add_connection(condition_src, 0, ifelse_id, 0);
                    shadow->add_connection(prev_false_src, 0, ifelse_id, 1);

                    if (k == 0) {
                        // First IFELSE: re-route failing node's outgoing.
                        std::vector<Connection> outgoing;
                        for (const auto& c : shadow->get_connections()) {
                            if (c.src_node == failing_id) outgoing.push_back(c);
                        }
                        for (const auto& c : outgoing) {
                            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
                        }
                        for (const auto& c : outgoing) {
                            if (first_downstream == 0) {
                                first_downstream = c.dst_node;
                                first_downstream_port = c.dst_port;
                            }
                            Node* dn = shadow->get_node(c.dst_node);
                            if (dn && dn->get_type() == NodeType::OUTPUT) {
                                // OUTPUT reads only port 0 — same GATED fix
                                // as the single-split path: per-port pass-
                                // through NEURON gates + ADD (bare ADD is a
                                // functional no-op for IFELSE semantics).
                                uint64_t g0 = shadow->add_node(NodeType::NEURON, "preserve_gate0");
                                uint64_t g1 = shadow->add_node(NodeType::NEURON, "preserve_gate1");
                                Node* gn0 = shadow->get_node(g0);
                                Node* gn1 = shadow->get_node(g1);
                                if (gn0 && gn0->get_type() == NodeType::NEURON) {
                                    static_cast<NeuronNode*>(gn0)->set_input_count(1);
                                    static_cast<NeuronNode*>(gn0)->set_weight(0, 1.0);
                                    static_cast<NeuronNode*>(gn0)->set_bias(0.0);
                                }
                                if (gn1 && gn1->get_type() == NodeType::NEURON) {
                                    static_cast<NeuronNode*>(gn1)->set_input_count(1);
                                    static_cast<NeuronNode*>(gn1)->set_weight(0, 1.0);
                                    static_cast<NeuronNode*>(gn1)->set_bias(0.0);
                                }
                                shadow->add_connection(ifelse_id, 0, g0, 0);
                                shadow->add_connection(ifelse_id, 1, g1, 0);
                                uint64_t add0 = shadow->add_node(NodeType::ADD, "preserve_out_add");
                                shadow->add_connection(g0, 0, add0, 0);
                                shadow->add_connection(g1, 0, add0, 1);
                                shadow->add_connection(add0, 0, c.dst_node, c.dst_port);
                            } else if (dn && (dn->get_type() == NodeType::NEURON
                                       || dn->get_type() == NodeType::LINEAR)) {
                                shadow->add_connection(ifelse_id, 0, c.dst_node, c.dst_port);
                                auto* dnn = static_cast<NeuronNode*>(dn);
                                size_t next_port = dnn->get_num_weights();
                                dnn->set_input_count(next_port + 1);
                                dnn->set_weight(next_port, dnn->get_weight(c.dst_port));
                                shadow->add_connection(ifelse_id, 1, c.dst_node, next_port);
                            } else {
                                shadow->add_connection(ifelse_id, 0, c.dst_node, c.dst_port);
                            }
                        }
                    } else {
                        // Subsequent IFELSEs: TRUE side routes to the same
                        // downstream target as the first IFELSE (mirrored
                        // port, same init weight). FALSE side continues
                        // the chain via prev_false_src (set below).
                        Node* dn0 = shadow->get_node(first_downstream);
                        if (dn0 && (dn0->get_type() == NodeType::NEURON
                                    || dn0->get_type() == NodeType::LINEAR)) {
                            auto* dnn = static_cast<NeuronNode*>(dn0);
                            size_t np2 = dnn->get_num_weights();
                            dnn->set_input_count(np2 + 1);
                            dnn->set_weight(np2, dnn->get_weight(first_downstream_port));
                            shadow->add_connection(ifelse_id, 0, first_downstream, np2);
                        } else if (dn0 && dn0->get_type() == NodeType::OUTPUT) {
                            // OUTPUT takes one input: interpose an ADD once,
                            // then every subsequent side joins it.
                            Node* addchk = shadow->get_node(preserve_add);
                            if (!addchk) {
                                preserve_add = shadow->add_node(NodeType::ADD, "preserve_add");
                                std::vector<Connection> ex0;
                                for (const auto& c0 : shadow->get_connections()) {
                                    if (c0.dst_node == first_downstream) ex0.push_back(c0);
                                }
                                for (const auto& c0 : ex0) {
                                    shadow->remove_connection(c0.src_node, c0.src_port,
                                                             c0.dst_node, c0.dst_port);
                                    shadow->add_connection(c0.src_node, c0.src_port, preserve_add, 0);
                                }
                                shadow->add_connection(preserve_add, 0, first_downstream, 0);
                            }
                            // join this ifelse's true side to the ADD (port grows)
                            Node* an = shadow->get_node(preserve_add);
                            if (an) {
                                // ADD supports 2 inputs; use a NEURON-like
                                // accumulate via chained ADDs if >2 sides —
                                // simplest: connect to port min(existing,1)
                                size_t used = 0;
                                for (const auto& c0 : shadow->get_connections()) {
                                    if (c0.dst_node == preserve_add) ++used;
                                }
                                shadow->add_connection(ifelse_id, 0, preserve_add,
                                                       used >= 1 ? 1 : 0);
                            }
                        }
                    }
                    prev_false_src = ifelse_id;   // chain continues on false side
                }
                // Final false side routes downstream: the deepest region
                // (x <= last threshold) must reach a consumer or the whole
                // chain is dead code (observed: chains committed then
                // eliminated by compile because the chain end went nowhere).
                if (first_downstream != 0) {
                    Node* dnf = shadow->get_node(first_downstream);
                    if (dnf && (dnf->get_type() == NodeType::NEURON
                                || dnf->get_type() == NodeType::LINEAR)) {
                        auto* dnfnn = static_cast<NeuronNode*>(dnf);
                        size_t npf = dnfnn->get_num_weights();
                        dnfnn->set_input_count(npf + 1);
                        dnfnn->set_weight(npf, dnfnn->get_weight(first_downstream_port));
                        shadow->add_connection(prev_false_src, 0, first_downstream, npf);
                    } else if (dnf && dnf->get_type() == NodeType::OUTPUT) {
                        // OUTPUT downstream takes single input; route the
                        // chain end through an ADD combiner instead.
                        uint64_t add_end = shadow->add_node(NodeType::ADD, "preserve_end_add");
                        // re-route first_downstream's current input? Simpler:
                        // the first IFELSE's true side already feeds it; ADD
                        // the chain end alongside via a second OUTPUT is not
                        // possible — so combine at an ADD then rewire.
                        // Minimal: disconnect existing input, feed via ADD.
                        std::vector<Connection> ex;
                        for (const auto& c : shadow->get_connections()) {
                            if (c.dst_node == first_downstream) ex.push_back(c);
                        }
                        for (const auto& c : ex) {
                            shadow->remove_connection(c.src_node, c.src_port,
                                                     c.dst_node, c.dst_port);
                            shadow->add_connection(c.src_node, c.src_port, add_end, 0);
                        }
                        shadow->add_connection(prev_false_src, 0, add_end, 1);
                        shadow->add_connection(add_end, 0, first_downstream, 0);
                    }
                }
                Logger::info("  [PRESERVE-MULTI] built K=" + std::to_string(ks.size())
                            + " split chain");
                break;
            }
            // fallthrough: multi-split degenerated -> single split below
        }

        uint64_t threshold_id = shadow->add_node(NodeType::CONSTANT, "ifelse_threshold");
        Node* cnode = shadow->get_node(threshold_id);
        if (cnode && cnode->get_type() == NodeType::CONSTANT) {
            static_cast<ConstantNode*>(cnode)->set_value(hyp.split_threshold);
        }
        uint64_t ifelse_id = shadow->add_node(NodeType::IFELSE, "preserve_ifelse");
        shadow->add_connection(condition_src, 0, ifelse_id, 0);
        shadow->add_connection(failing_id,  0, ifelse_id, 1);

        // Route BOTH output ports downstream (replacing the direct edge).
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) outgoing.push_back(c);
        }
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }
        for (const auto& c : outgoing) {
            // true-side downstream keeps the original destination port,
            // false-side goes to the NEXT free input port of the same node.
            Node* dn = shadow->get_node(c.dst_node);
            if (dn && dn->get_type() == NodeType::OUTPUT) {
                // OUTPUT reads ONLY input port 0 (get_value = scale*in[0]+bias).
                // A second port would be silently ignored — the split would be
                // structurally present but functionally dead (the exact bug
                // that made committed chains invisible). Route each IFELSE
                // port through its own PASS-THROUGH NEURON GATE, then ADD:
                //   ifelse.out[0] → NEURON_g0(w=1) ──┐
                //                                    ├→ ADD → OUTPUT port 0
                //   ifelse.out[1] → NEURON_g1(w=1) ──┘
                // A bare ADD would be a functional NO-OP (out[0]+out[1] =
                // the value itself, since IFELSE zeroes the inactive port).
                // The gates let SGD learn SIDE-SPECIFIC corrections — the
                // actual point of the split. w=1 pass-through start: the
                // active side contributes tanh(x) initially (bounded), then
                // each side specializes independently.
                uint64_t g0 = shadow->add_node(NodeType::NEURON, "preserve_gate0");
                uint64_t g1 = shadow->add_node(NodeType::NEURON, "preserve_gate1");
                Node* gn0 = shadow->get_node(g0);
                Node* gn1 = shadow->get_node(g1);
                if (gn0 && gn0->get_type() == NodeType::NEURON) {
                    static_cast<NeuronNode*>(gn0)->set_input_count(1);
                    static_cast<NeuronNode*>(gn0)->set_weight(0, 1.0);
                    static_cast<NeuronNode*>(gn0)->set_bias(0.0);
                }
                if (gn1 && gn1->get_type() == NodeType::NEURON) {
                    static_cast<NeuronNode*>(gn1)->set_input_count(1);
                    static_cast<NeuronNode*>(gn1)->set_weight(0, 1.0);
                    static_cast<NeuronNode*>(gn1)->set_bias(0.0);
                }
                shadow->add_connection(ifelse_id, 0, g0, 0);
                shadow->add_connection(ifelse_id, 1, g1, 0);
                uint64_t add_id = shadow->add_node(NodeType::ADD, "preserve_out_add");
                shadow->add_connection(g0, 0, add_id, 0);
                shadow->add_connection(g1, 0, add_id, 1);
                shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
            } else if (dn && (dn->get_type() == NodeType::NEURON
                        || dn->get_type() == NodeType::LINEAR)) {
                shadow->add_connection(ifelse_id, 0, c.dst_node, c.dst_port);
                auto* dnn = static_cast<NeuronNode*>(dn);
                size_t next_port = dnn->get_num_weights();
                dnn->set_input_count(next_port + 1);
                dnn->set_weight(next_port, dnn->get_weight(c.dst_port));  // same init weight
                shadow->add_connection(ifelse_id, 1, c.dst_node, next_port);
            } else {
                // Other non-trainable downstream: only true side routed (best effort)
                shadow->add_connection(ifelse_id, 0, c.dst_node, c.dst_port);
            }
        }
        break;
    }

    case Hypothesis::MUX_INJECTION: {
        // GREATER(cond_src, thr) → MUX(g, a, b) → zero-gain NEURON → ADD
        //
        //   cond_src ──→ GREATER ──┐
        //   a ────────────────────→ MUX ──→ NEURON(0) ──→ ADD → downstream
        //   b ────────────────────→↑                        ↑
        //   failing_node ──────────────────────────────────┘
        //
        // The GREATER thresholding gives the condition true boolean
        // semantics (MUX execute uses !=0 truthiness). Zero-gain output
        // neuron = identity start; SGD grows the selected branch's share.

        uint64_t a = hyp.multiply_source_a;
        uint64_t b = hyp.multiply_source_b;
        uint64_t cs = hyp.bool_source_a;
        if (a == 0 || b == 0 || cs == 0) break;
        if (!shadow->get_node(a) || !shadow->get_node(b) || !shadow->get_node(cs)) break;

        uint64_t thr_id = shadow->add_node(NodeType::CONSTANT, "mux_threshold");
        Node* cn = shadow->get_node(thr_id);
        if (cn && cn->get_type() == NodeType::CONSTANT) {
            static_cast<ConstantNode*>(cn)->set_value(hyp.split_threshold);
        }

        uint64_t gt_id = shadow->add_node(NodeType::GREATER, "mux_cond");
        shadow->add_connection(cs, 0, gt_id, 0);
        shadow->add_connection(thr_id, 0, gt_id, 1);

        uint64_t mux_id = shadow->add_node(NodeType::MUX, "mux_select");
        shadow->add_connection(gt_id, 0, mux_id, 0);   // condition
        shadow->add_connection(a, 0, mux_id, 1);       // true branch
        shadow->add_connection(b, 0, mux_id, 2);       // false branch

        uint64_t gain_id = shadow->add_node(NodeType::NEURON, "mux_gain");
        Node* gn = shadow->get_node(gain_id);
        if (gn && gn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(gn)->set_input_count(1);
            static_cast<NeuronNode*>(gn)->set_weight(0, 0.0);
            static_cast<NeuronNode*>(gn)->set_bias(0.0);
        }
        shadow->add_connection(mux_id, 0, gain_id, 0);
        // least-squares gain init: the selection starts at optimal scale
        if (gn && gn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(gn)->set_weight(
                0, compute_gain_init(*shadow, mux_id, failing_id, diag));
        }

        std::vector<Connection> outgoing;
        for (const auto& conn : shadow->get_connections()) {
            if (conn.src_node == failing_id) outgoing.push_back(conn);
        }
        for (const auto& conn : outgoing) {
            shadow->remove_connection(conn.src_node, conn.src_port, conn.dst_node, conn.dst_port);
        }
        uint64_t add_id = shadow->add_node(NodeType::ADD, "mux_combine");
        shadow->add_connection(failing_id, 0, add_id, 0);
        shadow->add_connection(gain_id, 0, add_id, 1);
        for (const auto& conn : outgoing) {
            shadow->add_connection(add_id, 0, conn.dst_node, conn.dst_port);
        }
        break;
    }

    case Hypothesis::DELAY_LINE: {
        // k delayed copies of the raw INPUT signal as failing-node features:
        //
        //   INPUT(u) ──[taps=1]──→ failing node port p
        //   INPUT(u) ──[taps=2]──→ failing node port p+1
        //   ...
        //   INPUT(u) ──[taps=k]──→ failing node port p+k-1
        //
        // Delayed FORWARD edges (delay_taps>0, not recurrent): execute()
        // maintains history rings for them (M4.1 gap fix). Zero-init
        // weights: identity start; SGD learns which lags matter. Commits
        // stack: a second DELAY_LINE commit extends the lag window.

        // Find the INPUT that feeds the failing node — TRANSITIVELY. After
        // stack commits the bottleneck is downstream of NEURONs with no
        // direct INPUT edge; walk ancestors to reach the raw signal.
        uint64_t input_src = 0;
        {
            // Direct edge first (common case)
            for (const auto& c : shadow->get_connections()) {
                if (c.dst_node == failing_id) {
                    const Node* sn = shadow->get_node(c.src_node);
                    if (sn && sn->get_type() == NodeType::INPUT) {
                        input_src = c.src_node;
                        break;
                    }
                }
            }
            if (input_src == 0) {
                // Transitive: any INPUT among the failing node's ancestors.
                // Prefer the primary (first-seen in connection order) INPUT.
                for (const auto& c : shadow->get_connections()) {
                    const Node* sn = shadow->get_node(c.src_node);
                    if (sn && sn->get_type() == NodeType::INPUT) {
                        // is this input an ancestor of failing_id?
                        auto anc = shadow->get_ancestors(failing_id);
                        if (std::find(anc.begin(), anc.end(), c.src_node) != anc.end()) {
                            input_src = c.src_node;
                            break;
                        }
                    }
                }
            }
        }
        if (input_src == 0) {
            Logger::info("  [DELAY_LINE-DBG] routing break: no INPUT ancestor for failing node "
                        + std::to_string(failing_id));
            break;
        }

        Node* fn = shadow->get_node(failing_id);
        if (!fn || (fn->get_type() != NodeType::NEURON
                    && fn->get_type() != NodeType::LINEAR)) break;
        auto* nn = static_cast<NeuronNode*>(fn);

        int k = (hyp.compound_K > 0)
              ? std::min(hyp.compound_K, static_cast<int>(MAX_DELAY_TAPS))
              : config::DELAY_LINE_K;
        size_t first_port = nn->get_num_weights();
        nn->set_input_count(first_port + static_cast<size_t>(k));
        for (int lag = 1; lag <= k; ++lag) {
            size_t port = first_port + static_cast<size_t>(lag - 1);
            nn->set_weight(port, 0.0);
            // INPUT -> failing node, delayed by `lag` steps. add_connection
            // won't flag this recurrent (no cycle); delay_taps carries the
            // temporal read. set AFTER the connection exists.
            shadow->add_connection(input_src, 0, failing_id, port);
            shadow->set_connection_delay_taps(input_src, 0, failing_id, port, lag);
        }
        break;
    }

    case Hypothesis::RECURRENT_MULTI_TAP: {
        // K self-recurrent inputs at delays 1..K. Each port reads the
        // node's own output from k steps back via Connection::delay_taps
        // (history ring). Zero-init weights: identity start; SGD learns
        // which taps matter (NARMA-30 needs a ~30-step sum; even K=4 taps
        // exponentially extends effective memory via composition).
        Node* fn = shadow->get_node(failing_id);
        if (!fn || (fn->get_type() != NodeType::NEURON
                    && fn->get_type() != NodeType::LINEAR)) break;
        auto* nn = static_cast<NeuronNode*>(fn);
        int K = (hyp.compound_K > 0)
              ? std::min(hyp.compound_K, static_cast<int>(MAX_DELAY_TAPS))
              : config::RECURRENT_MULTI_TAP_K;
        size_t first_port = nn->get_num_weights();
        nn->set_input_count(first_port + static_cast<size_t>(K));
        for (int k = 1; k <= K; ++k) {
            size_t port = first_port + static_cast<size_t>(k - 1);
            nn->set_weight(port, 0.0);
            shadow->add_connection(failing_id, 0, failing_id, port);
            shadow->set_connection_delay_taps(failing_id, 0, failing_id, port, k);
        }
        break;
    }

    case Hypothesis::SIN_INJECTION: {
        // sin(wx+b): INPUT 鈫?NEURON(w init) 鈫?SIN 鈫?ADD.
        // For oscillating bounded residuals (sin(kx)) where TANH_SERIES
        // can't provide enough bumps. Multiple commits stack via ADD.
        // w inits to hyp.sin_freq_init when the emission estimated the
        // frequency from zero crossings (else 0 = legacy identity start).
        uint64_t src = hyp.multiply_source_a;
        if (src == 0 || !shadow->get_node(src)) break;
        uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "sin_neuron");
        Node* nn = shadow->get_node(neuron_id);
        if (nn && nn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(nn)->set_input_count(1);
            static_cast<NeuronNode*>(nn)->set_weight(0, hyp.sin_freq_init);
            static_cast<NeuronNode*>(nn)->set_bias(0.0);
        }
        shadow->add_connection(src, 0, neuron_id, 0);
        uint64_t sin_id = shadow->add_node(NodeType::SIN, "sin_act");
        shadow->add_connection(neuron_id, 0, sin_id, 0);
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections())
            if (c.src_node == failing_id) outgoing.push_back(c);
        for (const auto& c : outgoing)
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        uint64_t add_id = shadow->add_node(NodeType::ADD, "sin_combine");
        shadow->add_connection(failing_id, 0, add_id, 0);
        shadow->add_connection(sin_id, 0, add_id, 1);
        for (const auto& c : outgoing)
            shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
        break;
    }

    case Hypothesis::BOOLEAN_COMPOSE: {
        // Compose two boolean signals via XOR/AND/OR.
        //
        // Architecture (when sources need thresholding):
        //   src_a 鈹€鈹€鈫?GREATER 鈹€鈹€鈹?        //                        XOR/AND/OR 鈹€鈹€鈫?[feature output]
        //   src_b 鈹€鈹€鈫?GREATER 鈹€鈹€鈹?        //
        // Why threshold: XOR/AND/OR execute via `inputs_[i] != 0.0` truthiness.
        // For continuous inputs in [-1,1] or [0,1], almost every value is
        // "truthy" 鈥?so XOR(0.3, 0.7) returns 0 (both truthy). Thresholding
        // via GREATER(src, threshold) gives true boolean semantics.
        //
        // Sources that already produce booleans (GREATER, XOR, etc.) skip
        // the thresholding step.

        uint64_t src_a = hyp.bool_source_a;
        uint64_t src_b = hyp.bool_source_b;
        if (src_a == 0 || !shadow->get_node(src_a)) break;
        if (src_b == 0 || !shadow->get_node(src_b)) break;

        auto is_boolean = [](NodeType t) {
            return t == NodeType::GREATER     || t == NodeType::LESS      ||
                   t == NodeType::GREATER_EQUAL|| t == NodeType::LESS_EQUAL||
                   t == NodeType::EQUAL        || t == NodeType::NOT_EQUAL ||
                   t == NodeType::XOR          || t == NodeType::AND       ||
                   t == NodeType::OR           || t == NodeType::NOT;
        };

        // Threshold src_a if not already boolean
        uint64_t bool_a = src_a;
        const Node* na = shadow->get_node(src_a);
        if (!na || !is_boolean(na->get_type())) {
            uint64_t thr_id = shadow->add_node(NodeType::CONSTANT, "bool_thr_a");
            Node* cn = shadow->get_node(thr_id);
            if (cn && cn->get_type() == NodeType::CONSTANT) {
                static_cast<ConstantNode*>(cn)->set_value(hyp.bool_threshold);
            }
            uint64_t gt_id = shadow->add_node(NodeType::GREATER, "bool_a");
            shadow->add_connection(src_a, 0, gt_id, 0);
            shadow->add_connection(thr_id, 0, gt_id, 1);
            bool_a = gt_id;
        }

        // Threshold src_b if not already boolean
        uint64_t bool_b = src_b;
        const Node* nb = shadow->get_node(src_b);
        if (!nb || !is_boolean(nb->get_type())) {
            uint64_t thr_id = shadow->add_node(NodeType::CONSTANT, "bool_thr_b");
            Node* cn = shadow->get_node(thr_id);
            if (cn && cn->get_type() == NodeType::CONSTANT) {
                static_cast<ConstantNode*>(cn)->set_value(hyp.bool_threshold);
            }
            uint64_t gt_id = shadow->add_node(NodeType::GREATER, "bool_b");
            shadow->add_connection(src_b, 0, gt_id, 0);
            shadow->add_connection(thr_id, 0, gt_id, 1);
            bool_b = gt_id;
        }

        // Compose
        NodeType op = hyp.bool_op;
        if (op != NodeType::XOR && op != NodeType::AND && op != NodeType::OR) {
            op = NodeType::XOR;  // safe default
        }
        uint64_t compose_id = shadow->add_node(op, "bool_compose");
        shadow->add_connection(bool_a, 0, compose_id, 0);
        shadow->add_connection(bool_b, 0, compose_id, 1);

        // Route compose output as new feature (same routing as MULTIPLY_INJECTION)
        Node* dst = shadow->get_node(failing_id);
        if (dst && (dst->get_type() == NodeType::NEURON
                    || dst->get_type() == NodeType::LINEAR)) {
            size_t new_port = static_cast<NeuronNode*>(dst)->get_num_weights();
            static_cast<NeuronNode*>(dst)->set_input_count(new_port + 1);
            shadow->add_connection(compose_id, 0, failing_id, new_port);
        } else {
            uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "bool_feature_neuron");
            Node* nn = shadow->get_node(neuron_id);
            if (nn && nn->get_type() == NodeType::NEURON) {
                static_cast<NeuronNode*>(nn)->set_input_count(1);
            }
            shadow->add_connection(compose_id, 0, neuron_id, 0);

            std::vector<Connection> outgoing;
            for (const auto& c : shadow->get_connections()) {
                if (c.src_node == failing_id) outgoing.push_back(c);
            }
            for (const auto& c : outgoing) {
                shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
            }

            uint64_t add_id = shadow->add_node(NodeType::ADD, "bool_combine");
            shadow->add_connection(failing_id, 0, add_id, 0);
            shadow->add_connection(neuron_id, 0, add_id, 1);
            for (const auto& c : outgoing) {
                shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
            }
        }
        break;
    }

    case Hypothesis::DEEP_INSERTION: {
        // Residual depth: failing_node 鈫?NEURON(zero-init) 鈫?TANH 鈫?ADD
        //                                              鈫?        //   failing_node 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?        //
        // Creates a deeper processing path. At init: TANH(0路x+0)=0, so
        // ADD(old, 0) = old (identity start). SGD grows the NEURON weight
        // and bias, allowing the TANH to add nonlinear corrections to the
        // failing node's output. This adds DEPTH (hierarchical features)
        // unlike NEURON_TANH_INJECTION which adds WIDTH (parallel neurons
        // with the same inputs).

        // Collect outgoing connections from failing_node
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) outgoing.push_back(c);
        }
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }

        // 1. NEURON (zero-init): receives failing_node's output
        uint64_t neuron_id = shadow->add_node(NodeType::NEURON, "deep_neuron");
        Node* nn = shadow->get_node(neuron_id);
        if (nn && nn->get_type() == NodeType::NEURON) {
            static_cast<NeuronNode*>(nn)->set_input_count(1);
            static_cast<NeuronNode*>(nn)->set_weight(0, 0.0);
            static_cast<NeuronNode*>(nn)->set_bias(0.0);
        }
        shadow->add_connection(failing_id, 0, neuron_id, 0);

        // 2. TANH: bounded nonlinearity
        uint64_t tanh_id = shadow->add_node(NodeType::TANH, "deep_tanh");
        shadow->add_connection(neuron_id, 0, tanh_id, 0);

        // 3. ADD combiner: failing_node + TANH(NEURON(failing_node))
        uint64_t add_id = shadow->add_node(NodeType::ADD, "deep_combine");
        shadow->add_connection(failing_id, 0, add_id, 0);
        shadow->add_connection(tanh_id,   0, add_id, 1);

        // 4. Re-route: ADD 鈫?original downstream
        for (const auto& c : outgoing) {
            shadow->add_connection(add_id, 0, c.dst_node, c.dst_port);
        }
        break;
    }

    case Hypothesis::MULTI_LAYER_STACK: {
        // Replace the failing single neuron with a 2-layer MLP:
        //
        //   INPUT(s) 鈫?NEURON_1a 鈹€鈹€鈹?        //            鈫?NEURON_1b 鈹€鈹€鈫?NEURON_combine(K) 鈫?OUTPUT
        //            鈫?NEURON_1c 鈹€鈹€鈹?        //             ...K total
        //
        // The K first-layer neurons each receive the SAME inputs as the
        // failing node, with independent Xavier init (diverse features).
        // The combining neuron has near-zero init with tiny symmetry
        // breaking so the stack starts as identity (output 鈮?0, leaving
        // the existing OUTPUT bias to carry the prediction). SGD then
        // grows the combining weights to select useful features.

        // Collect failing node's input connections
        std::vector<Connection> failing_inputs;
        for (const auto& c : shadow->get_connections()) {
            if (c.dst_node == failing_id) failing_inputs.push_back(c);
        }
        if (failing_inputs.empty()) break;

        // Collect outgoing connections from failing node
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) outgoing.push_back(c);
        }
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }

        // Determine hidden width K (from hypothesis or default)
        size_t num_inputs = failing_inputs.size();
        int K = (hyp.compound_K > 0)
              ? std::min(hyp.compound_K, config::MULTI_LAYER_STACK_K_MAX)
              : std::min(config::MULTI_LAYER_STACK_K_MAX,
                         std::max(config::MULTI_LAYER_STACK_K,
                                  static_cast<int>(num_inputs)));

        // Create K first-layer neurons (Xavier init via set_input_count)
        std::vector<uint64_t> hidden_ids;
        for (int k = 0; k < K; ++k) {
            uint64_t nid = shadow->add_node(NodeType::NEURON,
                                            "stack_l1_" + std::to_string(k));
            Node* nn = shadow->get_node(nid);
            if (nn && nn->get_type() == NodeType::NEURON) {
                static_cast<NeuronNode*>(nn)->set_input_count(num_inputs);
            }
            // Wire same inputs as failing node
            for (const auto& c : failing_inputs) {
                shadow->add_connection(c.src_node, c.src_port, nid, c.dst_port);
            }
            hidden_ids.push_back(nid);
        }

        // Create combining neuron (near-zero init for identity start)
        uint64_t combine_id = shadow->add_node(NodeType::NEURON, "stack_combine");
        Node* cn = shadow->get_node(combine_id);
        if (cn && cn->get_type() == NodeType::NEURON) {
            auto* cnp = static_cast<NeuronNode*>(cn);
            cnp->set_input_count(K);
            // Near-zero init with tiny symmetry breaking
            constexpr double STACK_EPS = 1e-3;
            for (int k = 0; k < K; ++k) {
                cnp->set_weight(k, STACK_EPS * (static_cast<double>(k)
                              - static_cast<double>(K) / 2.0 + 0.5));
            }
            cnp->set_bias(0.0);
        }

        // Wire hidden neurons 鈫?combining neuron
        for (int k = 0; k < K; ++k) {
            shadow->add_connection(hidden_ids[k], 0, combine_id, k);
        }

        // Route combining neuron 鈫?original downstream (REPLACE routing).
        // The failing node is orphaned (dead-code-eliminated). The new
        // bottleneck becomes the combine neuron, so subsequent stacks
        // go DEEPER (stack of stacks = added layers). Empirically better
        // than residual routing, which keeps the bottleneck at the old
        // node and only grows width.
        for (const auto& c : outgoing) {
            shadow->add_connection(combine_id, 0, c.dst_node, c.dst_port);
        }
        break;
    }

    case Hypothesis::PATCH_POOLING: {
        // Build one average-pool LINEAR node per patch_size虏 block of the
        // image, then wire all pooled features into the failing node.
        //
        //   pixels of patch (py,px) 鈹€鈹€鈫?LINEAR(pool_py_px, w=1/k each) 鈹€鈹€鈹?        //                                                                   鈹溾啋 failing node (new ports, zero-init)
        //   ... one per patch ...                                        鈹€鈹€鈹?        //
        // A LINEAR node with uniform 1/k weights computes the exact block
        // mean; SGD can refine weights into learned filters afterwards.
        // Failing node's new input weights start at 0 鈫?identity start.

        int patch = (hyp.compound_K > 0) ? hyp.compound_K : config::PATCH_POOL_PATCH_SIZE;

        // Graph INPUT node ids in creation order == column order
        std::vector<uint64_t> input_ids;
        for (const auto& n : shadow->get_nodes()) {
            if (n->get_type() == NodeType::INPUT) input_ids.push_back(n->get_id());
        }
        // Layout: channels=1 鈫?grayscale, index y*side+x.
        //         channels=3 鈫?pixel-interleaved RGB, index (y*side+x)*3+c.
        int channels = 1;
        int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(input_ids.size()))));
        if (side * side != static_cast<int>(input_ids.size())) {
            int s3 = static_cast<int>(std::lround(std::sqrt(input_ids.size() / 3.0)));
            if (3 * s3 * s3 == static_cast<int>(input_ids.size())) { side = s3; channels = 3; }
        }
        if (side < 2 || channels * side * side != static_cast<int>(input_ids.size())) break;
        if (side % patch != 0) break;
        int n_blocks = side / patch;               // blocks per axis
        int patch_pixels = patch * patch;

        Node* dst = shadow->get_node(failing_id);
        if (!dst || (dst->get_type() != NodeType::NEURON
                     && dst->get_type() != NodeType::LINEAR)) break;
        auto* dstn = static_cast<NeuronNode*>(dst);
        size_t first_new_port = dstn->get_num_weights();

        // Pre-reserve the failing node's input count (weights Xavier-init,
        // then zeroed below) so connections can attach.
        int total_pools = channels * n_blocks * n_blocks;
        dstn->set_input_count(first_new_port + static_cast<size_t>(total_pools));
        for (size_t p = first_new_port; p < dstn->get_num_weights(); ++p) {
            dstn->set_weight(p, 0.0);   // zero-init: pooled features contribute nothing until SGD engages
        }

        int pool_idx = 0;
        for (int c = 0; c < channels; ++c) {
        for (int by = 0; by < n_blocks; ++by) {
            for (int bx = 0; bx < n_blocks; ++bx) {
                std::string pname = channels == 1
                    ? "pool_" + std::to_string(by) + "_" + std::to_string(bx)
                    : "pool_c" + std::to_string(c) + "_" + std::to_string(by) + "_" + std::to_string(bx);
                uint64_t pool_id = shadow->add_node(NodeType::LINEAR, pname);
                Node* pn = shadow->get_node(pool_id);
                if (pn && pn->get_type() == NodeType::LINEAR) {
                    auto* lp = static_cast<NeuronNode*>(pn);
                    lp->set_input_count(patch_pixels);
                    for (int i = 0; i < patch_pixels; ++i) {
                        lp->set_weight(i, 1.0 / patch_pixels);  // exact block mean
                    }
                    lp->set_bias(0.0);
                }
                // Wire the patch's pixels for channel c.
                // Grayscale: pixel (y,x) at index y*side + x.
                // Interleaved RGB: at index (y*side + x)*3 + c.
                for (int dy = 0; dy < patch; ++dy) {
                    for (int dx = 0; dx < patch; ++dx) {
                        int y = by * patch + dy;
                        int x = bx * patch + dx;
                        size_t idx = (channels == 1)
                                   ? static_cast<size_t>(y * side + x)
                                   : static_cast<size_t>((y * side + x) * channels + c);
                        uint64_t pix = input_ids[idx];
                        shadow->add_connection(pix, 0, pool_id,
                                               static_cast<size_t>(dy * patch + dx));
                    }
                }
                // Pooled feature 鈫?failing node's reserved port
                shadow->add_connection(pool_id, 0, failing_id,
                                       first_new_port + static_cast<size_t>(pool_idx));
                ++pool_idx;
            }
        }
        }
        break;
    }

    case Hypothesis::RECURRENT_XOR: {
        // Create a recurrent XOR node for running parity:
        //   output[t] = input[t] XOR output[t-1]
        //
        // Architecture:
        //   INPUT 鈹€鈹€鈫?XOR 鈫愨攢鈹€ (recurrent self-loop: own output[0])
        //              鈫?        //           downstream (OUTPUT)
        //
        // The XOR node uses truthiness (!= 0.0) semantics. For binary
        // inputs (0/1), this correctly computes running parity. The
        // recurrent delay_buffer starts at 0, so output[0] = XOR(input[0], 0)
        // = input[0] (parity of first bit = first bit itself).

        // Find the INPUT node that feeds the failing node
        uint64_t input_src = 0;
        for (const auto& c : shadow->get_connections()) {
            if (c.dst_node == failing_id) {
                const Node* src = shadow->get_node(c.src_node);
                if (src && src->get_type() == NodeType::INPUT) {
                    input_src = c.src_node;
                    break;
                }
            }
        }
        if (input_src == 0) break;

        // Collect outgoing connections from failing_node
        std::vector<Connection> outgoing;
        for (const auto& c : shadow->get_connections()) {
            if (c.src_node == failing_id) outgoing.push_back(c);
        }
        for (const auto& c : outgoing) {
            shadow->remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);
        }

        // Create XOR node
        uint64_t xor_id = shadow->add_node(NodeType::XOR, "recurrent_xor");

        // Wire INPUT 鈫?XOR(port 0)
        shadow->add_connection(input_src, 0, xor_id, 0);
        // Wire XOR(self) recurrent 鈫?XOR(port 1)
        shadow->add_connection(xor_id, 0, xor_id, 1);

        // Route XOR output directly to OUTPUT node(s), replacing all
        // intermediate processing (SIN/TANH chains etc.) that may have
        // been added by previous structural commits.
        for (auto& n : shadow->get_nodes()) {
            if (n->get_type() == NodeType::OUTPUT) {
                // Remove existing input to this OUTPUT
                std::vector<Connection> out_conns;
                for (const auto& c : shadow->get_connections()) {
                    if (c.dst_node == n->get_id()) out_conns.push_back(c);
                }
                for (const auto& c : out_conns) {
                    shadow->remove_connection(c.src_node, c.src_port,
                                             c.dst_node, c.dst_port);
                }
                // Connect XOR 鈫?OUTPUT
                shadow->add_connection(xor_id, 0, n->get_id(), 0);
                // Reset OUTPUT scale/bias to pass-through
                auto* on = static_cast<OutputNode*>(n.get());
                on->set_scale(1.0);
                on->set_bias(0.0);
            }
        }
        break;
    }

    case Hypothesis::PARITY_TREE: {
        // Linear-fold XOR tree over all INPUT nodes:
        //   root = XOR(x0, XOR(x1, XOR(x2, ... )))
        // Exact k-bit parity in one structural change. XOR uses (in>0)
        // truthiness, so raw 0/1 inputs give 0/1 outputs. Root wires to
        // every OUTPUT (replacing existing inputs, scale/bias reset) 鈥?        // SGD then grows scale/bias for confident sigmoid outputs.

        std::vector<uint64_t> input_ids;
        for (const auto& n : shadow->get_nodes()) {
            if (n->get_type() == NodeType::INPUT) input_ids.push_back(n->get_id());
        }
        if (input_ids.size() < 2) break;

        // Linear fold: acc = XOR(acc, x_i)
        uint64_t acc = input_ids[0];
        for (size_t i = 1; i < input_ids.size(); ++i) {
            uint64_t xid = shadow->add_node(NodeType::XOR, "parity_fold");
            shadow->add_connection(acc, 0, xid, 0);
            shadow->add_connection(input_ids[i], 0, xid, 1);
            acc = xid;
        }
        // Rename final fold to parity_root for the dedup check
        {
            Node* root = shadow->get_node(acc);
            if (root) root->set_name("parity_root");
        }

        // Root 鈫?every OUTPUT (replace inputs; pass-through scale/bias)
        for (auto& n : shadow->get_nodes()) {
            if (n->get_type() == NodeType::OUTPUT) {
                std::vector<Connection> out_conns;
                for (const auto& c : shadow->get_connections()) {
                    if (c.dst_node == n->get_id()) out_conns.push_back(c);
                }
                for (const auto& c : out_conns) {
                    shadow->remove_connection(c.src_node, c.src_port,
                                              c.dst_node, c.dst_port);
                }
                shadow->add_connection(acc, 0, n->get_id(), 0);
                auto* on = static_cast<OutputNode*>(n.get());
                on->set_scale(4.0);   // sigmoid(卤4) 鈮?0.982 鈥?confident start
                on->set_bias(0.0);
            }
        }
        break;
    }

    case Hypothesis::NONE:
    default:
        return nullptr;
    }

    return shadow;
}

// ============================================================================
// Phase 6: Graph compilation helpers
// ============================================================================

// is_purely_constant 鈥?check if a node and all its transitive inputs are CONSTANT
static bool is_purely_constant(const Graph& g, uint64_t node_id,
                               std::unordered_set<uint64_t>& visited) {
    if (!visited.insert(node_id).second) return true;  // already confirmed

    const Node* n = g.get_node(node_id);
    if (!n) return false;
    if (n->get_type() == NodeType::CONSTANT) return true;

    // Check if all upstream sources are constant
    const auto& conns = g.get_connections();
    bool all_upstream_const = true;
    bool has_input = false;
    for (const auto& c : conns) {
        if (c.dst_node == node_id) {
            has_input = true;
            if (!is_purely_constant(g, c.src_node, visited)) {
                all_upstream_const = false;
                break;
            }
        }
    }
    return has_input && all_upstream_const;
}

int graph_eliminate_dead_code(Graph& graph) {
    auto reachable = graph.compute_reachable_ids();
    std::unordered_set<uint64_t> reachable_set(reachable.begin(), reachable.end());

    // Also include INPUT and CONSTANT nodes (they are "sources")
    const auto& nodes = graph.get_nodes();
    std::vector<uint64_t> to_remove;
    for (const auto& n : nodes) {
        if (reachable_set.count(n->get_id()) == 0) {
            // Don't remove INPUT nodes (they're data sources)
            if (n->get_type() != NodeType::INPUT) {
                to_remove.push_back(n->get_id());
            }
        }
    }

    for (auto nid : to_remove) {
        graph.remove_node(nid);
    }

    return (int)to_remove.size();
}

int graph_fold_constants(Graph& graph) {
    int folded = 0;
    bool changed = true;

    // Iterate until no more folds possible
    while (changed) {
        changed = false;

        // Execute the graph once to populate all outputs
        // Set all INPUT/needed nodes to 0 for execution 鈥?we only care about CONSTANTS
        // Actually, we need the graph to have SOME state. Let's just execute.
        // But executing requires INPUT values. We'll use a different approach 鈥?just check
        // if all inputs settle at compile time.

        const auto& nodes = graph.get_nodes();
        const auto& conns = graph.get_connections();

        std::unordered_set<uint64_t> visited_const;

        for (const auto& n : nodes) {
            uint64_t nid = n->get_id();
            NodeType nt = n->get_type();

            // Skip IO and special nodes
            if (nt == NodeType::INPUT || nt == NodeType::OUTPUT ||
                nt == NodeType::CONSTANT || nt == NodeType::SINK ||
                nt == NodeType::IFELSE || nt == NodeType::ABSENT) continue;

            // Check if node is purely fed by CONSTANTS
            visited_const.clear();
            bool pure = is_purely_constant(graph, nid, visited_const);
            if (!pure) continue;

            // Execute the graph to get computed outputs
            // Set all INPUT nodes to 0 first
            for (const auto& nn : nodes) {
                if (nn->get_type() == NodeType::INPUT) {
                    graph.set_input_value(nn->get_id(), 0.0);
                }
            }
            graph.execute();

            // Get the node's output value and replace it with a CONSTANT
            Value out_val = graph.get_any_node_output(nid);

            // Collect all downstream connections
            std::vector<Connection> downstream;
            for (const auto& c : conns) {
                if (c.src_node == nid) downstream.push_back(c);
            }

            // Remove the node and its connections
            graph.remove_node(nid);

            // Add CONSTANT replacement
            uint64_t const_id = graph.add_node(NodeType::CONSTANT, "folded_c");
            Node* cnode = graph.get_node(const_id);
            if (cnode && cnode->get_type() == NodeType::CONSTANT) {
                static_cast<ConstantNode*>(cnode)->set_value(out_val);
            }

            // Re-wire downstream connections
            for (const auto& c : downstream) {
                graph.add_connection(const_id, 0, c.dst_node, c.dst_port);
            }

            folded++;
            changed = true;
            break;  // restart iteration since node indices changed
        }
    }

    return folded;
}

int graph_compress_neurons(Graph& graph) {
    int compressed = 0;
    bool changed = true;

    while (changed) {
        changed = false;

        const auto& conns = graph.get_connections();

        // Build a map from node 鈫?outgoing connections
        std::unordered_map<uint64_t, std::vector<Connection>> outgoing;
        std::unordered_map<uint64_t, std::vector<Connection>> incoming;

        for (const auto& c : conns) {
            outgoing[c.src_node].push_back(c);
            incoming[c.dst_node].push_back(c);
        }

        // Find pairs: NEURON/LINEAR A 鈫?NEURON/LINEAR B (same type only,
        // to avoid merging tanh with linear activation) where B has no other
        // inputs from B's port.
        for (const auto& [src_id, out_conns] : outgoing) {
            const Node* src_node = graph.get_node(src_id);
            if (!src_node) continue;
            NodeType src_t = src_node->get_type();
            if (src_t != NodeType::NEURON && src_t != NodeType::LINEAR) continue;

            for (const auto& c : out_conns) {
                uint64_t dst_id = c.dst_node;
                const Node* dst_node = graph.get_node(dst_id);
                if (!dst_node) continue;
                NodeType dst_t = dst_node->get_type();
                if (dst_t != src_t) continue;  // only merge same activation type

                // Check that destination NEURON has exactly one input at port c.dst_port
                bool single_input = true;
                for (const auto& ic : incoming[dst_id]) {
                    if (ic.dst_port != c.dst_port && ic.src_node != src_id) {
                        single_input = false;
                        break;
                    }
                }
                if (!single_input) continue;

                // Compress: merge src NEURON's weights into dst NEURON
                // src 鈫?dst: src_output = bias_src + sum(w_src_i * x_i)
                // dst = bias_dst + w_dst_0 * src_output + sum(w_dst_j * other_inputs)
                // Merged: dst = (bias_dst + w_dst_0 * bias_src) + sum(w_dst_0 * w_src_i * x_i) + sum(other)

                Value w_merge = dst_node->get_input(c.dst_port);
                (void)w_merge;
                // For this to work, we need the dst_neuron's weight at the port connecting from src
                // Actually, we need to look at the NEURON's internal weights, not the connection

                // The destination NEURON has weights[port] for each input.
                // We need to multiply src weights by dst's weight at the connecting port,
                // then add the biases.

                // But get_weight/get_num_weights are on NeuronNode specifically
                const NeuronNode* sn = static_cast<const NeuronNode*>(src_node);
                const NeuronNode* dn = static_cast<const NeuronNode*>(dst_node);

                size_t src_num_inputs = sn->get_num_inputs();
                size_t dst_num_inputs = dn->get_num_inputs();
                if (dst_num_inputs == 0) continue;

                Value dst_weight_on_src_port = (c.dst_port < dst_num_inputs) ? dn->get_weight(c.dst_port) : 0.0;

                // Remove connection src 鈫?dst
                graph.remove_connection(c.src_node, c.src_port, c.dst_node, c.dst_port);

                // For each input to src NEURON, add equivalent direct connections to dst
                // merged_weight = dst_weight_on_src_port * src_weight
                for (size_t i = 0; i < src_num_inputs; ++i) {
                    Value src_w = sn->get_weight(i);
                    (void)src_w;
                    // Find who feeds src NEURON at port i
                    for (const auto& inc : incoming[src_id]) {
                        if (inc.dst_port == i) {
                            // Add connection from inc.src_node 鈫?dst_node
                            graph.add_connection(inc.src_node, inc.src_port, dst_id, dst_num_inputs + i);
                            break;
                        }
                    }
                }

                // Update dst bias: dst_bias += dst_weight * src_bias
                // and set new weights for the added inputs
                Node* dst_mut = graph.get_node(dst_id);
                if (dst_mut && (dst_mut->get_type() == NodeType::NEURON
                                || dst_mut->get_type() == NodeType::LINEAR)) {
                    NeuronNode* dnn = static_cast<NeuronNode*>(dst_mut);
                    Value new_bias = dnn->get_bias() + dst_weight_on_src_port * sn->get_bias();
                    dnn->set_bias(new_bias);

                    // Set merged weights for new inputs
                    for (size_t i = 0; i < src_num_inputs; ++i) {
                        dnn->set_weight(dst_num_inputs + i, dst_weight_on_src_port * sn->get_weight(i));
                    }
                }

                // Remove src NEURON
                graph.remove_node(src_id);

                compressed++;
                changed = true;
                goto restart_compress;
            }
        }
        restart_compress:;
    }

    return compressed;
}

// ============================================================================
// compute_validation_loss 鈥?evaluate any graph on validation_data_ (read-only)
// ============================================================================
double EvolutionEngine::sample_softmax_ce(
    const std::unordered_map<uint64_t, Value>& targets) const {
    // Gather (graph output value, target) pairs for all mapped outputs.
    std::vector<std::pair<Value, Value>> pairs;   // (logit, target)
    for (const auto& kv : targets) {
        auto mi = output_data_to_graph_.find(kv.first);
        if (mi == output_data_to_graph_.end()) continue;
        const Node* n = graph_->get_node(mi->second);
        if (!n || n->get_type() != NodeType::OUTPUT) continue;
        pairs.emplace_back(graph_->get_output_value(mi->second), kv.second);
    }
    if (pairs.size() < 2) return 0.0;
    double mx = -1e300;
    for (auto& p : pairs) mx = std::max(mx, static_cast<double>(p.first));
    double denom = 0.0;
    for (auto& p : pairs) denom += std::exp(static_cast<double>(p.first) - mx);
    double lse = mx + std::log(denom);
    for (auto& p : pairs) {
        if (p.second > 0.5) return lse - static_cast<double>(p.first);
    }
    return 0.0;
}

double EvolutionEngine::compute_validation_loss(Graph& g) {
    if (validation_data_.samples.empty()) return 0.0;

    // NOTE: do NOT reset recurrent state here. For sequence tasks (d6),
    // the validation targets are cumulative from the start of the full
    // sequence. The caller (validate_shadow_only) runs the sanity check
    // on all training samples first, which leaves the recurrent state
    // at the correct position for the start of validation. Resetting
    // would break running-parity / running-sum tasks.
    //
    // PERF: strided subsample on large validation sets (>= 2x the cap).
    // Shadow validation calls this per candidate per cycle; on 48k-sample
    // tasks the val split is ~12k, and sequential evaluation of 12k
    // executes x 8 candidates was a measurable share of cycle time.
    // Sequence mode keeps full data (temporal continuity).
    const size_t v_total = validation_data_.samples.size();
    const size_t v_cap = static_cast<size_t>(config::SHADOW_TRAIN_MAX_SAMPLES);
    const size_t v_stride = (!cfg_.sequence_mode && v_total > v_cap) ? (v_total / v_cap) : 1;

    double total_loss = 0.0;
    int count = 0;
    for (size_t v_i = 0; v_i < v_total; v_i += v_stride) {
        const auto& sample = validation_data_.samples[v_i];
        for (const auto& kv : sample.inputs) {
            auto map_it = input_data_to_graph_.find(kv.first);
            if (map_it != input_data_to_graph_.end()) {
                g.set_input_value(map_it->second, kv.second);
            }
        }
        g.execute();
        if (cfg_.loss_type == Graph::LossType::SOFTMAX_CE) {
            // Joint softmax-CE on graph g (arbitrary, not graph_).
            double mx = -1e300;
            std::vector<std::pair<Value,Value>> prs;
            for (const auto& kv2 : sample.targets) {
                auto mi2 = output_data_to_graph_.find(kv2.first);
                if (mi2 == output_data_to_graph_.end()) continue;
                Value lg = g.get_output_value(mi2->second);
                prs.emplace_back(lg, kv2.second);
                mx = std::max(mx, static_cast<double>(lg));
            }
            if (prs.size() >= 2) {
                double den = 0.0;
                for (auto& pr : prs)
                    den += std::exp(static_cast<double>(pr.first) - mx);
                double lse = mx + std::log(den);
                for (auto& pr : prs) {
                    if (pr.second > 0.5) {
                        total_loss += lse - static_cast<double>(pr.first);
                        break;
                    }
                }
            }
        } else
        for (const auto& kv : sample.targets) {
            auto map_it = output_data_to_graph_.find(kv.first);
            if (map_it == output_data_to_graph_.end()) continue;
            Value pred   = g.get_output_value(map_it->second);
            Value target = kv.second;
            Value diff   = pred - target;
            if (cfg_.loss_type == Graph::LossType::BCE) {
                Value sig_pred = 1.0 / (1.0 + std::exp(-pred));
                constexpr Value eps = config::BCE_LOG_CLAMP_EPSILON;
                Value clamped = std::max(eps, std::min(1.0 - eps, sig_pred));
                total_loss += -(target * std::log(clamped) + (1.0 - target) * std::log(1.0 - clamped));
            } else {
                total_loss += diff * diff;
            }
        }
        count++;
    }
    return count > 0 ? total_loss / count : 0.0;
}

// ============================================================================
// evaluate_external 鈥?held-out test-set evaluation (per-output MSE / R虏 / MAE)
// ============================================================================
std::vector<EvolutionEngine::ExternalEval> EvolutionEngine::evaluate_external(
    const Dataset& data) {
    std::vector<ExternalEval> result;

    // Collect output data keys in sorted order (stable per-output reporting)
    std::vector<uint64_t> out_keys;
    if (!data.samples.empty()) {
        for (const auto& kv : data.samples[0].targets) out_keys.push_back(kv.first);
        std::sort(out_keys.begin(), out_keys.end());
    }

    // Accumulators per output
    size_t K = out_keys.size();
    std::vector<double> sse(K, 0.0), sae(K, 0.0), sum_t(K, 0.0), sum_t2(K, 0.0);
    std::vector<size_t> n(K, 0);

    graph_->reset_recurrent_state();
    for (const auto& sample : data.samples) {
        for (const auto& kv : sample.inputs) {
            auto map_it = input_data_to_graph_.find(kv.first);
            if (map_it != input_data_to_graph_.end()) {
                graph_->set_input_value(map_it->second, kv.second);
            }
        }
        graph_->execute();
        for (size_t j = 0; j < K; ++j) {
            auto map_it = output_data_to_graph_.find(out_keys[j]);
            if (map_it == output_data_to_graph_.end()) continue;
            auto tit = sample.targets.find(out_keys[j]);
            if (tit == sample.targets.end()) continue;
            Value pred = graph_->get_output_value(map_it->second);
            Value diff = pred - tit->second;
            sse[j] += diff * diff;
            sae[j] += std::abs(diff);
            sum_t[j]  += tit->second;
            sum_t2[j] += tit->second * tit->second;
            n[j]++;
        }
    }

    for (size_t j = 0; j < K; ++j) {
        ExternalEval e;
        if (n[j] == 0) { e.mse = e.r2 = e.mae = 0.0; result.push_back(e); continue; }
        e.mse = sse[j] / n[j];
        e.mae = sae[j] / n[j];
        double mean_t = sum_t[j] / n[j];
        double var_t  = sum_t2[j] / n[j] - mean_t * mean_t;
        e.r2 = (var_t > 1e-12) ? 1.0 - e.mse / var_t : 0.0;
        result.push_back(e);
    }
    return result;
}

// ============================================================================
// evaluate_external_softmax 鈥?comparable one-hot LM/classification metric
// ============================================================================
std::vector<double> EvolutionEngine::evaluate_external_softmax(const Dataset& data) {
    if (data.samples.empty()) return {};

    // Sorted output data keys (stable class order)
    std::vector<uint64_t> out_keys;
    for (const auto& kv : data.samples[0].targets) out_keys.push_back(kv.first);
    std::sort(out_keys.begin(), out_keys.end());
    size_t K = out_keys.size();
    if (K < 2) return {};

    std::vector<uint64_t> graph_out(K, 0);
    for (size_t j = 0; j < K; ++j) {
        auto it = output_data_to_graph_.find(out_keys[j]);
        if (it == output_data_to_graph_.end()) return {};
        graph_out[j] = it->second;
    }

    double ce_sum = 0.0;
    size_t correct = 0, count = 0;
    graph_->reset_recurrent_state();
    for (const auto& sample : data.samples) {
        for (const auto& kv : sample.inputs) {
            auto mi = input_data_to_graph_.find(kv.first);
            if (mi != input_data_to_graph_.end())
                graph_->set_input_value(mi->second, kv.second);
        }
        graph_->execute();

        std::vector<Value> logits(K);
        for (size_t j = 0; j < K; ++j)
            logits[j] = graph_->get_output_value(graph_out[j]);
        uint64_t truth = 0;
        for (size_t j = 0; j < K; ++j) {
            auto tit = sample.targets.find(out_keys[j]);
            if (tit != sample.targets.end() && tit->second > 0.5) truth = j;
        }

        double mx = *std::max_element(logits.begin(), logits.end());
        double denom = 0.0, best_val = -1e18;
        size_t best = 0;
        for (size_t j = 0; j < K; ++j) {
            denom += std::exp(logits[j] - mx);
            if (logits[j] > best_val) { best_val = logits[j]; best = j; }
        }
        ce_sum += -(logits[truth] - mx - std::log(denom));
        if (best == truth) ++correct;
        ++count;
    }
    if (count == 0) return {};
    return { ce_sum / count, static_cast<double>(correct) / count };
}

// ============================================================================
// validate_shadow_only 鈥?full validation pipeline WITHOUT mutating graph_
// ============================================================================
// Performs: compile 鈫?train 鈫?sanity check 鈫?validation loss 鈫?commit gate.
// Returns a ShadowValidationResult; caller inspects `acceptable` and decides
// whether to commit. baseline_val is the current graph's validation loss
// (precomputed by caller to avoid N redundant recomputations when validating
// N candidates in parallel).
// ============================================================================
EvolutionEngine::ShadowValidationResult EvolutionEngine::validate_shadow_only(
    std::unique_ptr<Graph>& shadow_graph,
    double baseline_loss,
    double baseline_val,
    int hyp_rank,
    int hyp_type,
    bool structural_evidence) {

    ShadowValidationResult result;
    result.hyp_rank = hyp_rank;
    result.hyp_type = hyp_type;

    if (!shadow_graph || validation_data_.samples.empty()) {
        result.reject_reason = "empty shadow or validation data";
        return result;
    }

    // Verify shadow graph has at least one OUTPUT reachable from inputs
    bool has_output = false;
    for (const auto& node : shadow_graph->get_nodes()) {
        if (node->get_type() == NodeType::OUTPUT) { has_output = true; break; }
    }
    if (!has_output) {
        result.reject_reason = "no OUTPUT node";
        return result;
    }

    // Phase 6 (shadow): Compile the shadow graph before validation.
    // M5.7 diagnostic: dump evidence-candidate shadows (pre- and
    // post-compile) for the routing audit — file per candidate rank.
    if (structural_evidence) {
        try {
            save_graph_to_file(*shadow_graph,
                "evidence_shadow_pre_" + std::to_string(hyp_rank) + ".json");
        } catch (...) {}
    }
    graph_eliminate_dead_code(*shadow_graph);
    if (structural_evidence) {
        try {
            save_graph_to_file(*shadow_graph,
                "evidence_shadow_post_" + std::to_string(hyp_rank) + ".json");
        } catch (...) {}
    }
    graph_fold_constants(*shadow_graph);
    graph_compress_neurons(*shadow_graph);

    // Verify shadow still has an OUTPUT after compilation
    has_output = false;
    for (const auto& node : shadow_graph->get_nodes()) {
        if (node->get_type() == NodeType::OUTPUT) { has_output = true; break; }
    }
    if (!has_output) {
        result.reject_reason = "OUTPUT removed by compile";
        return result;
    }

    // Phase 6.5: Train shadow graph on training data so new NEURON weights are fitted.
    if (!training_data_.samples.empty()) {
        // M5.7 diagnostic: pre-training train loss for evidence
        // candidates. Identity-start structures should sit at ~baseline
        // BEFORE SGD; a high pre-train loss means the ROUTING itself
        // breaks the forward pass (vs SGD divergence after).
        if (structural_evidence) {
            double pre_loss = 0.0;
            int pre_n = 0;
            const size_t pt_total = training_data_.samples.size();
            const size_t pt_stride = (pt_total > 200) ? (pt_total / 200) : 1;
            for (size_t pi = 0; pi < pt_total; pi += pt_stride) {
                for (const auto& kv : training_data_.samples[pi].inputs) {
                    auto git = input_data_to_graph_.find(kv.first);
                    if (git != input_data_to_graph_.end()) {
                        shadow_graph->set_input_value(git->second, kv.second);
                    }
                }
                shadow_graph->execute();
                for (const auto& kv : training_data_.samples[pi].targets) {
                    auto oit = output_data_to_graph_.find(kv.first);
                    if (oit != output_data_to_graph_.end()) {
                        Value diff = shadow_graph->get_output_value(oit->second) - kv.second;
                        pre_loss += diff * diff;
                        ++pre_n;
                    }
                }
                shadow_graph->reset_recurrent_state();
            }
            Logger::info("  [EVIDENCE-PRE] pre-train loss="
                        + std::to_string(pre_n > 0 ? pre_loss / pre_n : -1.0));
        }
        Graph::TrainConfig train_cfg;
        train_cfg.epochs = cfg_.sgd_epochs_per_phase;
        // MULTI_LAYER_STACK trains K+1 neurons from near-scratch 鈥?needs
        // the most budget (6x vs 3x for smaller compounds).
        // PATCH_POOLING adds n_blocks虏 pool neurons + zero-init ports 鈥?        // also needs the large budget.
        if (hyp_type == static_cast<int>(Hypothesis::MULTI_LAYER_STACK)
            || hyp_type == static_cast<int>(Hypothesis::PATCH_POOLING)) {
            train_cfg.epochs *= config::MULTI_LAYER_STACK_SGD_MULTIPLIER;
        }
        // Compound hypotheses add 4 new nodes (MULTIPLY鈫扤EURON鈫扵ANH鈫扐DD)
        // with random weights. The default budget isn't enough to train
        // them from scratch to beat a baseline that's been training for
        // hundreds of epochs. Give them extra room to converge.
        if (hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY_NEURON)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_TANH_SERIES)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY3_NEURON)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY_ABS)
            || hyp_type == static_cast<int>(Hypothesis::RECURRENT_SELF_WIRE)
            || hyp_type == static_cast<int>(Hypothesis::RECURRENT_MULTI_TAP)
            || hyp_type == static_cast<int>(Hypothesis::DELAY_LINE)
            || hyp_type == static_cast<int>(Hypothesis::SIN_INJECTION)
            || hyp_type == static_cast<int>(Hypothesis::DEEP_INSERTION)
            || hyp_type == static_cast<int>(Hypothesis::MULTI_LAYER_STACK)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_SIN_PRODUCT)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_DIVIDE_PRODUCT)
            || hyp_type == static_cast<int>(Hypothesis::IFELSE_PRESERVE)
            || hyp_type == static_cast<int>(Hypothesis::EMBED_TRUNK)) {
            // PRESERVE gets 2x the compound budget: each chain branch only
            // receives gradient from its side of the split (1/K of samples),
            // so retraining the separated copies needs K-times the epochs.
            train_cfg.epochs *= config::SHADOW_COMPOUND_SGD_MULTIPLIER * 2;
        }
        // M5.7: measured-evidence boundary candidates get an extended
        // budget — masking re-routes half the samples through fresh
        // structure; the payoff materializes only after retraining, which
        // the default budget under-prices (t22 boundary candidates lost
        // every validation race to immediate micro-gains).
        if (structural_evidence
            && hyp_type == static_cast<int>(Hypothesis::IFELSE_BOUNDARY_SPLIT)) {
            train_cfg.epochs *= config::SHADOW_BOUNDARY_SGD_MULTIPLIER;
        }
        // M2.3: the attention head's table starts random — its retrieval
        // value only shows after the embeddings/readout train TOGETHER.
        // Both scalar (v1) and V-port (v2) heads measured losing every
        // validation cycle to micro-gain EMBED/DEEP commits at default
        // budget (induction probes: val flat at 3.61-3.62). Extended
        // budget so the head's value materializes within validation —
        // same class of fix as the boundary shadows.
        if (hyp_type == static_cast<int>(Hypothesis::ATTENTION_MIX)) {
            train_cfg.epochs *= config::SHADOW_BOUNDARY_SGD_MULTIPLIER;
        }
        train_cfg.learning_rate = cfg_.sgd_learning_rate;
        // Compound shadow: reduce LR so zero-init chains grow without
        // disrupting the well-trained existing graph. Without this, the
        // large residual gradients flowing through the new TANH chains
        // destabilise existing weights and cause catastrophic regression.
        // EXEMPT: COMPOUND_DIVIDE_PRODUCT (gain-init puts it at the right
        // scale immediately — the 0.2x LR under-trains the exact-feature
        // variants during shadow validation, costing I.32.8 its races).
        if (hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY_NEURON)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_TANH_SERIES)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY3_NEURON)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY_ABS)
            || hyp_type == static_cast<int>(Hypothesis::RECURRENT_SELF_WIRE)
            || hyp_type == static_cast<int>(Hypothesis::RECURRENT_MULTI_TAP)
            || hyp_type == static_cast<int>(Hypothesis::DELAY_LINE)
            || hyp_type == static_cast<int>(Hypothesis::DEEP_INSERTION)
            || hyp_type == static_cast<int>(Hypothesis::MULTI_LAYER_STACK)
            || hyp_type == static_cast<int>(Hypothesis::PATCH_POOLING)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_SIN_PRODUCT)
            || hyp_type == static_cast<int>(Hypothesis::EMBED_TRUNK)) {
            train_cfg.learning_rate *= config::SHADOW_COMPOUND_LR_MULTIPLIER;
        }
        // M5.7: evidence-boundary shadows train longer (3x) — without the
        // compound LR reduction the fresh split gates diverge during the
        // extended budget and the drift guard rejects a candidate whose
        // val is already perfect (measured on t22: val=0.0, shadow_train
        // 0.198 vs baseline 0.009 — pure LR blowup, not structure).
        // The structure is MEASURED — only the gates need learning, so a
        // deep LR cut protects the converged weights while gates adapt.
        // Momentum=0: the split boundary flips per-sample gradients;
        // momentum accumulates opposite updates across the discontinuity
        // and oscillates the gates (PRESERVE chains diverged at 0.27
        // train even at 0.1x LR with momentum on).
        if (structural_evidence) {
            train_cfg.learning_rate *= 0.1;
            train_cfg.momentum = 0.0;
        }
        train_cfg.gradient_clip = cfg_.sgd_gradient_clip;
        train_cfg.momentum = cfg_.sgd_momentum;
        train_cfg.weight_decay = cfg_.sgd_weight_decay;
        train_cfg.loss_type = cfg_.loss_type;
        train_cfg.batch_size = (cfg_.sequence_mode || training_data_.samples.size() < 500)
                               ? 0 : config::DEFAULT_SGD_BATCH_SIZE;
        train_cfg.early_stop_patience = 0;  // always run full SGD budget for shadow validation
        // Wall-clock watchdog: shadows must not live-lock (pooled-CIFAR
        // shadow spun 40+ min). Budget scales with dataset size and the
        // heavy-hypothesis multiplier: ~0.15s/sample/50-epochs for full-
        // batch small sets, bounded 2-15 min.
        {
            double per_epoch_sec = 0.003 * static_cast<double>(training_data_.samples.size());
            int budget_sec = static_cast<int>(per_epoch_sec * train_cfg.epochs * 0.6);
            if (budget_sec < 120)  budget_sec = 120;
            if (budget_sec > 900)  budget_sec = 900;
            train_cfg.watchdog_seconds = budget_sec;
        }
        train_cfg.input_data_to_graph = input_data_to_graph_;
        train_cfg.output_data_to_graph = output_data_to_graph_;
        // PERF: shadow validation trains on a CAPPED SUBSAMPLE. Shadows only
        // need to RANK candidates — the ranking is stable at ~8k samples
        // (relative val differences >> subsample noise), while training all
        // ~8 shadows on the full 48k set cost ~16 min per structural cycle
        // (measured on charLM w8). The WINNER gets full-data SGD after
        // commit. Sequence mode keeps full data (temporal order matters).
        if (cfg_.sequence_mode
            || training_data_.samples.size() <= static_cast<size_t>(config::SHADOW_TRAIN_MAX_SAMPLES)) {
            shadow_graph->train(training_data_.samples, train_cfg);
        } else {
            std::vector<Graph::SampleIODesc> sub;
            size_t n_total = training_data_.samples.size();
            size_t cap = static_cast<size_t>(config::SHADOW_TRAIN_MAX_SAMPLES);
            size_t stride = n_total / cap;
            sub.reserve(n_total / stride + 1);
            for (size_t i = 0; i < n_total; i += stride) {
                sub.push_back(training_data_.samples[i]);
            }
            shadow_graph->train(sub, train_cfg);
        }
    }

    // Sanity check: shadow's loss on the training data (the data it was just
    // trained on). Two guards: catastrophic-damage (1.5x) and drift
    // (max of abs/rel tolerance).
    if (!training_data_.samples.empty() && baseline_loss > 0.0) {
        shadow_graph->reset_recurrent_state();
        double shadow_train_loss = 0.0;
        int train_count = 0;
        // PERF: strided subsample for the sanity gate — it's a diagnostic
        // (catastrophic-regression check), not a training signal; 200 strided
        // samples estimate the loss to well within the gate's 1.5x margin.
        // Was ALL samples sequentially per candidate (up to ~15 candidates
        // per structural cycle).
        const size_t st_total = training_data_.samples.size();
        const size_t st_cap = static_cast<size_t>(config::COMPUTE_TARGETS_MAX_SAMPLES);
        const size_t st_stride = (st_total > st_cap) ? (st_total / st_cap) : 1;
        for (size_t st_i = 0; st_i < st_total; st_i += st_stride) {
            const auto& sample = training_data_.samples[st_i];
            for (const auto& kv : sample.inputs) {
                auto map_it = input_data_to_graph_.find(kv.first);
                if (map_it != input_data_to_graph_.end()) {
                    shadow_graph->set_input_value(map_it->second, kv.second);
                }
            }
            shadow_graph->execute();
            if (cfg_.loss_type == Graph::LossType::SOFTMAX_CE) {
                double mx = -1e300;
                std::vector<std::pair<Value,Value>> prs;
                for (const auto& kv2 : sample.targets) {
                    auto mi2 = output_data_to_graph_.find(kv2.first);
                    if (mi2 == output_data_to_graph_.end()) continue;
                    Value lg = shadow_graph->get_output_value(mi2->second);
                    prs.emplace_back(lg, kv2.second);
                    mx = std::max(mx, static_cast<double>(lg));
                }
                if (prs.size() >= 2) {
                    double den = 0.0;
                    for (auto& pr : prs)
                        den += std::exp(static_cast<double>(pr.first) - mx);
                    double lse = mx + std::log(den);
                    for (auto& pr : prs) {
                        if (pr.second > 0.5) {
                            shadow_train_loss += lse - static_cast<double>(pr.first);
                            break;
                        }
                    }
                }
            } else
            for (const auto& kv : sample.targets) {
                auto map_it = output_data_to_graph_.find(kv.first);
                if (map_it == output_data_to_graph_.end()) continue;
                Value pred   = shadow_graph->get_output_value(map_it->second);
                Value target = kv.second;
                Value diff   = pred - target;
                if (cfg_.loss_type == Graph::LossType::BCE) {
                    Value sig_pred = 1.0 / (1.0 + std::exp(-pred));
                    constexpr Value eps = config::BCE_LOG_CLAMP_EPSILON;
                    Value clamped = std::max(eps, std::min(1.0 - eps, sig_pred));
                    shadow_train_loss += -(target * std::log(clamped) + (1.0 - target) * std::log(1.0 - clamped));
                } else {
                    shadow_train_loss += diff * diff;
                }
            }
            train_count++;
        }
        result.train_loss = train_count > 0 ? shadow_train_loss / train_count : 0.0;

        if (result.train_loss > baseline_loss * config::SHADOW_TRAIN_REGRESSION_FACTOR) {
            result.reject_reason = "training loss regressed catastrophically (shadow_train="
                                 + std::to_string(result.train_loss)
                                 + " vs baseline_loss=" + std::to_string(baseline_loss) + ")";
            return result;
        }
        double max_allowed_regression = std::max(config::SHADOW_TRAIN_MAX_REGRESSION_ABS,
                                                 baseline_loss * config::SHADOW_TRAIN_MAX_REGRESSION_REL);
        if (result.train_loss > baseline_loss + max_allowed_regression) {
            result.reject_reason = "training loss drifted up (shadow_train="
                                 + std::to_string(result.train_loss)
                                 + " vs baseline_loss=" + std::to_string(baseline_loss)
                                 + ", delta=" + std::to_string(result.train_loss - baseline_loss)
                                 + ", max_allowed=" + std::to_string(max_allowed_regression) + ")";
            return result;
        }
    }

    // Compute shadow's loss on validation set (sequential 鈥?validation set is
    // small, typically ~40 samples; no need for nested threading which would
    // oversubscribe when multiple candidates run in parallel).
    // Compute shadow's loss on validation set (sequential; strided-subsampled
    // inside compute_validation_loss for large sets — ranking-stable, and
    // large-dataset val sets are proportional so this matters).
    result.val_loss = compute_validation_loss(*shadow_graph);

    // Commit gate: two-sided relative scaling. Required improvement =
    // max(epsilon, min(validation_threshold, 1% 脳 baseline)).
    //   Large baselines: 1% branch dominates (noise protection, e.g. CIFAR
    //     at 1.9 where a 1e-3 gain is noise).
    //   Small baselines: the min() caps the requirement BELOW the absolute
    //     threshold, so it shrinks with the baseline 鈥?near-converged tasks
    //     (loss < threshold) can still accept real improvements (observed:
    //     DIVIDE improving val 3.1e-5 鈫?2.9e-5 was blocked by a flat 1e-3
    //     floor on Feynman ratio equations).
    double rel = baseline_val * config::COMMIT_MIN_IMPROVEMENT_FRACTION;
    double required_improvement = std::max(config::COMMIT_MIN_IMPROVEMENT,
                                           std::min(cfg_.validation_threshold, rel));

    if (result.val_loss < baseline_val - required_improvement) {
        result.acceptable = true;
    } else if (structural_evidence
               && result.val_loss <= baseline_val * 1.001) {
        // M5.7: MEASURED structure (label-space plateau edge) commits on
        // any improvement. The threshold is evidence, not a statistical
        // fit — the standard gate prices it like an ordinary hypothesis
        // and rejects the exact boundary while accepting micro-gain
        // alternatives (t22: boundary candidate rejected at val 0.0086 vs
        // 0.0087 while a TANH commit won with less). Identity-start split
        // chains can sit exactly AT baseline before per-side training —
        // the 0.1% tolerance accepts neutral, deferring payoff to
        // post-commit SGD.
        result.acceptable = true;
    } else if ((hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY_NEURON)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_TANH_SERIES)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY3_NEURON)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY_ABS)
                || hyp_type == static_cast<int>(Hypothesis::RECURRENT_SELF_WIRE)
            || hyp_type == static_cast<int>(Hypothesis::RECURRENT_MULTI_TAP)
            || hyp_type == static_cast<int>(Hypothesis::DELAY_LINE)
                || hyp_type == static_cast<int>(Hypothesis::SIN_INJECTION)
                || hyp_type == static_cast<int>(Hypothesis::DEEP_INSERTION)
                || hyp_type == static_cast<int>(Hypothesis::MULTI_LAYER_STACK)
                || hyp_type == static_cast<int>(Hypothesis::PATCH_POOLING)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_SIN_PRODUCT)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_DIVIDE_PRODUCT)
                || hyp_type == static_cast<int>(Hypothesis::IFELSE_PRESERVE)
                || hyp_type == static_cast<int>(Hypothesis::EMBED_TRUNK))
               && baseline_val > 1e-9
               && result.val_loss <= baseline_val * (1.0 + config::SHADOW_COMPOUND_VAL_TOLERANCE)) {
        // Compound tolerance: the MULTIPLY鈫扤EURON鈫扵ANH architecture adds 4
        // fresh nodes with random weights. Even with 3x SGD budget, they
        // can't BEAT a baseline that's been training for hundreds of epochs
        // in a single validation pass. But the profile already confirmed
        // the architecture matches the residual signature (cross-term +
        // bounded). Accept within 2% of baseline 鈥?future SGD cycles will
        // refine the new chain. Without this, sin(xy)-class problems are
        // permanently stuck at the first NEURON_TANH plateau.
        result.acceptable = true;
        result.reject_reason = "accepted within compound tolerance (val=" + std::to_string(result.val_loss)
                             + " vs baseline=" + std::to_string(baseline_val) + ")";
    } else {
        result.reject_reason = "val_loss=" + std::to_string(result.val_loss)
                             + " did not beat baseline_val=" + std::to_string(baseline_val)
                             + " by required=" + std::to_string(required_improvement);
    }

    return result;
}

// ============================================================================
// Phase 7: validate_and_commit 鈥?single-candidate convenience wrapper
// ============================================================================
// Validates one shadow and commits if acceptable. Kept for backward compat;
// the parallel candidate loop in evolve() calls validate_shadow_only directly
// and commits the winner.
// ============================================================================
bool EvolutionEngine::validate_and_commit(std::unique_ptr<Graph>& shadow_graph,
                                          double baseline_loss,
                                          const std::string& /*hyp_description*/) {
    if (!shadow_graph || validation_data_.samples.empty()) return false;

    // Compute baseline_val on demand (single-candidate path doesn't precompute)
    double baseline_val = compute_validation_loss(*graph_);

    ShadowValidationResult result = validate_shadow_only(shadow_graph,
                                                         baseline_loss,
                                                         baseline_val,
                                                         /*hyp_rank=*/0,
                                                         /*hyp_type=*/0);

    if (result.acceptable) {
        graph_ = std::move(shadow_graph);
        Logger::verbose("  validate: COMMIT (val=" + std::to_string(result.val_loss)
                       + " train=" + std::to_string(result.train_loss) + ")");
        return true;
    }

    Logger::verbose("  validate: REJECT 鈥?" + result.reject_reason);
    return false;
}

// ============================================================================
// compile 鈥?orchestrator for all three Phase 6 passes
// ============================================================================
EvolutionEngine::CompileResult EvolutionEngine::compile() {
    CompileResult result{};

    result.dead_nodes_removed = graph_eliminate_dead_code(*graph_);
    result.constants_folded   = graph_fold_constants(*graph_);
    result.neurons_compressed = graph_compress_neurons(*graph_);

    Logger::verbose("compile: dead=" + std::to_string(result.dead_nodes_removed)
                   + ", folded=" + std::to_string(result.constants_folded)
                   + ", compressed=" + std::to_string(result.neurons_compressed));

    return result;
}

} // namespace aria
