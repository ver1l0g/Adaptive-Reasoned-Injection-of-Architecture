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
    NodeType starter_type = (cfg_.loss_type == Graph::LossType::BCE)
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

    int num_threads = (config::EVOLUTION_PARALLEL && !graph_->has_recurrent_connections())
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

                    if (cfg_.loss_type == Graph::LossType::BCE) {
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
                const char* hyp_names[] = {"NONE", "IFELSE_BOUNDARY_SPLIT", "NEURON_TANH_INJECTION", "CONTEXT_WIRE", "MULTIPLY_INJECTION", "BOOLEAN_COMPOSE", "COMPOUND_MULTIPLY_NEURON", "COMPOUND_TANH_SERIES", "COMPOUND_MULTIPLY3_NEURON", "COMPOUND_MULTIPLY_ABS", "RECURRENT_SELF_WIRE", "SIN_INJECTION", "DEEP_INSERTION", "RECURRENT_XOR", "MULTI_LAYER_STACK", "PATCH_POOLING", "PARITY_TREE", "DIVIDE_INJECTION", "COMPOUND_SIN_PRODUCT", "COMPOUND_DIVIDE_PRODUCT"};

                struct ShadowSpec {
                    int         rank;
                    int         type;  // Hypothesis::Type as int
                    std::unique_ptr<Graph> shadow;
                };
                std::vector<ShadowSpec> specs;

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

                    std::unique_ptr<Graph> shadow = apply_shadow_routing(hyp, diag);
                    if (!shadow) {
                        Logger::verbose("  shadow routing failed for rank=" + std::to_string(hyp_idx));
                        hyp_idx++;
                        continue;
                    }

                    specs.push_back({hyp_idx, static_cast<int>(hyp.type), std::move(shadow)});
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
                            results[i] = validate_shadow_only(
                                specs[i].shadow,
                                current_loss,
                                baseline_val,
                                specs[i].rank,
                                specs[i].type);
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
                            Logger::verbose("  rank=" + std::to_string(results[i].hyp_rank)
                                           + " REJECT 鈥?" + results[i].reject_reason);
                            continue;
                        }
                        double effective = results[i].val_loss
                                         + rank_bonus * static_cast<double>(results[i].hyp_rank);
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
                        graph_ = std::move(specs[winner_idx].shadow);
                        stats_.structural_changes++;
                        current_loss = evaluate_loss(training_data_);
                        Logger::info("  COMMIT rank=" + std::to_string(results[winner_idx].hyp_rank)
                                    + " type=" + hyp_names[results[winner_idx].hyp_type]
                                    + " val_loss=" + std::to_string(results[winner_idx].val_loss)
                                    + " train_loss=" + std::to_string(results[winner_idx].train_loss)
                                    + " new train_loss=" + std::to_string(current_loss));
                        structural_commit_succeeded = true;
                        committed_hyp_type = results[winner_idx].hyp_type;
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
                        == static_cast<int>(Hypothesis::COMPOUND_DIVIDE_PRODUCT)) {
                    plateau_counter_ = -config::COMPOUND_COMMIT_GRACE_EPOCHS;
                    epochs_since_structural_ = -config::COMPOUND_COMMIT_GRACE_EPOCHS;
                    Logger::info("Compound grace period 鈥?" + std::to_string(config::COMPOUND_COMMIT_GRACE_EPOCHS)
                                + " extra epochs before next structural attempt");
                }
                Logger::info("Plateau state reset 鈥?best_phase2=" + std::to_string(best_phase2_loss_));
                consecutive_structural_failures_ = 0;

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
        // (CIFAR gray: train 1.9 / val 5.1 鈥?worse than the 3.25 base rate).
        if (!validation_data_.samples.empty()) {
            double val_loss = compute_validation_loss(*graph_);
            if (val_loss < best_val_loss_) {
                best_val_loss_ = val_loss;
                best_val_snapshot_train_loss_ = current_loss;
                best_graph_snapshot_ = graph_->clone();
                Logger::verbose("New best val loss: " + std::to_string(val_loss)
                               + " (train " + std::to_string(current_loss) + ")");
            }
        } else if (current_loss < best_overall_loss_) {
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

    // Collect all nodes in the graph for local_inputs 鈥?any node could be upstream
    // We'll store all node outputs as "local inputs" (candidates for reconnection)
    for (const auto& sample : training_data_.samples) {
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
        if (ibs.condition_source_node != 0) {
            auto reg_it = blackboard_registry_.find(ibs.condition_source_node);
            if (reg_it != blackboard_registry_.end() && reg_it->second.size() >= 2) {
                std::vector<Value> src_sorted = reg_it->second;
                std::sort(src_sorted.begin(), src_sorted.end());
                ibs.split_threshold = src_sorted[src_sorted.size() / 2];
                threshold_found = true;
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
        else if (ftype == FailureType::LINEAR_OFFSET)     score = std::max(score, config::SCORE_IFELSE_LINEAR_BOOST);
        candidates.push_back({std::move(ibs), score});
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
            if (high_complexity && graph_input_count <= config::MULTI_LAYER_STACK_MAX_INPUTS) {
                Hypothesis mls;
                mls.type = Hypothesis::MULTI_LAYER_STACK;
                mls.compound_K = config::MULTI_LAYER_STACK_K;
                double score = config::SCORE_MULTI_LAYER_STACK;
                if (profile.lipschitz_max > config::MULTI_LAYER_STACK_BOOST_LIPSCHITZ
                    && !product_signature) {
                    score = 0.95;  // extreme complexity → rank above everything
                    mls.compound_K = config::MULTI_LAYER_STACK_K_MAX;
                }
                candidates.push_back({std::move(mls), score});
                Logger::info("Candidate emitted: MULTI_LAYER_STACK (lipschitz="
                            + std::to_string(profile.lipschitz_max)
                            + " var_r=" + std::to_string(profile.var_r) + ")");
            }
        }

        // --- Candidate 3d: PATCH_POOLING 鈥?coarse convolutional prior ---
        // For image-like input layouts (input count a perfect square 鈮?        // PATCH_POOL_MIN_SIDE虏, or 3脳square for pixel-interleaved RGB),
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
            double freq_est = 0.0;
            if (sorted_xt.size() >= 2) {
                double x_lo = sorted_xt.front().first, x_hi = sorted_xt.back().first;
                double x_range = x_hi - x_lo;
                if (x_range > 1e-6) {
                    double periods = static_cast<double>(sign_changes) / 2.0;
                    constexpr double kTwoPi = 6.28318530717958647692;
                    freq_est = kTwoPi * periods / x_range;
                    if (freq_est < 0.5 || freq_est > 60.0) freq_est = 0.0;  // sanity clamp
                }
            }
            uint64_t src_id = graph_input_ids[std::min(axis, graph_input_ids.size()-1)];

            // Freq-init variant: starts at the estimated frequency; SGD only
            // refines amplitude/phase. Ranked slightly above the zero-init
            // fallback; validation gate decides which (if either) commits.
            if (freq_est > 0.0) {
                Hypothesis sinj_f;
                sinj_f.type = Hypothesis::SIN_INJECTION;
                sinj_f.multiply_source_a = src_id;
                sinj_f.sin_freq_init = freq_est;
                candidates.push_back({std::move(sinj_f), config::SCORE_COMPOUND_TANH_SERIES + 0.01});
                Logger::info("Candidate emitted: SIN_INJECTION freq-init (w="
                            + std::to_string(freq_est) + ", " + std::to_string(sign_changes)
                            + " sign changes)");
            }
            // Zero-init variant (proven fallback; identity start).
            Hypothesis sinj;
            sinj.type = Hypothesis::SIN_INJECTION;
            sinj.multiply_source_a = src_id;
            candidates.push_back({std::move(sinj), config::SCORE_COMPOUND_TANH_SERIES});
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
            auto matches = library_->find_matches(needed, 1);
            if (!matches.empty() && matches[0].distance < config::LIBRARY_MATCH_THRESHOLD) {
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
                        sh.score += config::LIBRARY_BOOST;
                        already_present = true;
                        Logger::info("Library prior (dist="
                                     + std::to_string(matches[0].distance).substr(0,5)
                                     + " src=" + library_->entry(matches[0].index).source_task
                                     + ") boosts " + hnames[static_cast<int>(suggested)]
                                     + " by " + std::to_string(config::LIBRARY_BOOST));
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
        // threshold constant 鈥?NOT the failing node's (tanh-squashed) output.
        // This is the key fix: previously, GREATER(failing_node, condition_src)
        // split on the post-activation output (bounded [-1,1]), which could not
        // correctly partition the input domain.
        //
        // IFELSE semantics: input[0]=condition, input[1]=value.
        //   condition=true  鈫?output[0]=value, output[1]=0
        //   condition=false 鈫?output[0]=0,       output[1]=value
        //
        // We connect output[1] (false branch) to downstream. This creates a
        // domain mask: "pass failing_node through when x 鈮?threshold, output 0
        // when x > threshold." For cliff/piecewise tasks where one region should
        // be suppressed, this is the correct behavior.

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

        // Gain NEURON — least-squares init (was 0.0: the quotient-of-
        // products starts at the scale the residual wants, immediately
        // competitive in first-cycle validation; I.32.8's blocker).
        uint64_t gain_id = shadow->add_node(NodeType::NEURON, "divprod_gain");
        Node* gn = shadow->get_node(gain_id);
        if (gn && gn->get_type() == NodeType::NEURON) {
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
double EvolutionEngine::compute_validation_loss(Graph& g) {
    if (validation_data_.samples.empty()) return 0.0;

    // NOTE: do NOT reset recurrent state here. For sequence tasks (d6),
    // the validation targets are cumulative from the start of the full
    // sequence. The caller (validate_shadow_only) runs the sanity check
    // on all training samples first, which leaves the recurrent state
    // at the correct position for the start of validation. Resetting
    // would break running-parity / running-sum tasks.
    double total_loss = 0.0;
    int count = 0;
    for (const auto& sample : validation_data_.samples) {
        for (const auto& kv : sample.inputs) {
            auto map_it = input_data_to_graph_.find(kv.first);
            if (map_it != input_data_to_graph_.end()) {
                g.set_input_value(map_it->second, kv.second);
            }
        }
        g.execute();
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
    int hyp_type) {

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
    graph_eliminate_dead_code(*shadow_graph);
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
            || hyp_type == static_cast<int>(Hypothesis::SIN_INJECTION)
            || hyp_type == static_cast<int>(Hypothesis::DEEP_INSERTION)
            || hyp_type == static_cast<int>(Hypothesis::MULTI_LAYER_STACK)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_SIN_PRODUCT)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_DIVIDE_PRODUCT)) {
            train_cfg.epochs *= config::SHADOW_COMPOUND_SGD_MULTIPLIER;
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
            || hyp_type == static_cast<int>(Hypothesis::DEEP_INSERTION)
            || hyp_type == static_cast<int>(Hypothesis::MULTI_LAYER_STACK)
            || hyp_type == static_cast<int>(Hypothesis::PATCH_POOLING)
            || hyp_type == static_cast<int>(Hypothesis::COMPOUND_SIN_PRODUCT)) {
            train_cfg.learning_rate *= config::SHADOW_COMPOUND_LR_MULTIPLIER;
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
        shadow_graph->train(training_data_.samples, train_cfg);
    }

    // Sanity check: shadow's loss on the training data (the data it was just
    // trained on). Two guards: catastrophic-damage (1.5x) and drift
    // (max of abs/rel tolerance).
    if (!training_data_.samples.empty() && baseline_loss > 0.0) {
        shadow_graph->reset_recurrent_state();
        double shadow_train_loss = 0.0;
        int train_count = 0;
        for (const auto& sample : training_data_.samples) {
            for (const auto& kv : sample.inputs) {
                auto map_it = input_data_to_graph_.find(kv.first);
                if (map_it != input_data_to_graph_.end()) {
                    shadow_graph->set_input_value(map_it->second, kv.second);
                }
            }
            shadow_graph->execute();
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
    } else if ((hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY_NEURON)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_TANH_SERIES)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY3_NEURON)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_MULTIPLY_ABS)
                || hyp_type == static_cast<int>(Hypothesis::RECURRENT_SELF_WIRE)
                || hyp_type == static_cast<int>(Hypothesis::SIN_INJECTION)
                || hyp_type == static_cast<int>(Hypothesis::DEEP_INSERTION)
                || hyp_type == static_cast<int>(Hypothesis::MULTI_LAYER_STACK)
                || hyp_type == static_cast<int>(Hypothesis::PATCH_POOLING)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_SIN_PRODUCT)
                || hyp_type == static_cast<int>(Hypothesis::COMPOUND_DIVIDE_PRODUCT))
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
