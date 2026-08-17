#include "evolution.h"
#include "logger.h"
#include "constants.h"
#include "subgraph_library.h"
#include "serialize.h"
#include <windows.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <map>
#include <queue>
#include <unordered_map>

namespace gpnn {

void print_header() {
    std::cout << "================================================\n";
    std::cout << "  GP-NN Evolutionary Graph Training System\n";
    std::cout << "================================================\n\n";
}

void print_progress(int epoch, double loss, const std::string& phase) {
    if (epoch == 0) return;  // skip initialisation tick

    int bar_len = config::PROGRESS_BAR_DEFAULT_WIDTH;
    int filled = static_cast<int>(loss * bar_len / 2.0);
    if (filled < 0) filled = 0;
    if (filled > bar_len) filled = bar_len;

    std::ios state(nullptr);
    state.copyfmt(std::cout);

    std::cout << "  [" << std::setw(4) << epoch << "] "
              << std::setw(6) << phase << "  loss="
              << std::fixed << std::setprecision(6) << loss
              << "  |" << std::string(filled, '=');

    std::cout.copyfmt(state);
    std::cout << "|\r" << std::flush;
}

} // namespace gpnn

int main(int argc, char* argv[]) {
    using namespace gpnn;

    // Unbuffered stdout: when redirected to a file (background runs),
    // block buffering would hide all progress until exit.
    std::cout.setf(std::ios::unitbuf);

    // ---------- defaults ----------
    std::string csv_path;
    int input_cols  = 0;
    int output_cols = config::DEFAULT_OUTPUT_COLS;
    int max_epochs  = config::DEFAULT_CLI_MAX_EPOCHS;
    bool has_header = false;
    bool verbose = false;
    std::string log_file;

    // Post-training prediction sweep across an input range.
    // Used to measure empirical "switch point" on piecewise / cliff tasks.
    bool   sweep_enabled = false;
    double sweep_min     = config::DEFAULT_SWEEP_MIN;
    double sweep_max     = config::DEFAULT_SWEEP_MAX;
    double sweep_step    = config::DEFAULT_SWEEP_STEP;

    // Held-out test-set evaluation (Feynman/SRBench harness support).
    std::string eval_csv_path;

    // Structural analysis of the final graph (node histogram, depth, params).
    bool dump_graph = false;

    // Periodic graph checkpointing (empty = derive default; "none" = off).
    std::string save_graph_path;
    int save_interval = 25;

    // Offline mode: load a serialized graph instead of evolving from
    // scratch. With --max-epochs 0 this enables standalone analysis
    // (--dump-graph) and evaluation (--eval-csv) of checkpointed graphs
    // without any training.
    std::string load_graph_path;

    // Dataset split + loss options.
    bool   no_shuffle = false;                          // preserve row order (sequence/recurrence tasks)
    Graph::LossType loss_type = Graph::LossType::MSE;   // --loss {mse,bce}

    // Determinism. Default fixed seed => reproducible runs; --seed overrides
    // (0 => non-deterministic, uses random_device).
    unsigned int seed = config::DEFAULT_SEED;

    // ---------- simple CLI parse ----------
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--input-cols" && i + 1 < argc) {
            input_cols = std::stoi(argv[++i]);
        } else if (arg == "--output-cols" && i + 1 < argc) {
            output_cols = std::stoi(argv[++i]);
        } else if (arg == "--max-epochs" && i + 1 < argc) {
            max_epochs = std::stoi(argv[++i]);
        } else if (arg == "--header") {
            has_header = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--log-file" && i + 1 < argc) {
            log_file = argv[++i];
        } else if (arg == "--sweep" && i + 3 < argc) {
            sweep_min     = std::stod(argv[++i]);
            sweep_max     = std::stod(argv[++i]);
            sweep_step    = std::stod(argv[++i]);
            sweep_enabled = true;
        } else if (arg == "--no-shuffle") {
            no_shuffle = true;
        } else if (arg == "--loss" && i + 1 < argc) {
            std::string lt = argv[++i];
            if (lt == "bce")      loss_type = Graph::LossType::BCE;
            else if (lt == "mse") loss_type = Graph::LossType::MSE;
            else {
                std::cerr << "Error: --loss must be 'mse' or 'bce' (got '" << lt << "')\n";
                return 1;
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        } else if (arg == "--eval-csv" && i + 1 < argc) {
            eval_csv_path = argv[++i];
        } else if (arg == "--save-graph" && i + 1 < argc) {
            save_graph_path = argv[++i];
        } else if (arg == "--save-interval" && i + 1 < argc) {
            save_interval = std::stoi(argv[++i]);
        } else if (arg == "--load-graph" && i + 1 < argc) {
            load_graph_path = argv[++i];
        } else if (arg == "--dump-graph") {
            dump_graph = true;
        } else if (arg == "--help") {
            std::cout << "Usage: gpnn --csv <file> --input-cols <N> [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --csv <file>        CSV dataset file (required)\n";
            std::cout << "  --input-cols <N>    Number of input columns (required)\n";
            std::cout << "  --output-cols <N>   Number of output/target columns (default: 1)\n";
            std::cout << "  --max-epochs <N>    Evolution epochs (default: " << config::DEFAULT_CLI_MAX_EPOCHS << ")\n";
            std::cout << "  --header            First row is a header to skip\n";
            std::cout << "  --verbose           Enable verbose logging (decision-level detail)\n";
            std::cout << "  --log-file <path>   Redirect log output to file\n";
            std::cout << "  --eval-csv <path>   After training, evaluate held-out CSV; print Eval R2/MSE/MAE per output\n";
            std::cout << "  --save-graph <dir>  Checkpoint best graph every --save-interval epochs (default: checkpoints/)\n";
            std::cout << "  --save-interval <N> Checkpoint interval in epochs (default: 25)\n";
            std::cout << "  --sweep <min> <max> <step>  After training, sweep input x across [min,max] and print predictions\n";
            std::cout << "  --no-shuffle          Preserve CSV row order in train/val split (sequence/recurrence tasks)\n";
            std::cout << "  --loss <mse|bce>      Loss function (default: mse)\n";
            std::cout << "  --seed <N>            RNG seed (default: " << config::DEFAULT_SEED << ", 0 = random)\n";
            return 0;
        }
    }

    if (csv_path.empty() || input_cols <= 0) {
        std::cerr << "Error: --csv and --input-cols are required.\n";
        std::cerr << "Run with --help for usage.\n";
        return 1;
    }

    Logger::init(verbose, log_file);

    print_header();

    // ---------- build input/output ID vectors ----------
    std::vector<uint64_t> input_ids(input_cols);
    for (int i = 0; i < input_cols; ++i) input_ids[i] = static_cast<uint64_t>(i);

    std::vector<uint64_t> output_ids(output_cols);
    for (int i = 0; i < output_cols; ++i) output_ids[i] = static_cast<uint64_t>(i);

    // ---------- load dataset ----------
    std::cout << "Loading dataset from: " << csv_path << "\n";
    Dataset full_data = load_csv_dataset(csv_path, input_cols, input_ids, output_ids, has_header);
    if (full_data.empty()) {
        std::cerr << "Error: no samples loaded from " << csv_path << "\n";
        return 1;
    }
    std::cout << "  Loaded " << full_data.size() << " samples\n";

    // ---------- normalize input features (z-score standardization) ----------
    // Standardize each input feature to mean≈0, std≈1. Critical for
    // high-dimensional data with large value ranges (e.g. 64-pixel MNIST
    // with values in [0,16]). Without normalization, large feature magnitudes
    // cause sigmoid saturation and OUTPUT-scale collapse.
    // SKIPPED for: (a) binary (0/1) features (breaks XOR truthiness), and
    // (b) features already in a reasonable range (std < 2.0) where tanh-based
    // neurons work well without rescaling.
    // The transform params are kept in scope so --eval-csv data gets the
    // SAME transform (train/eval must live in the same input space).
    std::unordered_map<uint64_t, double> feat_mean, feat_std;
    std::unordered_map<uint64_t, bool>   feat_binary;
    {
        std::unordered_map<uint64_t, double> feat_sq;
        for (const auto& s : full_data.samples) {
            for (const auto& kv : s.inputs) {
                feat_mean[kv.first] += kv.second;
                feat_sq[kv.first]   += kv.second * kv.second;
                if (kv.second < -0.01 || (kv.second > 0.01 && std::abs(kv.second - 1.0) > 0.01))
                    feat_binary[kv.first] = false;
                else
                    feat_binary.try_emplace(kv.first, true);
            }
        }
        size_t n = full_data.samples.size();
        for (auto& kv : feat_mean) {
            double m = kv.second / n;
            double var = feat_sq[kv.first] / n - m * m;
            feat_std[kv.first] = (var > 1e-12) ? std::sqrt(var) : 1.0;
            kv.second = m;
        }
        for (auto& s : full_data.samples) {
            for (auto& kv : s.inputs) {
                if (feat_binary[kv.first]) continue;  // keep raw 0/1 values
                if (feat_std[kv.first] < 2.0) continue;  // already well-scaled
                kv.second = (kv.second - feat_mean[kv.first]) / feat_std[kv.first];
            }
        }
    }

    // ---------- train / validation split ----------
    Dataset train_data, val_data;
    full_data.split(train_data, val_data, config::DEFAULT_VALIDATION_FRACTION,
                    /*seed=*/seed, /*shuffle=*/!no_shuffle);
    std::cout << "  Train: " << train_data.size()
              << "  Validation: " << val_data.size()
              << (no_shuffle ? "  [order preserved]\n\n" : "\n\n");

    // ---------- build initial graph (empty — EvolutionEngine seeds it) ----------
    auto graph = std::make_unique<Graph>();

    // ---------- offline load (--load-graph) ----------
    // Deserialize a checkpointed graph. The engine's ensure_minimal_
    // architecture skips seeding for non-empty graphs and rebuilds the
    // data-ID mappings from node names ("input_N"/"output_M").
    if (!load_graph_path.empty()) {
        try {
            load_graph_from_file(*graph, load_graph_path);
            std::cout << "  Loaded graph from " << load_graph_path
                      << " (" << graph->node_count() << " nodes)\n";
        } catch (...) {
            std::cerr << "Error: failed to load graph from " << load_graph_path << "\n";
            return 1;
        }
    }

    // ---------- configure evolution ----------
    EvolutionEngine::Config cfg;
    cfg.max_epochs = max_epochs;  // CLI override; all other fields use Config defaults
    cfg.loss_type  = loss_type;   // CLI override (mse/bce)
    cfg.seed       = seed;        // CLI override (determinism)
    cfg.sequence_mode = no_shuffle;  // sequence/recurrence candidates gated on this
    cfg.snapshot_interval = save_interval;

    // Checkpoint path: problem-specific folder derived from the CSV name —
    // checkpoints/<csv-stem>/graph.json (e.g. checkpoints/bench_digits/graph.json).
    // Explicit --save-graph overrides (uses the path verbatim as the FILE
    // path); "none" disables. Folder is created if missing.
    if (save_graph_path != "none") {
        std::string snap_path = save_graph_path;
        if (snap_path.empty()) {
            std::string stem = csv_path;
            auto slash = stem.find_last_of("/\\");
            if (slash != std::string::npos) stem = stem.substr(slash + 1);
            auto dot = stem.find_last_of('.');
            if (dot != std::string::npos) stem = stem.substr(0, dot);
            snap_path = "checkpoints/" + stem + "/graph.json";
        }
        // Ensure the directory exists (create all levels).
        auto dir_end = snap_path.find_last_of("/\\");
        if (dir_end != std::string::npos) {
            std::string dir = snap_path.substr(0, dir_end);
            std::string acc;
            size_t pos = 0;
            while (pos <= dir.size()) {
                auto sep = dir.find_first_of("/\\", pos);
                if (sep == std::string::npos) sep = dir.size();
                acc += dir.substr(pos, sep - pos);
                if (!acc.empty()) CreateDirectoryA(acc.c_str(), nullptr);
                acc += "/";
                pos = sep + 1;
                if (sep == dir.size()) break;
            }
        }
        cfg.snapshot_path = snap_path;
        std::cout << "  Checkpointing to: " << snap_path
                  << " (every " << save_interval << " epochs)\n";
    }

    // ---------- create engine ----------
    EvolutionEngine engine(std::move(graph), std::move(train_data), std::move(val_data), cfg);

    // Load the subgraph library (if it exists) so the engine can use it as a
    // behavioral prior during candidate generation. The library accumulates
    // across runs (post-evolution extraction saves to subgraph_library.txt).
    SubgraphLibrary global_lib;
    if (global_lib.load("subgraph_library.txt")) {
        engine.set_library(&global_lib);
        std::cout << "  Loaded subgraph library: " << global_lib.size() << " entries\n";
    }

    // ---------- run evolution ----------
    auto t_start = std::chrono::steady_clock::now();

    std::cout << "Starting evolution...\n";
    engine.evolve(print_progress);
    std::cout << "\n\n";

    auto t_end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    // ---------- report ----------
    const auto& stats = engine.get_stats();

    std::cout << "================================================\n";
    std::cout << "  Evolution Complete\n";
    std::cout << "================================================\n\n";
    std::cout << "  Total epochs:         " << stats.total_epochs          << "\n";
    std::cout << "  SGD epochs run:       " << stats.sgd_epochs_run        << "\n";
    std::cout << "  Structural changes:   " << stats.structural_changes    << "\n";
    std::cout << "  Failed commits:       " << stats.failed_commits        << "\n";
    std::cout << "  Compiles performed:   " << stats.compiles_performed    << "\n";
    std::cout << "  Dead nodes removed:   " << stats.dead_nodes_removed    << "\n";
    std::cout << "  Constants folded:     " << stats.constants_folded      << "\n";
    std::cout << "  Neurons compressed:   " << stats.neurons_compressed    << "\n";
    std::cout << "  Initial loss:         " << stats.initial_loss          << "\n";
    std::cout << "  Final loss:           " << stats.final_loss            << "\n";
    std::cout << "  Best loss:            " << stats.best_loss             << "\n";
    std::cout << "  Wall time:            " << ms << " ms\n";

    // ---------- post-evolution: subgraph library extraction (OFFLINE) ----------
    // Compute a behavioral fingerprint of the evolved graph and add it to a
    // persistent library. This does NOT slow evolution — it runs once, after
    // evolve() returns. On subsequent runs, the library is loaded and can be
    // queried to find previously-solved tasks with similar behavior.
    {
        const std::string lib_path = "subgraph_library.txt";

        // Collect INPUT and OUTPUT node IDs from the trained graph.
        std::vector<uint64_t> input_ids;
        uint64_t first_output = 0;
        bool found_out = false;
        for (const auto& n : engine.get_graph().get_nodes()) {
            if (n->get_type() == NodeType::INPUT) input_ids.push_back(n->get_id());
            if (!found_out && n->get_type() == NodeType::OUTPUT) { first_output = n->get_id(); found_out = true; }
        }

        if (!input_ids.empty() && found_out) {
            // Print the symbolic expression (the graph as a readable formula).
            std::string expr = engine.get_graph().to_expression(first_output);
            // Truncate very long expressions for terminal readability.
            if (expr.size() > 200) expr = expr.substr(0, 197) + "...";
            std::cout << "\n  [Expression] " << expr << "\n";

            // Compute fingerprint (runs the graph on a probe bench — cheap, offline).
            BehavioralFingerprint fp = fingerprint_subgraph(engine.get_graph(), input_ids, first_output);

            // Build a short description from the fingerprint.
            std::ostringstream desc;
            desc << "k=" << fp.num_inputs
                 << " bounded=" << fp.bounded
                 << " interact=" << fp.interaction_dominant
                 << " sign_sym=" << fp.sign_symmetric
                 << " sharp=" << fp.sharp_boundary
                 << " lipsch=" << std::fixed << std::setprecision(1) << fp.lipschitz_max;

            std::cout << "\n  [Library] Behavioral fingerprint: " << desc.str() << "\n";

            // Load existing library, find matches, add this entry, save.
            SubgraphLibrary lib;
            lib.load(lib_path);
            if (lib.size() > 0) {
                auto matches = lib.find_matches(fp, 3);
                if (!matches.empty()) {
                    std::cout << "  [Library] Closest existing entries:\n";
                    for (const auto& m : matches) {
                        std::cout << "    " << lib.entry(m.index).source_task
                                  << " (dist=" << std::fixed << std::setprecision(3) << m.distance << ")\n";
                    }
                }
            }

            // Extract task name from CSV path (strip dir + extension).
            std::string task_name = csv_path;
            auto slash = task_name.find_last_of("/\\");
            if (slash != std::string::npos) task_name = task_name.substr(slash + 1);

            SubgraphLibraryEntry entry;
            entry.fingerprint = fp;
            entry.source_task = task_name;
            entry.description = desc.str();
            // Generate canonical expression + pattern for dedup and display.
            std::string raw_expr = engine.get_graph().to_expression(first_output);
            entry.canonical_expression = canonicalize_expression(raw_expr);
            entry.pattern = recognize_pattern(entry.canonical_expression);
            std::cout << "  [Library] Pattern: " << entry.pattern << "\n";
            bool added = lib.add(entry);
            lib.save(lib_path);
            if (added) {
                std::cout << "  [Library] Saved (library now has " << lib.size() << " entries)\n";
            } else {
                std::cout << "  [Library] Skipped (duplicate expression)\n";
            }

            // Extract reusable sub-expression blocks (product, sin, boundary, etc.)
            // from the canonical expression and add each as a fine-grained entry.
            auto sub_entries = extract_sub_expressions(entry.canonical_expression, task_name);
            int sub_added = 0;
            for (const auto& se : sub_entries) {
                if (lib.add(se)) ++sub_added;
            }
            if (sub_added > 0) {
                std::cout << "  [Library] Extracted " << sub_added << " sub-expression block(s)";
                lib.save(lib_path);
                std::cout << " (library now has " << lib.size() << " entries)\n";
            }
        }
    }
    // ---------- end subgraph library extraction ----------

    // ---------- structural analysis (--dump-graph) ----------
    // Prints a machine-parseable summary of the final graph: node counts
    // by type, edges, trainable parameter count, and depth (longest
    // INPUT→OUTPUT path, recurrent edges excluded). Consumed by
    // analyze_graphs.py for cross-run distribution analysis.
    if (dump_graph) {
        const auto& gref = engine.get_graph();
        const auto& nodes = gref.get_nodes();
        const auto& conns = gref.get_connections();

        // Node-type histogram
        std::map<NodeType, int> type_count;
        for (const auto& n : nodes) ++type_count[n->get_type()];

        // Trainable parameters: NEURON/LINEAR weights+biases, OUTPUT
        // scale+bias, CONSTANT values.
        size_t params = 0;
        for (const auto& n : nodes) {
            auto t = n->get_type();
            if (t == NodeType::NEURON || t == NodeType::LINEAR) {
                auto* nn = static_cast<const NeuronNode*>(n.get());
                params += nn->get_num_weights() + 1;
            } else if (t == NodeType::OUTPUT) {
                params += 2;
            } else if (t == NodeType::CONSTANT) {
                params += 1;
            }
        }

        // Depth: longest acyclic INPUT→OUTPUT path (Kahn levels).
        // Recurrent edges are excluded (they're state, not depth).
        std::unordered_map<uint64_t, int> level;
        std::unordered_map<uint64_t, int> indeg;
        std::unordered_map<uint64_t, std::vector<uint64_t>> adj;
        for (const auto& c : conns) {
            if (c.is_recurrent) continue;
            adj[c.src_node].push_back(c.dst_node);
            ++indeg[c.dst_node];
        }
        std::queue<uint64_t> q;
        for (const auto& n : nodes) {
            if (indeg[n->get_id()] == 0) {
                level[n->get_id()] = (n->get_type() == NodeType::INPUT) ? 1 : 0;
                q.push(n->get_id());
            }
        }
        int max_depth = 0;
        while (!q.empty()) {
            uint64_t u = q.front(); q.pop();
            max_depth = std::max(max_depth, level[u]);
            for (uint64_t v : adj[u]) {
                level[v] = std::max(level[v], level[u] + 1);
                if (--indeg[v] == 0) q.push(v);
            }
        }

        // Fan-in / fan-out means
        double fan_in = 0.0, fan_out = 0.0;
        for (const auto& n : nodes) {
            fan_in  += static_cast<double>(n->get_num_inputs());
        }
        std::unordered_map<uint64_t, int> outdeg;
        for (const auto& c : conns) ++outdeg[c.src_node];
        for (const auto& kv : outdeg) fan_out += kv.second;

        std::cout << "\n  [Graph] nodes=" << nodes.size()
                  << " edges=" << conns.size()
                  << " params=" << params
                  << " depth=" << max_depth << "\n";
        std::cout << "  [Graph] types:";
        for (const auto& kv : type_count) {
            std::cout << " " << node_type_to_string(kv.first) << "=" << kv.second;
        }
        std::cout << "\n";
        std::cout << "  [Graph] fan_in_mean="
                  << (nodes.empty() ? 0.0 : fan_in / nodes.size())
                  << " fan_out_mean="
                  << (nodes.empty() ? 0.0 : fan_out / nodes.size()) << "\n";
    }

    // ---------- held-out test-set evaluation (--eval-csv) ----------
    // Loads an external CSV with the same column layout, runs the trained
    // graph on it, and prints per-output MSE / R² / MAE. R² is computed
    // against the test targets' own mean — the SRBench convention.
    if (!eval_csv_path.empty()) {
        Dataset eval_data = load_csv_dataset(eval_csv_path, input_cols, input_ids, output_ids, has_header);
        if (eval_data.empty()) {
            std::cerr << "Error: no samples loaded from eval file " << eval_csv_path << "\n";
        } else {
            // Apply the SAME normalization transform as training data
            for (auto& s : eval_data.samples) {
                for (auto& kv : s.inputs) {
                    if (feat_binary.count(kv.first) && feat_binary[kv.first]) continue;
                    if (feat_std.count(kv.first) && feat_std[kv.first] >= 2.0) {
                        kv.second = (kv.second - feat_mean[kv.first]) / feat_std[kv.first];
                    }
                }
            }
            auto results = engine.evaluate_external(eval_data);
            std::cout << "\n================================================\n";
            std::cout << "  Held-out Evaluation  (" << eval_data.size() << " samples)\n";
            std::cout << "================================================\n";
            for (size_t j = 0; j < results.size(); ++j) {
                std::cout << "  Eval R2[out" << j << "]:   " << std::fixed << std::setprecision(6)
                          << results[j].r2 << "\n";
                std::cout << "  Eval MSE[out" << j << "]:  " << results[j].mse
                          << "   MAE: " << results[j].mae << "\n";
            }

            // ---- Softmax cross-entropy + accuracy (comparable LM metric) ----
            // When outputs form a one-hot class set (LM / classification),
            // one-vs-rest BCE numbers are NOT comparable to published
            // softmax-CE figures (unigram floor etc.). Compute proper
            // softmax cross-entropy over ALL outputs per sample, plus
            // top-1 accuracy, plus bits/unit (CE / ln 2).
            if (results.size() >= 2 && cfg.loss_type == Graph::LossType::BCE) {
                auto sm = engine.evaluate_external_softmax(eval_data);
                if (sm.size() == 2) {
                    constexpr double kLn2 = 0.6931471805599453;
                    std::cout << "  Eval SoftmaxCE:   " << std::setprecision(6) << sm[0]
                              << "   (" << (sm[0] / kLn2) << " bits/unit)" << "\n";
                    std::cout << "  Eval Accuracy:    " << (100.0 * sm[1]) << "%\n";
                }
            }
        }
    }

    // ---------- post-training sweep ----------
    // Walk input x across [sweep_min, sweep_max] in step increments and print
    // (x, prediction). For piecewise/cliff tasks this reveals the empirical
    // "switch point" — the x value where the model's output changes character.
    if (sweep_enabled) {
        // Take ownership of the trained graph (engine surrenders it).
        auto trained = engine.take_graph();

        // Locate the first INPUT and OUTPUT nodes (single-input/single-output case).
        uint64_t in_node = 0, out_node = 0;
        bool found_in = false, found_out = false;
        for (const auto& n : trained->get_nodes()) {
            if (!found_in  && n->get_type() == NodeType::INPUT)  { in_node  = n->get_id(); found_in  = true; }
            if (!found_out && n->get_type() == NodeType::OUTPUT) { out_node = n->get_id(); found_out = true; }
        }

        if (found_in && found_out) {
            std::cout << "\n================================================\n";
            std::cout << "  Prediction Sweep  (x in [" << sweep_min << ", " << sweep_max
                      << "], step " << sweep_step << ")\n";
            std::cout << "================================================\n";
            std::cout << "      x          pred\n";
            for (double x = sweep_min; x <= sweep_max + config::SWEEP_RANGE_EPSILON; x += sweep_step) {
                trained->set_input_value(in_node, x);
                trained->execute();
                Value pred = trained->get_output_value(out_node);
                std::cout << "  " << std::setw(8) << std::fixed << std::setprecision(4) << x
                          << "    " << std::setw(8) << std::setprecision(4) << pred << "\n";
            }
        } else {
            std::cerr << "Sweep requested but could not find INPUT/OUTPUT nodes.\n";
        }
    }

    Logger::close();

    return 0;
}
