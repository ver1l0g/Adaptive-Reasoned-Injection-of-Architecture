#pragma once

#include "graph.h"
#include "constants.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <random>
#include <unordered_set>

namespace gpnn {

// ============================================================================
// Dataset helper — holds training / validation sample pairs
// ============================================================================
struct Dataset {
    std::vector<Graph::SampleIODesc> samples;

    size_t size() const { return samples.size(); }
    bool empty() const { return samples.empty(); }

    // Split into train + validation fractions. When shuffle=false the sample
    // order is preserved (train = first fraction, val = tail) — required for
    // sequence/recurrence datasets where row order carries temporal meaning.
    void split(Dataset& train, Dataset& val, double val_fraction = config::DEFAULT_VALIDATION_FRACTION, unsigned int seed = 0, bool shuffle = true) const;
};

// Load dataset from CSV file.
// format: each row is input1,input2,...,target
// input_cols: number of input columns (remaining columns are outputs)
// header: whether first row is a header to skip
Dataset load_csv_dataset(const std::string& filepath, int input_cols,
                         const std::vector<uint64_t>& input_ids,
                         const std::vector<uint64_t>& output_ids,
                         bool header = false);

// Forward declaration for the subgraph library (used as a behavioral prior).
class SubgraphLibrary;

// ============================================================================
// EvolutionEngine — high-level orchestrator of the 7-phase evolution loop
// ============================================================================
class EvolutionEngine {
public:
    // Set a loaded subgraph library to use as a behavioral prior during
    // candidate generation. nullptr = no library (no overhead). The engine
    // does NOT own the pointer.
    void set_library(const SubgraphLibrary* lib) { library_ = lib; }
public:
    struct Config {
        int   max_epochs              = config::DEFAULT_EVOLUTION_MAX_EPOCHS;     // outer evolution epochs
        int   sgd_epochs_per_phase    = config::DEFAULT_SGD_EPOCHS_PER_PHASE;    // Phase 2 SGD iterations per evolution cycle
        double sgd_learning_rate       = config::DEFAULT_TRAIN_LEARNING_RATE;
        double sgd_gradient_clip       = config::DEFAULT_TRAIN_GRADIENT_CLIP;
        double sgd_momentum            = config::DEFAULT_SGD_MOMENTUM;
        double sgd_weight_decay        = config::DEFAULT_SGD_WEIGHT_DECAY;
        int   plateau_patience         = config::DEFAULT_PLATEAU_PATIENCE;        // SGD epochs without improvement before Phase 3
        double plateau_min_improvement = config::DEFAULT_PLATEAU_MIN_IMPROVEMENT; // minimum loss decrease to count as progress
        int   force_structural_every   = config::DEFAULT_FORCE_STRUCTURAL_EVERY;  // force structural evolution every N epochs regardless of plateau
        double attribution_epsilon     = config::ATTRIBUTION_DEFAULT_EPSILON;     // perturbation epsilon for blame analysis
        int   max_failures_to_fix      = config::DEFAULT_MAX_FAILURES_TO_FIX;     // top N blameworthy nodes to address per cycle
                                              // (must be >= #hypothesis types so the
                                              // NEURON_TANH_INJECTION fallback at
                                              // score 0.1 is always reachable)
        int   blackboard_max_candidates = config::DEFAULT_BLACKBOARD_MAX_CANDIDATES; // max Blackboard signals to check
        int   shadow_max_hypotheses    = 5;        // max shadow routines to try per cycle (currently unused; evolve() iterates candidates directly)
        double validation_threshold    = config::DEFAULT_VALIDATION_THRESHOLD;    // minimum improvement to commit (0 = any improvement)
        Graph::LossType loss_type      = Graph::LossType::MSE;

        // Compilation control
        bool  compile_enabled          = config::DEFAULT_COMPILE_ENABLED;          // Phase 6 on/off
        int   compile_interval         = config::DEFAULT_COMPILE_INTERVAL;         // compile every N evolution cycles

        // Determinism: 0 = non-deterministic (random_device); nonzero = every
        // RNG path (engine rng_, neuron weight-init via std::srand, attribution
        // seeds derived from rng_) is seeded from this value, so a run is
        // byte-for-byte reproducible. Default is a fixed nonzero value so the
        // engine is reproducible by default; --seed overrides.
        unsigned int seed              = config::DEFAULT_SEED;

        // Sequence mode: set when the dataset is a temporal sequence (--no-
        // shuffle). Enables the RECURRENT_SELF_WIRE candidate so the engine can
        // attempt memory-based targets (running parity, etc.). Ignored for
        // ordinary (shuffled) datasets.
        bool sequence_mode             = false;

        // Outer-loop early stopping (0 = run all max_epochs)
        int   early_stop_patience      = config::DEFAULT_EARLY_STOP_PATIENCE;

        // Structural cooldown after repeated failed structural cycles
        int   structural_failure_threshold = config::DEFAULT_STRUCTURAL_FAILURE_THRESHOLD;
        int   structural_cooldown       = config::DEFAULT_STRUCTURAL_COOLDOWN;

        // Periodic graph checkpointing: when non-empty, the best-val (or
        // best-train, if no validation data) snapshot is serialized to
        // this path every snapshot_interval epochs AND at evolve() exit.
        // Lets long runs (CIFAR, hours) survive crashes/reboots with the
        // best-so-far model on disk. Directory must exist.
        std::string snapshot_path;
        int   snapshot_interval          = 25;
    };

    // ---- Construction ----
    // Takes ownership of the graph and datasets. Builds an initial minimal
    // architecture if the graph is empty.
    EvolutionEngine(std::unique_ptr<Graph> graph,
                    Dataset training_data,
                    Dataset validation_data,
                    const Config& cfg);

    // ---- Main evolution loop ----
    // Runs evolution for up to cfg.max_epochs outer cycles.
    // Returns the final loss on the training dataset.
    // Calls progress_callback(epoch, loss, phase) after each major step if set.
    double evolve(std::function<void(int epoch, double loss, const std::string& phase)> progress_cb = {});

    // ---- Accessors ----
    const Graph& get_graph() const { return *graph_; }

    // Takes ownership — caller surrenders graph
    std::unique_ptr<Graph> take_graph() { return std::move(graph_); }

    // ---- Reporting ----
    struct EvolutionStats {
        int    total_epochs          = 0;
        int    sgd_epochs_run        = 0;
        int    structural_changes    = 0;
        int    successful_commits    = 0;
        int    failed_commits        = 0;
        int    compiles_performed    = 0;
        int    dead_nodes_removed    = 0;
        int    constants_folded      = 0;
        int    neurons_compressed    = 0;
        double initial_loss          = 0.0;
        double final_loss            = 0.0;
        double best_loss             = 0.0;
    };
    const EvolutionStats& get_stats() const { return stats_; }

    // Held-out test-set evaluation (--eval-csv): per-output MSE / R² / MAE.
    struct ExternalEval { double mse; double r2; double mae; };
    std::vector<ExternalEval> evaluate_external(const Dataset& data);

    // Softmax cross-entropy + top-1 accuracy over the FULL output set per
    // sample — the comparable metric for one-hot LM/classification tasks
    // (one-vs-rest BCE is not comparable to published softmax-CE floors).
    // Returns {mean_ce_nats, accuracy_fraction}; empty vector on error.
    std::vector<double> evaluate_external_softmax(const Dataset& data);

private:
    // Test harness access (tests/main_tests.cpp). Friend keeps the public API
    // clean while letting unit tests inspect the private reasoning pipeline
    // (blackboard, diagnose, complexity profile, candidate generation) — the
    // layer loss-only benchmarks cannot reach.
    friend class GpnnTestAccess;
    // ========================================================================
    // Phase 1: Forward Pass & Evaluation
    // ========================================================================
    // Execute the graph on the given dataset and return the average loss.
    double evaluate_loss(const Dataset& data);

    // ========================================================================
    // Phase 2: Micro-Evolution (SGD parameter tuning with plateau detection)
    // ========================================================================
    // Train NEURON weights and CONSTANT values.
    struct TrainResult { bool improved; double loss_before; double loss_after; };
    TrainResult train_parameters();

    // ========================================================================
    // Phase 3: Diagnosis & Target Propagation
    // ========================================================================
    struct FailureDiagnosis {
        uint64_t failing_node;                // node flagged as problematic
        Value    mean_blame;                  // average blame across samples
        bool     is_constant_output;          // node always produces same value
        bool     has_dead_branch;             // one of its IFELSE branches never fires

        // Mini-dataset: (local inputs, target output) pairs per sample
        // local_inputs[i] = map of upstream_node_id → output value for sample i
        // targets[i]      = value this node should have produced for sample i
        std::vector<std::unordered_map<uint64_t, Value>> local_inputs;
        std::vector<Value> targets;
    };

    // Run blame analysis and target propagation.
    // Returns list of failing nodes with computed targets, sorted by |blame|.
    std::vector<FailureDiagnosis> diagnose();

    // For a single node, compute the target output value per sample
    // that would minimize the final output error.
    void compute_targets(FailureDiagnosis& diag, uint64_t output_node_id);

    // ========================================================================
    // Phase 4: Pattern & Context Search (Blackboard)
    // ========================================================================
    struct BlackboardSignal {
        uint64_t node_id;
        Value    correlation;         // Pearson r with the target
        bool     is_input;            // true if this is an INPUT node
    };

    // Search the Blackboard (all INPUT nodes + stable internal node outputs)
    // for signals correlated with the failure target.
    std::vector<BlackboardSignal> search_blackboard(const FailureDiagnosis& diag);

    // ========================================================================
    // Phase 5: Reasoned Assembly (Shadow Routing)
    // ========================================================================

    // Failure type classification — guides which structural fix to try first.
    enum class FailureType { UNKNOWN, LINEAR_OFFSET, BOOLEAN_BOUNDARY, NON_LINEAR_CURVE };

    // Analyze the mini-dataset (local_inputs → targets) to classify the
    // nature of the failure: linear offset, boolean boundary, or non-linear curve.
    FailureType classify_failure(const FailureDiagnosis& diag) const;

    // ------------------------------------------------------------------------
    // ComplexityProfile — fingerprint of the residual at the bottleneck node.
    // ------------------------------------------------------------------------
    // Computed from the (INPUT features → target residual) mini-dataset stored
    // in FailureDiagnosis. Cheap (O(N·F²) for the polynomial fit). Used to:
    //   - Detect signatures like "bounded + high pairwise interaction" (sin(xy))
    //     that require compound templates (MULTIPLY then NEURON_TANH).
    //   - Replace brittle SCORE_*_BOOST constants with data-driven priors.
    //   - Rank hypotheses by template_match(profile, h.type) + simplicity.
    //
    // All fields are computed from INPUT features only (F = # INPUT nodes).
    // Internal-node features are ignored because the goal is to discover what
    // RAW FEATURE TRANSFORM is missing, not which existing internal signal to
    // rewire (that is the blackboard's job).
    struct ComplexityProfile {
        size_t num_inputs  = 0;
        size_t num_samples = 0;

        // --- Degree-2 polynomial fit (least-squares, Gaussian elimination) ---
        // Layout: [bias, x_1..x_F, x_1²..x_F², x_1·x_2, x_1·x_3, ..., x_{F-1}·x_F]
        // Indexing helpers below let callers pull specific terms back out.
        std::vector<double> poly_coeffs;
        double poly_r2 = 0.0;  // 1 - SS_res/SS_tot of the degree-2 fit
        // Tracking for the largest |coefficient| (excluding bias) — used by
        // compound-hypothesis detection to see if the dominant term is a
        // cross-term (x_i·x_j interaction) vs a linear term.
        size_t max_coef_index = 0;
        double max_coef_value = 0.0;

        // --- Sobol pairwise interaction index (correlation-robust) ---
        // For F≥2: max over input pairs of [V_int(x_a,x_b) / Var(r)]
        // where V_int = Var(E[r|xa,xb]) - Var(E[r|xa]) - Var(E[r|xb]).
        // High value ⇒ single-input explanations insufficient ⇒ MULTIPLY signature.
        double sobol_pairwise = 0.0;
        size_t sobol_pair_a   = 0;  // indices of the strongest pair
        size_t sobol_pair_b   = 0;

        // --- Lipschitz estimate per input axis ---
        // Sort samples by x_j, take max(|Δr|/|Δx_j|) between adjacent pairs.
        // Large value ⇒ sharp transition along axis j ⇒ IFELSE signature.
        std::vector<double> lipschitz_per_axis;
        double lipschitz_max = 0.0;

        // --- Residual stats ---
        double var_r      = 0.0;
        double mean_r     = 0.0;
        double min_r      = 0.0;
        double max_r      = 0.0;
        double bound_ratio = 0.0;  // (max-min)/stddev — small ⇒ bounded

        // --- Sign-quadrant means for first two inputs (F≥2) ---
        // Order: (++, +-, -+, --). Pattern like [+,-,-,+] ⇒ odd-symmetric product
        // (xy or sin(xy)). Pattern like [+,+,-,-] ⇒ step / IFELSE.
        std::vector<double> quadrant_means;

        // --- Verdict flags (set from the thresholds in constants.h) ---
        bool high_pairwise_interaction = false;
        bool bounded                   = false;
        bool sharp_boundary            = false;

        // Interaction-dominance from the polynomial fit (more robust than
        // Sobol on low-variance residuals). When true, interact_a/interact_b
        // name the input INDICES (among INPUT nodes) of the dominant feature:
        // a square term x_i^2 yields (i,i); a cross term x_i*x_j yields (i,j).
        // MULTIPLY_INJECTION uses these as its source pair when set.
        bool   interaction_dominant = false;
        size_t interact_a           = 0;
        size_t interact_b           = 0;
    };

    // Compute the complexity profile for one diagnosis. Returns an empty
    // profile (num_inputs=0) if INPUTs can't be identified or N is too small.
    ComplexityProfile compute_complexity_profile(const FailureDiagnosis& diag) const;

    // One-line human-readable summary for logging.
    std::string format_profile(const ComplexityProfile& prof) const;

    struct Hypothesis {
        enum Type {
            NONE,
            IFELSE_BOUNDARY_SPLIT,   // Insert IFELSE to isolate failing data
            NEURON_TANH_INJECTION,   // Append NEURON+TANH for non-linear mapping
            CONTEXT_WIRE,            // Cross-graph connection from Blackboard signal
            MULTIPLY_INJECTION,      // Add product feature (x*y or x^2) for interactions
            BOOLEAN_COMPOSE,         // Compose two boolean signals (XOR/AND/OR) for parity
            COMPOUND_MULTIPLY_NEURON,// MULTIPLY(src_a,src_b) → NEURON → TANH (sin(xy), x·y interactions with bounded output)
            COMPOUND_TANH_SERIES,    // K parallel NEURON→TANH chains summed via ADD (sin(x) over multiple periods)
            COMPOUND_MULTIPLY3_NEURON,// MULTIPLY(MULTIPLY(a,b),c) → NEURON → TANH (sin(xyz), 3-way interactions)
            COMPOUND_MULTIPLY_ABS,    // MULTIPLY(a,b) → ABS → NEURON(zero-init) → ADD (|x·y|, sign-symmetric interactions)
            RECURRENT_SELF_WIRE,     // NEURON += self-recurrent input (BPTT) for sequence/memory targets (d6)
            SIN_INJECTION,           // NEURON(zero-init) → SIN → ADD (sin(wx+b) for oscillating residuals)
            DEEP_INSERTION,          // failing_node → NEURON(zero-init) → TANH → ADD (residual depth for hierarchical features)
            RECURRENT_XOR,           // Recurrent XOR node for running parity (d6): output = input XOR prev_output
            MULTI_LAYER_STACK,       // K parallel hidden NEURONs + combining NEURON (2-layer MLP for spirals/checkerboard)
            PATCH_POOLING,           // patch_size² average-pool LINEAR nodes for image-like inputs (coarse conv prior)
            PARITY_TREE,             // linear-fold XOR tree over all binary inputs → OUTPUT (k-bit parity, XOR-5D)
            DIVIDE_INJECTION,        // quotient feature a/b (ratio targets: m/V, (u+v)/(1+uv)) — both orderings, denominator-safety-gated
            COMPOUND_SIN_PRODUCT,    // MULTIPLY(a,b) → NEURON(freq-init) → SIN → ADD (sin(x·y), x·sin(kx) targets: Korns F4/F8, Feynman I.29.16)
            COMPOUND_DIVIDE_PRODUCT, // DIVIDE(MULTIPLY(a,b), c) → zero-gain NEURON → ADD (q²a²/c³-class: Feynman I.32.8)
        };
        Type     type = NONE;

        // For IFELSE_BOUNDARY_SPLIT: the comparator details
        uint64_t condition_source_node;  // node providing the condition value
        Value    split_threshold;

        // For CONTEXT_WIRE: which Blackboard signal to connect
        uint64_t wire_source_node;

        // For MULTIPLY_INJECTION: the two sources to multiply.
        // If multiply_source_b == 0 or == multiply_source_a, becomes x^2 (self-product).
        uint64_t multiply_source_a = 0;
        uint64_t multiply_source_b = 0;

        // For BOOLEAN_COMPOSE: two sources + boolean op (XOR/AND/OR).
        // Continuous sources are thresholded via GREATER(src, bool_threshold)
        // before the boolean op, since XOR/AND/OR use != 0.0 truthiness.
        uint64_t  bool_source_a   = 0;
        uint64_t  bool_source_b   = 0;
        NodeType  bool_op         = NodeType::XOR;
        Value     bool_threshold  = 0.0;

        // For COMPOUND_TANH_SERIES: override the chain count K (0 = use
        // COMPOUND_TANH_SERIES_K default). Set adaptively by generate_candidates
        // from the residual's Lipschitz estimate (more zero-crossings -> more
        // chains) so high-frequency sin targets get more capacity.
        int       compound_K      = 0;

        // For SIN_INJECTION: initial frequency for the sin(wx+b) chain.
        // 0 = zero-init (legacy identity-start behavior). When nonzero,
        // estimated from residual zero-crossings: w = 2π·periods/x_range.
        // Growing w from 0 to the target frequency (~7.3 on d2) via SGD is
        // far slower than the validation budget; smart-init skips that phase.
        double    sin_freq_init   = 0.0;
    };

    // Form a hypothesis about how to fix the failure.
    Hypothesis form_hypothesis(const FailureDiagnosis& diag,
                               const std::vector<BlackboardSignal>& blackboard);

    // Generate a scored, ranked list of candidate hypotheses for the same
    // failure. Phase 7 will try them in order; the first to pass validation
    // is committed. If none pass, the failure is abandoned.
    // ftype guides scoring: the best strategy for this failure type ranks highest.
    // profile drives compound hypotheses (e.g. MULTIPLY+NEURON_TANH for sin(xy)).
    std::vector<Hypothesis> generate_candidates(const FailureDiagnosis& diag,
                                                const std::vector<BlackboardSignal>& blackboard,
                                                FailureType ftype,
                                                const ComplexityProfile& profile);

    // Apply the hypothesis to the graph in a non-destructive shadow form.
    // Returns a cloned graph with the modification applied.
    std::unique_ptr<Graph> apply_shadow_routing(const Hypothesis& hyp,
                                                 const FailureDiagnosis& diag);

    // ========================================================================
    // Phase 6: Graph Compilation (Distillation)
    // ========================================================================
    // Eliminate dead code, fold constants, compress neural chains.
    // Returns number of nodes removed.
    struct CompileResult {
        int dead_nodes_removed;
        int constants_folded;
        int neurons_compressed;
    };
    CompileResult compile();

    // ========================================================================
    // Phase 7: Validation & Commit / Revert
    // ========================================================================
    // Test a shadow-modified graph against the validation dataset.
    // If it improves loss, commit the changes; otherwise revert.
    // Returns true if committed.
    bool validate_and_commit(std::unique_ptr<Graph>& shadow_graph, double baseline_loss, const std::string& hyp_description);

    // Result of a shadow validation pass — used for parallel candidate
    // evaluation. Multiple shadows can be validated concurrently; the caller
    // then picks the lowest-val_loss acceptable shadow and commits it.
    struct ShadowValidationResult {
        bool                  acceptable = false;  // passed both sanity + commit gate
        double                train_loss = 0.0;    // shadow's loss on training_data_
        double                val_loss   = 0.0;    // shadow's loss on validation_data_
        std::unique_ptr<Graph> shadow;             // owned; moved into graph_ if winner
        int                   hyp_rank   = -1;     // candidate rank (lower = higher priority)
        int                   hyp_type   = 0;      // Hypothesis::Type as int (for logging)
        std::string           reject_reason;       // empty if acceptable
    };

    // Validate a single shadow WITHOUT mutating graph_. Returns a result
    // struct the caller can inspect. baseline_val is the current graph's
    // validation loss (precomputed once per epoch to avoid N redundant
    // recomputations when validating N candidates in parallel).
    ShadowValidationResult validate_shadow_only(std::unique_ptr<Graph>& shadow_graph,
                                                double baseline_loss,
                                                double baseline_val,
                                                int hyp_rank,
                                                int hyp_type);

    // Evaluate any graph on validation_data_ and return mean loss.
    // Read-only — does not mutate the input graph beyond temporary input
    // value sets during execution.
    double compute_validation_loss(Graph& g);

    // ========================================================================
    // Helpers
    // ========================================================================
    // Build a minimal initial architecture from the datasets if graph is empty.
    void ensure_minimal_architecture();

    // Pearson correlation coefficient between two same-length value vectors.
    static double pearson_correlation(const std::vector<Value>& a, const std::vector<Value>& b);

    // ========================================================================
    // Members
    // ========================================================================
    std::unique_ptr<Graph> graph_;
    Dataset training_data_;
    Dataset validation_data_;
    Config cfg_;
    EvolutionStats stats_;

    // Blackboard registry: node_id → output values across all training samples
    // Rebuilt each evolution cycle after Phase 1.
    std::unordered_map<uint64_t, std::vector<Value>> blackboard_registry_;

    // Mapping from data-space sample key to graph-space node ID.
    // key = the uint64_t used in SampleIODesc.inputs / .targets maps;
    // value = the graph's auto-assigned node ID.
    std::unordered_map<uint64_t, uint64_t> input_data_to_graph_;
    std::unordered_map<uint64_t, uint64_t> output_data_to_graph_;

    // RNG — seeded from cfg_.seed in the constructor (deterministic when seed
    // != 0). Default-init here; reseeded before any use.
    std::mt19937 rng_{std::random_device{}()};

    // Track last best loss for plateau detection
    double best_phase2_loss_ = config::LOSS_SENTINEL_INF;
    int    plateau_counter_  = 0;
    int    epochs_since_structural_ = 0;
    double best_overall_loss_ = config::LOSS_SENTINEL_INF;

    // Validation-based model selection: the deployed snapshot tracks the
    // best VALIDATION loss, not train. Train-based selection deploys the
    // most-overfit epoch on high-dimensional tasks (observed: CIFAR gray
    // train BCE 1.9 but val 5.1 — worse than the 3.25 base-rate constant).
    // best_overall_loss_ (train) still drives early stopping + divergence.
    double best_val_loss_ = config::LOSS_SENTINEL_INF;
    double best_val_snapshot_train_loss_ = 0.0;

    // Outer-loop early stopping: epochs since best_overall_loss_ last improved
    int    epochs_since_best_ = 0;

    // Structural cooldown: when consecutive_structural_failures_ hits the
    // threshold, structural_cooldown_ is set and the structural block is
    // skipped for that many epochs (gives SGD time to settle).
    int    consecutive_structural_failures_ = 0;
    int    structural_cooldown_             = 0;
    int    divergence_counter_              = 0;

    // Degenerate-loop detection: tracks consecutive same-type commits that
    // don't improve best_overall_loss_. After 3 such, that hypothesis type
    // is suppressed (added to suppressed_hyp_types_) for the rest of the run.
    int    last_committed_hyp_type_         = -1;
    int    consecutive_ineffective_commits_  = 0;
    std::unordered_set<int> suppressed_hyp_types_;

    // Snapshot of the graph at the moment of best_overall_loss_.
    // SGD with momentum can overshoot a good minimum; this lets evolve()
    // restore the best-known state at the end instead of returning an
    // overshoot. Updated whenever best_overall_loss_ improves.
    std::unique_ptr<Graph> best_graph_snapshot_;

    // Subgraph library (behavioral prior for candidate scoring). Non-owning.
    const SubgraphLibrary* library_ = nullptr;
};

// ============================================================================
// Phase 6 implementation methods — added to Graph for compilation
// ============================================================================

// Remove nodes that don't contribute to any OUTPUT (dead code).
// Returns number of nodes removed.
int graph_eliminate_dead_code(Graph& graph);

// Fold compile-time-known computations: collapse chains of pure CONSTANT ops
// into a single CONSTANT. Returns number of nodes folded.
int graph_fold_constants(Graph& graph);

// Collapse sequential NEURON nodes without activation functions between
// them into a single NEURON via matrix multiply. Returns number of pairs merged.
int graph_compress_neurons(Graph& graph);

} // namespace gpnn