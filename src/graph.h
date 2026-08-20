#pragma once

#include "node.h"
#include "constants.h"
#include <vector>
#include <unordered_map>
#include <set>
#include <memory>
#include <string>

namespace aria {

// ============================================================================
// Connection — links one output port to one input port
// ============================================================================
constexpr size_t MAX_DELAY_TAPS = 4;   // ring-buffer depth (k=1..4 BPTT)

struct Connection {
    uint64_t src_node;
    size_t   src_port;   // output port index on source node
    uint64_t dst_node;
    size_t   dst_port;   // input port on target node
    bool     is_recurrent = false;   // auto-flagged when connection forms a cycle
    // Multi-tap delay ring: history[0] = previous timestep (k=1, the
    // classic delay_buffer), history[1] = two steps back, etc.
    // delay_taps = k (how far back this connection reads). Enables
    // multi-step memory without deep unrolling (NARMA-30).
    int      delay_taps = 0;         // 0 → legacy single-step behavior
    Value    delay_buffer = 0.0;     // k=1 value (kept for serialization compat)
    std::array<Value, MAX_DELAY_TAPS> history{};   // ring history
};

// ============================================================================
// Graph 鈥?container of nodes and connections
// ============================================================================
class Graph {
public:
    Graph();
    ~Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    // Deep copy 鈥?creates an independent clone in memory (no serialization overhead)
    std::unique_ptr<Graph> clone() const;

    // ---- Node management ----
    // Create a node of the given type. Returns the assigned node ID.
    uint64_t add_node(NodeType type, const std::string& name = "");

    // Remove a node and all connections referencing it.
    void remove_node(uint64_t node_id);

    // Lookup a node by ID.
    Node* get_node(uint64_t node_id);
    const Node* get_node(uint64_t node_id) const;

    // ---- Connection management ----
    // Add a connection. Returns false if ports or nodes don't exist.
    // For Neuron targets, auto-resizes the input count if needed.
    bool add_connection(uint64_t src_node, size_t src_port,
                        uint64_t dst_node, size_t dst_port);

    // Remove a specific connection.
    void remove_connection(uint64_t src_node, size_t src_port,
                           uint64_t dst_node, size_t dst_port);

    const std::vector<Connection>& get_connections() const { return connections_; }

    // Set the delay depth (taps) of an existing recurrent connection.
    // Returns false if the connection doesn't exist. Used by RECURRENT_
    // MULTI_TAP routing after add_connection flags the self-edge recurrent.
    bool set_connection_delay_taps(uint64_t src, size_t sp,
                                   uint64_t dst, size_t dp, int taps);

    // ---- Execution ----
    // Single-pass wave propagation. Input/Constant values must be set beforehand.
    // After execution, read Output nodes' values.
    void execute();

    // Reset all recurrent connection delay buffers to zero.
    // Call before starting a new sequence of graph executions.
    void reset_recurrent_state();
    bool has_recurrent_connections() const;

    // Set the output value of an Input or Constant node.
    void set_input_value(uint64_t node_id, Value value);

    // Preload an output value onto any node (used for recurrence/state carry-forward).
    // Marks the node clean so it survives the per-execution reset and acts as a
    // pseudo-source in the next execute() call.
    void set_node_output(uint64_t node_id, size_t port, Value val);

    // Get the value held by an Output node.
    Value get_output_value(uint64_t node_id) const;

    // Get the output of any node (returns 0.0 if node doesn't exist or has no outputs).
    // Call execute() first. Useful for fitness functions to inspect internal sub-computations.
    Value get_any_node_output(uint64_t node_id) const;

    // ---- Access ----
    const std::vector<std::unique_ptr<Node>>& get_nodes() const { return nodes_; }
    size_t node_count() const { return nodes_.size(); }

    // ---- Validation ----
    // Check that all connections reference valid nodes/ports.
    bool validate(std::string& error) const;

    // ---- Importance / gradient analysis ----
    // Runs a backward pass (backpropagation-inspired) to compute how sensitive
    // the given output node's value is to each node in the graph (鈭俹ut/鈭俷ode).
    // Returns a map of node_id 鈫?gradient. The output node itself gets grad=1.0.
    // Call execute() first so that the graph has valid input/output values.
    std::unordered_map<uint64_t, Value> compute_importance(uint64_t output_node_id);

    // ---- Structural hashing (for duplicate detection) ----
    // Returns a hash of the graph's structural topology + value-carrying node state.
    // Two graphs that compute identically for all inputs will have the same hash.
    size_t compute_structural_hash() const;

    // ---- Error attribution (perturbation-based blame analysis) ----
    // For each node with outputs, perturb its output by 蔚, re-execute, and measure
    // how the error changes. Works on ALL node types including logic/comparison gates.
    // blame > 0 鈫?node hurts output (perturbing increases error: node is helping)
    // blame < 0 鈫?node harms output (perturbing decreases error: node is the problem)
    // Results are sorted by |blame| descending.
    struct SampleIODesc {
        std::unordered_map<uint64_t, Value> inputs;   // INPUT node_id 鈫?value
        std::unordered_map<uint64_t, Value> targets;  // OUTPUT node_id 鈫?expected value
    };

    // ---- Loss type for train() ----
    // MSE: raw outputs, grad = 2*(pred-target)
    // BCE: sigmoid outputs, grad = sig-target (one-vs-rest, independent)
    // SOFTMAX_CE: raw logits stored; grad = softmax(z)_c - y_c over ALL
    //   OUTPUT nodes jointly (outputs compete — the coupling that lets
    //   bigram/context statistics become learnable; charLM fix).
    enum class LossType { BCE, MSE, SOFTMAX_CE };

    // ---- Weight training (SGD) ----
    struct TrainConfig {
        int epochs;
        double learning_rate;
        double gradient_clip;
        double momentum;
        double weight_decay;
        LossType loss_type;
        // Adam optimizer parameters (replaces momentum SGD)
        bool   use_adam = true;
        double adam_beta1 = 0.9;
        double adam_beta2 = 0.999;
        double adam_eps    = 1e-8;
        // Within-SGD early stopping: if the training loss hasn't improved by
        // this fraction in early_stop_patience consecutive epochs, stop the
        // SGD phase early (prevents overfitting on noisy data like d9).
        // Set patience to 0 to disable.
        int    early_stop_patience = 5;
        double early_stop_min_improvement = 1e-4;
        // Mini-batch size (0 = full-batch). When >0, each SGD epoch iterates
        // over mini-batches of this size, giving ns/batch_size gradient steps
        // per epoch instead of 1. Critical for high-dimensional tasks (MNIST)
        // where full-batch GD is too slow to converge.
        int    batch_size = 0;
        // Wall-clock watchdog (seconds; 0 = off). Training aborts at epoch
        // boundaries once exceeded. Guards shadow validations against
        // live-locks (observed: pooled-CIFAR shadow spun 40+ min on one
        // thread with no progress). Shadow callers set this; the main
        // loop leaves it off.
        int    watchdog_seconds = 0;
        std::unordered_map<uint64_t, uint64_t> input_data_to_graph;
        std::unordered_map<uint64_t, uint64_t> output_data_to_graph;
        TrainConfig()
            : epochs(config::DEFAULT_TRAIN_EPOCHS),
              learning_rate(config::DEFAULT_TRAIN_LEARNING_RATE),
              gradient_clip(config::DEFAULT_TRAIN_GRADIENT_CLIP),
              momentum(config::DEFAULT_TRAIN_MOMENTUM),
              weight_decay(config::DEFAULT_TRAIN_WEIGHT_DECAY),
              loss_type(LossType::BCE) {}
    };

    // Train NEURON weights/biases and CONSTANT values via gradient descent
    // on the given samples. Uses the existing backward_input_grads()
    // infrastructure to compute 鈭俵oss/鈭俻aram for each trainable parameter,
    // then applies SGD updates in-place.
    // Call this inside the fitness function before computing the error score.
    void train(const std::vector<SampleIODesc>& samples,
               const TrainConfig& cfg = TrainConfig{});

    struct ErrorAttributionResult {
        uint64_t node_id;
        Value base_error;       // MSE without perturbation
        Value perturbed_error;  // MSE with perturbation
        Value blame;            // perturbed - base (positive = node is helping)
    };

    std::vector<ErrorAttributionResult> compute_error_attribution(
        const std::vector<SampleIODesc>& samples,
        Value epsilon = config::ATTRIBUTION_DEFAULT_EPSILON,
        int max_candidates = 0,
        unsigned int seed = 0,
        const std::unordered_map<uint64_t, uint64_t>& input_data_to_graph = {},
        const std::unordered_map<uint64_t, uint64_t>& output_data_to_graph = {});

    // ---- Node-type replacement search (uses error attribution) ----
    // For each of the top_k most blameworthy nodes (blame < 0), searches
    // for graph modifications that reduce MSE on the given samples:
    //   REPLACE 鈥?swap node with a different non-NN type (variable input counts
    //              handled via random subsetting / INPUT-node filling).
    //   INSERT  鈥?insert a new non-NN node that intercepts a connection near
    //              the blameworthy node (pre- or post-processing).
    //   DELETE  鈥?remove the node and bypass it (cartesian product of
    //              incoming 脳 outgoing connections).
    //   REPLACE_INSERT / DELETE_INSERT 鈥?two mutations in one trial.
    // Returns results sorted by improvement descending.
    enum class SearchOperation {
        REPLACE,
        INSERT,
        REWIRE_INPUT,        // try alternative source node for an existing input
        DELETE_BYPASS,
        REPLACE_INSERT,
        DELETE_INSERT,
    };

    struct NodeImprovementResult {
        uint64_t node_id;          // the blameworthy node
        SearchOperation operation;
        NodeType original_type;
        NodeType new_type;         // replacement / inserted type (for DELETE_BYPASS: sentinel)
        Value original_error;      // baseline MSE
        Value new_error;           // MSE after modification
        Value improvement;         // original - new (positive = better)
    };

    std::vector<NodeImprovementResult> search_improvements(
        const std::vector<SampleIODesc>& samples,
        const std::vector<ErrorAttributionResult>& attribution,
        int top_k = config::SEARCH_IMPROVEMENT_DEFAULT_TOP_K,
        bool non_nn_only = true,
        std::unique_ptr<Graph>* best_trial = nullptr,
        unsigned int seed = 0);

    // ---- Constant output detection (for targeted mutation) ----
    // Check if a vector of sigmoid output values represents a constant/degenerate output.
    // Returns true if all values are on the same side of 0.5 OR variance < threshold.
    static bool is_output_constant(const std::vector<double>& sigmoid_outputs);

    // Mark which output nodes are currently producing constant outputs.
    // Called by the fitness function after evaluating all samples.
    void set_constant_outputs(const std::vector<uint64_t>& outputs);
    const std::set<uint64_t>& get_constant_outputs() const { return constant_outputs_; }

    // Find all nodes that feed into the given node (ancestors via backward DFS
    // on the reverse adjacency). Includes the node itself.
    std::vector<uint64_t> get_ancestors(uint64_t node_id) const;

    // ---- Subgraph extraction (Building Block Library) ----
    // Extract the subgraph that feeds into a specific output node.
    // Returns a new Graph containing all ancestor nodes and the connections
    // between them. INPUT and OUTPUT nodes from the source are copied.
    // Also returns the set of INPUT node IDs the subgraph references.
    std::unique_ptr<Graph> extract_subgraph(uint64_t output_node_id,
                                           std::vector<uint64_t>& out_input_ids) const;

    // ---- Size diagnostics ----
    // Returns IDs of all nodes reachable from any OUTPUT node (backward BFS).
    // Includes INPUT, OUTPUT, and internal nodes that contribute to computation.
    std::vector<uint64_t> compute_reachable_ids() const;

    // Fraction of all nodes that contribute to at least one output.
    // 1.0 = fully connected, < 0.5 = significant dead code.
    double compute_utilization_ratio() const;

    // Fraction of internal nodes (non-INPUT, non-OUTPUT) whose ALL input ports
    // have incoming connections. 0.0 = every node has spare ports, 1.0 = fully
    // saturated 鈥?graph may need to grow to add new connection points.
    double compute_saturation_ratio() const;

    // ---- Connectivity diagnostics (for empty-seed / cold-start detection) ----
    // Returns true if every OUTPUT node has at least one forward path from an
    // INPUT or CONSTANT node. If false, the graph cannot compute anything useful
    // because its outputs have no data dependencies. Call before training to
    // skip expensive SGD on dead graphs.
    bool has_input_to_output_path() const;

    // Count how many OUTPUT nodes have a forward path from any INPUT or CONSTANT.
    // Useful for partial connectivity detection: if 1/2 outputs are reachable,
    // the graph can still be partially useful.
    int count_reachable_outputs() const;

    // ---- Serialization helpers (used by serialize.cpp) ----
    uint64_t get_next_id() const { return next_id_; }
    void set_next_id(uint64_t id) { next_id_ = id; }
    void clear();
    void add_node_existing(std::unique_ptr<Node> node);
    void add_connection_existing(Connection conn);

    // ---- Symbolic expression (graph 鈫?readable formula) ----
    // Converts the subgraph feeding the given OUTPUT node into a human-readable
    // math expression using arithmetic operators, function calls (tanh, sin,
    // etc.), and ternaries for logic nodes. INPUT nodes become x0, x1, ...;
    // CONSTANT nodes become literals; NEURON becomes tanh(w*expr + ... + b).
    // Used by the subgraph library for precise matching and by post-evolution
    // reporting for interpretability.
    std::string to_expression(uint64_t output_node_id) const;

private:
    size_t find_node_index(uint64_t id) const;
    static constexpr size_t INVALID_INDEX = config::INDEX_INVALID;

    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<Connection> connections_;
    uint64_t next_id_ = config::DEFAULT_GRAPH_NODE_ID_START;

    // ---- Cached indices (lazy-rebuilt when caches_dirty_, invalidated on mutation) ----
    // src_adj_[src_node_id]    = connections originating from source node
    // rev_adj_[dst_node_id]    = connections targeting destination node
    // node_idx_[node_id]       = index into nodes_ vector  (replaces O(N) find_node_index)
    mutable std::unordered_map<uint64_t, std::vector<const Connection*>> src_adj_;
    mutable std::unordered_map<uint64_t, std::vector<const Connection*>> rev_adj_;
    mutable std::unordered_map<uint64_t, size_t> node_idx_;
    mutable bool caches_dirty_ = true;

    // Structural hash cache 鈥?invalidated on any structural mutation
    mutable bool hash_dirty_ = true;
    mutable size_t cached_hash_ = 0;

    // Rebuild all cached indices (idempotent, called at start of execute/train/etc.)
    void ensure_caches() const;

    // Mark caches dirty. Call after any structural mutation.
    void invalidate_caches();

    // ---- Incremental execution (dirty-flag propagation) ----
    // Mark node_id and all downstream nodes as dirty (BFS through src_adj_).
    void mark_transitively_dirty(uint64_t node_id);

    // Mark every node in the graph as dirty.
    void set_all_dirty();

    std::set<uint64_t> constant_outputs_;
};

} // namespace aria
