#include "graph.h"
#include "serialize.h"
#include "logger.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <stack>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <thread>
#include <set>
#include <cstdio>
#include <unordered_set>

namespace aria {

// ============================================================================
// Graph constructor/destructor
// ============================================================================
Graph::Graph() {
    // Ensure built-in types are registered
    static bool once = []() {
        register_builtin_node_types();
        return true;
    }();
    (void)once;
}

Graph::~Graph() = default;

// ============================================================================
// Internal helpers
// ============================================================================
size_t Graph::find_node_index(uint64_t id) const {
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i]->get_id() == id) return i;
    }
    return INVALID_INDEX;
}

void Graph::invalidate_caches() {
    caches_dirty_ = true;
    hash_dirty_ = true;
}

void Graph::ensure_caches() const {
    if (!caches_dirty_) return;
    caches_dirty_ = false;

    src_adj_.clear();
    rev_adj_.clear();
    node_idx_.clear();

    for (size_t i = 0; i < nodes_.size(); ++i) {
        node_idx_[nodes_[i]->get_id()] = i;
    }

    for (const auto& conn : connections_) {
        src_adj_[conn.src_node].push_back(&conn);
        rev_adj_[conn.dst_node].push_back(&conn);
    }
}

// ---- Incremental execution (dirty-flag propagation) ----

void Graph::mark_transitively_dirty(uint64_t node_id) {
    ensure_caches();
    std::queue<uint64_t> q;
    q.push(node_id);
    while (!q.empty()) {
        uint64_t cur = q.front(); q.pop();
        Node* n = get_node(cur);
        if (!n) continue;
        if (!n->is_dirty()) {
            n->mark_dirty();
            auto adj_it = src_adj_.find(cur);
            if (adj_it != src_adj_.end()) {
                for (const auto* conn : adj_it->second) {
                    q.push(conn->dst_node);
                }
            }
        }
    }
}

void Graph::set_all_dirty() {
    for (auto& n : nodes_) {
        n->mark_dirty();
    }
}

// ============================================================================
// Node management
// ============================================================================
uint64_t Graph::add_node(NodeType type, const std::string& name) {
    uint64_t id = next_id_++;
    std::string node_name = name.empty() ?
        (std::string(node_type_to_string(type)) + "_" + std::to_string(id)) : name;
    nodes_.push_back(NodeRegistry::instance().create(type, id, node_name));
    invalidate_caches();
    // New isolated node has no connections — no existing computation paths affected.
    // The new node starts dirty (default) and will be computed when connected later.
    return id;
}

void Graph::remove_node(uint64_t node_id) {
    // Collect downstream nodes before removing connections (needed for targeted dirty marking).
    // Only nodes that received input from the removed node are affected.
    std::set<uint64_t> downstream;
    for (const auto& c : connections_) {
        if (c.src_node == node_id) {
            downstream.insert(c.dst_node);
        }
    }

    // Remove connections referencing this node
    connections_.erase(
        std::remove_if(connections_.begin(), connections_.end(),
            [node_id](const Connection& c) {
                return c.src_node == node_id || c.dst_node == node_id;
            }),
        connections_.end());

    // Remove the node itself
    size_t idx = find_node_index(node_id);
    if (idx != INVALID_INDEX) {
        nodes_.erase(nodes_.begin() + static_cast<ptrdiff_t>(idx));
    }
    invalidate_caches();

    // Only mark downstream nodes dirty — the rest of the graph is unaffected
    for (uint64_t dst_id : downstream) {
        mark_transitively_dirty(dst_id);
    }
}

Node* Graph::get_node(uint64_t node_id) {
    // O(1) via cached index map (rebuilt lazily on mutation).
    // Critical for high-dim graphs: set_input_value alone does ~F lookups
    // per sample (F=1024 on CIFAR), so the old O(N) scan dominated runtime.
    ensure_caches();
    auto it = node_idx_.find(node_id);
    return (it != node_idx_.end()) ? nodes_[it->second].get() : nullptr;
}

const Node* Graph::get_node(uint64_t node_id) const {
    ensure_caches();
    auto it = node_idx_.find(node_id);
    return (it != node_idx_.end()) ? nodes_[it->second].get() : nullptr;
}

// ============================================================================
// Connection management
// ============================================================================
bool Graph::add_connection(uint64_t src_node, size_t src_port,
                           uint64_t dst_node, size_t dst_port) {
    Node* src = get_node(src_node);
    Node* dst = get_node(dst_node);
    if (!src || !dst) return false;
    if (src_port >= src->get_num_outputs()) return false;

    // For Neuron/Linear targets, auto-expand inputs
    if (dst->get_type() == NodeType::NEURON
        || dst->get_type() == NodeType::LINEAR) {
        auto* neuron = dynamic_cast<NeuronNode*>(dst);
        if (dst_port >= dst->get_num_inputs()) {
            neuron->set_input_count(dst_port + 1);
        }
    } else {
        if (dst_port >= dst->get_num_inputs()) return false;
    }

    // Remove any existing connection to the same destination port.
    // Each input port should have exactly one source in a computation graph.
    connections_.erase(
        std::remove_if(connections_.begin(), connections_.end(),
            [&](const Connection& c) {
                return c.dst_node == dst_node && c.dst_port == dst_port;
            }),
        connections_.end());

    // Detect if this connection forms a cycle (dst_node feeds src_node),
    // auto-flagging it as recurrent with implicit delay-buffer state.
    invalidate_caches();                // flush caches because we may have erased above
    ensure_caches();                    // rebuild from current connections_ state
    auto ancestors = get_ancestors(src_node);
    bool is_recurrent = std::find(ancestors.begin(), ancestors.end(), dst_node) != ancestors.end();

    Connection nc;
    nc.src_node = src_node; nc.src_port = src_port;
    nc.dst_node = dst_node; nc.dst_port = dst_port;
    nc.is_recurrent = is_recurrent;
    connections_.push_back(nc);
    dst->mark_input_connected(dst_port);
    invalidate_caches();
    // Only the new connection's downstream is affected — mark dst_node transitively dirty
    mark_transitively_dirty(dst_node);
    return true;
}

bool Graph::set_connection_delay_taps(uint64_t src, size_t sp,
                                      uint64_t dst, size_t dp, int taps) {
    for (auto& c : connections_) {
        if (c.src_node == src && c.src_port == sp
            && c.dst_node == dst && c.dst_port == dp) {
            c.delay_taps = taps;
            return true;
        }
    }
    return false;
}

void Graph::remove_connection(uint64_t src_node, size_t src_port,
                              uint64_t dst_node, size_t dst_port) {
    connections_.erase(
        std::remove_if(connections_.begin(), connections_.end(),
            [&](const Connection& c) {
                return c.src_node == src_node && c.src_port == src_port &&
                       c.dst_node == dst_node && c.dst_port == dst_port;
            }),
        connections_.end());
    // Clear structural connection flag on the destination node's input port
    Node* dst = get_node(dst_node);
    if (dst && dst_port < dst->get_num_inputs()) {
        dst->clear_input_connected(dst_port);
    }
    invalidate_caches();
    // Only the removed connection's downstream path is affected
    mark_transitively_dirty(dst_node);
}

// ============================================================================
// Execution
// ============================================================================
void Graph::execute() {
    ensure_caches();

    // Only reset dirty nodes -- clean nodes keep their cached outputs
    for (auto& n : nodes_) {
        if (n->is_dirty()) {
            n->reset();
        }
    }

    const size_t N = nodes_.size();
    std::vector<bool> executed(N, false);
    std::queue<size_t> ready;

    // Helper: deliver a source node's output values to all downstream inputs.
    // Clean downstream: relay cached outputs (mark executed + push to ready).
    // Dirty downstream: set input; if all inputs arrive, execute and clear dirty.
    // Dirty node missing inputs: wait for remaining upstreams (neither branch runs).
    auto propagate = [&](size_t src_idx) {
        auto& src = nodes_[src_idx];
        uint64_t src_id = src->get_id();
        auto adj_it = src_adj_.find(src_id);
        if (adj_it == src_adj_.end()) return;
        for (size_t out = 0; out < src->get_num_outputs(); ++out) {
            Value val = src->get_output(out);
            for (const auto* conn : adj_it->second) {
                if (conn->src_port != out) continue;
                auto nit = node_idx_.find(conn->dst_node);
                if (nit == node_idx_.end()) continue;
                size_t dst_idx = nit->second;
                if (executed[dst_idx]) continue;
                auto& dst = nodes_[dst_idx];
                Value input_val = conn->is_recurrent
                    ? ((conn->delay_taps > 1
                        && static_cast<size_t>(conn->delay_taps - 1) < MAX_DELAY_TAPS)
                       ? conn->history[conn->delay_taps - 1]
                       : conn->delay_buffer)
                    : val;
                dst->set_input(conn->dst_port, input_val);
                if (dst->is_dirty() && dst->has_all_inputs()) {
                    dst->execute();
                    if (std::abs(dst->get_perturbation()) >= config::GRADIENT_ZERO_THRESHOLD) {
                        Value p = dst->get_perturbation();
                        for (size_t i = 0; i < dst->get_num_outputs(); ++i)
                            dst->set_output(i, dst->get_output(i) + p);
                    }
                    dst->clear_dirty();
                    executed[dst_idx] = true;
                    ready.push(dst_idx);
                } else if (!dst->is_dirty() && !executed[dst_idx]) {
                    // Clean node -- relay cached outputs downstream without re-executing
                    executed[dst_idx] = true;
                    ready.push(dst_idx);
                }
                // Dirty node without all inputs: wait for remaining upstreams
            }
        }
    };

    // 0. Pre-deliver recurrent delay buffers to destination ports.
    //    Multi-tap: read history[delay_taps-1] when k>1 (k=1 → history[0]
    //    == delay_buffer, unchanged behavior).
    for (const auto& conn : connections_) {
        if (conn.is_recurrent) {
            Node* dst = get_node(conn.dst_node);
            if (dst) {
                Value v = (conn.delay_taps > 1
                           && static_cast<size_t>(conn.delay_taps - 1) < MAX_DELAY_TAPS)
                          ? conn.history[conn.delay_taps - 1]
                          : conn.delay_buffer;
                dst->set_input(conn.dst_port, v);
            }
        }
    }

    // 1. Seed ready queue: source nodes (INPUT/CONSTANT)
    //    Execute dirty sources; clean sources propagate cached values.
    //    Clean non-source nodes: mark executed so wavefront can reach dirty downstream.
    for (size_t i = 0; i < N; ++i) {
        auto& n = nodes_[i];
        if (n->get_type() == NodeType::INPUT || n->get_type() == NodeType::CONSTANT) {
            if (n->is_dirty()) {
                n->execute();
                if (std::abs(n->get_perturbation()) >= config::GRADIENT_ZERO_THRESHOLD) {
                    Value p = n->get_perturbation();
                    for (size_t i2 = 0; i2 < n->get_num_outputs(); ++i2)
                        n->set_output(i2, n->get_output(i2) + p);
                }
                n->clear_dirty();
            }
            // Source nodes always ready (cached or just-computed output)
            executed[i] = true;
            ready.push(i);
        } else if (!n->is_dirty() && n->get_num_outputs() > 0) {
            // Clean non-source node with outgoing connections: preloaded via set_node_output().
            // Push into ready queue so its output propagates downstream as a pseudo-source.
            executed[i] = true;
            ready.push(i);
        }
        // Dirty non-source node: leave as not-executed, wait for upstream propagation
    }

    // 2. Wavefront propagation
    while (!ready.empty()) {
        size_t idx = ready.front();
        ready.pop();
        propagate(idx);
    }

    // 3. Handle cycles / unresolved dirty nodes -- zero-fill missing inputs and force execute
    //    ABSENT nodes: skip zero-fill so they can detect that input was never connected
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < N; ++i) {
            if (executed[i]) continue;
            auto& n = nodes_[i];
            if (!n->is_dirty()) {
                executed[i] = true;
                continue;
            }
            if (n->get_type() != NodeType::ABSENT) {
                for (size_t in = 0; in < n->get_num_inputs(); ++in) {
                    if (!n->is_input_filled(in)) {
                        n->set_input(in, config::ZERO_FILL_VALUE);
                    }
                }
            }
            n->execute();
            if (std::abs(n->get_perturbation()) >= config::GRADIENT_ZERO_THRESHOLD) {
                Value p = n->get_perturbation();
                for (size_t i2 = 0; i2 < n->get_num_outputs(); ++i2)
                    n->set_output(i2, n->get_output(i2) + p);
            }
            n->clear_dirty();
            executed[i] = true;
            propagate(i);
            changed = true;
        }
    }

    // Save current outputs into recurrent connection history rings
    // for future timesteps (multi-tap BPTT). history[0]=t-1 (also mirrored
    // to delay_buffer for serialization compat), history[1]=t-2, ...
    for (auto& conn : connections_) {
        if (conn.is_recurrent) {
            Node* node = get_node(conn.src_node);
            if (node && conn.src_port < node->get_num_outputs()) {
                Value v = node->get_output(conn.src_port);
                // Shift ring: [h3,h2,h1,h0] <- [h2,h1,h0,v]
                for (size_t i = MAX_DELAY_TAPS - 1; i > 0; --i) {
                    conn.history[i] = conn.history[i - 1];
                }
                conn.history[0] = v;
                conn.delay_buffer = v;
            }
        }
    }

    // Clear any analysis perturbations so they don't leak into subsequent
    // execution passes (e.g. after compute_error_attribution).
    for (auto& n : nodes_) n->set_perturbation(0.0);
}

void Graph::reset_recurrent_state() {
    for (auto& conn : connections_) {
        if (conn.is_recurrent) {
            conn.delay_buffer = Value{0.0};
            conn.history.fill(Value{0.0});
        }
    }
}

bool Graph::has_recurrent_connections() const {
    for (const auto& conn : connections_) {
        if (conn.is_recurrent) return true;
    }
    return false;
}

// ============================================================================
// I/O helpers
// ============================================================================
void Graph::set_input_value(uint64_t node_id, Value value) {
    Node* n = get_node(node_id);
    if (!n) throw std::runtime_error("Graph::set_input_value: node not found");

    if (n->get_type() == NodeType::INPUT) {
        dynamic_cast<InputNode*>(n)->set_value(value);
    } else if (n->get_type() == NodeType::CONSTANT) {
        dynamic_cast<ConstantNode*>(n)->set_value(value);
    } else {
        throw std::runtime_error("Graph::set_input_value: node is not Input or Constant");
    }
    mark_transitively_dirty(node_id);
}

Value Graph::get_output_value(uint64_t node_id) const {
    const Node* n = get_node(node_id);
    if (!n) throw std::runtime_error("Graph::get_output_value: node not found");
    if (n->get_type() != NodeType::OUTPUT) {
        throw std::runtime_error("Graph::get_output_value: node is not Output");
    }
    return dynamic_cast<const OutputNode*>(n)->get_value();
}

Value Graph::get_any_node_output(uint64_t node_id) const {
    const Node* n = get_node(node_id);
    if (!n || n->get_num_outputs() == 0) return 0.0;
    return n->get_output(0);
}

void Graph::set_node_output(uint64_t node_id, size_t port, Value val) {
    Node* n = get_node(node_id);
    if (!n || port >= n->get_num_outputs()) return;
    // Mark all downstream nodes dirty BEFORE setting the new output so they
    // re-execute with the updated value.  The node itself stays clean so it
    // survives the per-execution reset as a pseudo-source.
    mark_transitively_dirty(node_id);
    n->set_output(port, val);
    n->clear_dirty();
}

// ============================================================================
// Validation
// ============================================================================
bool Graph::validate(std::string& error) const {
    std::ostringstream oss;
    bool ok = true;

    for (const auto& conn : connections_) {
        const Node* src = get_node(conn.src_node);
        const Node* dst = get_node(conn.dst_node);
        if (!src) {
            oss << "Connection references missing source node " << conn.src_node << "\n";
            ok = false;
        } else if (conn.src_port >= src->get_num_outputs()) {
            oss << "Connection src_port " << conn.src_port
                << " out of range for node " << conn.src_node
                << " (has " << src->get_num_outputs() << " outputs)\n";
            ok = false;
        }
        if (!dst) {
            oss << "Connection references missing target node " << conn.dst_node << "\n";
            ok = false;
        } else if (conn.dst_port >= dst->get_num_inputs()) {
            oss << "Connection dst_port " << conn.dst_port
                << " out of range for node " << conn.dst_node
                << " (has " << dst->get_num_inputs() << " inputs)\n";
            ok = false;
        }
    }

    error = oss.str();
    return ok;
}

// ============================================================================
// Serialization helpers
// ============================================================================
void Graph::clear() {
    nodes_.clear();
    connections_.clear();
    next_id_ = config::DEFAULT_GRAPH_NODE_ID_START;
    invalidate_caches();
}

void Graph::add_node_existing(std::unique_ptr<Node> node) {
    if (node->get_id() >= next_id_) {
        next_id_ = node->get_id() + 1;
    }
    nodes_.push_back(std::move(node));
    invalidate_caches();
}

void Graph::add_connection_existing(Connection conn) {
    connections_.push_back(std::move(conn));
    invalidate_caches();
}

// ============================================================================
// Deep copy �?in-memory clone (avoids JSON serialize/deserialize round-trip)
// ============================================================================
std::unique_ptr<Graph> Graph::clone() const {
    auto g = std::make_unique<Graph>();
    g->nodes_.reserve(nodes_.size());
    for (const auto& n : nodes_) {
        g->nodes_.push_back(n->clone());
    }
    g->connections_ = connections_;   // flat copy, IDs are identity
    // Recurrent delay buffers start fresh in the clone
    for (auto& conn : g->connections_) {
        if (conn.is_recurrent) {
            conn.delay_buffer = Value{0.0};
            conn.history.fill(Value{0.0});
        }
    }
    g->next_id_ = next_id_;
    g->constant_outputs_ = constant_outputs_;  // carry forward constant-output markings

    // Preserve runtime output state so mutations only re-evaluate what they change.
    // After clone, all nodes are considered "fresh" (clean). Mutation functions
    // (add_node/add_connection/remove_node/remove_connection) mark only the
    // specific downstream nodes dirty. Subsequent execute() skips clean nodes.
    for (size_t i = 0; i < nodes_.size(); ++i) {
        g->nodes_[i]->copy_outputs_from(*nodes_[i]);
        g->nodes_[i]->clear_dirty();
    }

    return g;
}

// ============================================================================
// Importance / gradient analysis (backpropagation-inspired backward pass)
// ============================================================================
std::unordered_map<uint64_t, Value> Graph::compute_importance(uint64_t output_node_id) {
    // 1. Zero all gradient accumulators
    for (auto& n : nodes_) {
        n->zero_grad();
    }

    Node* output_node = get_node(output_node_id);
    if (!output_node) return {};

    // 2. Reuse cached reverse adjacency (built by ensure_caches)
    ensure_caches();

    // 3. Build reverse topological order (iterative post-order DFS, going upstream)
    std::vector<uint64_t> topo_order;
    // Reserve for typical graph size to avoid reallocation
    topo_order.reserve(nodes_.size());
    std::unordered_set<uint64_t> visited;

    // Iterative post-order DFS using explicit stack with state flag.
    // State=false: node not yet visited, push (node, true) for post-order
    //              processing, then push children as (child, false).
    // State=true:  all children have been processed, add to topo_order.
    std::stack<std::pair<uint64_t, bool>> stack;
    stack.emplace(output_node_id, false);

    while (!stack.empty()) {
        auto [node_id, done] = stack.top();
        stack.pop();

        if (done) {
            topo_order.push_back(node_id);
            continue;
        }

        if (visited.count(node_id)) continue;
        visited.insert(node_id);

        // Schedule post-order processing (after children)
        stack.emplace(node_id, true);

        auto it = rev_adj_.find(node_id);
        if (it != rev_adj_.end()) {
            for (const auto* conn : it->second) {
                if (!visited.count(conn->src_node)) {
                    stack.emplace(conn->src_node, false);
                }
            }
        }
    }

    // 4. Seed the output node with gradient 1.0 (∂out/∂out = 1)
    output_node->accumulate_grad(config::GRADIENT_SEED);

    // 5. Walk in reverse post-order (output �?inputs), routing gradients upstream
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        uint64_t node_id = *it;
        Node* node = get_node(node_id);
        if (!node) continue;

        Value grad = node->get_grad();
        if (std::abs(grad) < config::GRADIENT_ZERO_THRESHOLD) continue;

        // Ask the node how to split its output gradient across its inputs
        std::vector<Value> input_grads = node->backward_input_grads(grad);

        // Route each input gradient to the upstream source node
        auto adj_it = rev_adj_.find(node_id);
        if (adj_it != rev_adj_.end()) {
            for (const auto* conn : adj_it->second) {
                if (conn->dst_port < input_grads.size()) {
                    Node* upstream = get_node(conn->src_node);
                    if (upstream) {
                        upstream->accumulate_grad(input_grads[conn->dst_port]);
                    }
                }
            }
        }
    }

    // 6. Build result map (node_id �?gradient), omit zero-grad nodes
    std::unordered_map<uint64_t, Value> result;
    for (auto& n : nodes_) {
        Value g = n->get_grad();
        if (std::abs(g) >= config::GRADIENT_ZERO_THRESHOLD) {
            result[n->get_id()] = g;
        }
    }
    return result;
}

// ============================================================================
// Weight training (SGD via reverse-mode autodiff)
// ============================================================================
void Graph::train(const std::vector<SampleIODesc>& samples,
                  const TrainConfig& cfg)
{
    if (samples.empty()) return;
    ensure_caches();

    // Collect trainable nodes once
    struct Trainable {
        Node* node;
        bool is_neuron;   // NEURON type
        bool is_output;   // OUTPUT type (has scale + bias)
        Trainable(Node* n, bool neuron, bool output)
            : node(n), is_neuron(neuron), is_output(output) {}
    };
    std::vector<Trainable> trainables;
    for (auto& n : nodes_) {
        auto t = n->get_type();
        if (t == NodeType::NEURON || t == NodeType::LINEAR) {
            trainables.emplace_back(n.get(), true, false);
        } else if (t == NodeType::OUTPUT) {
            trainables.emplace_back(n.get(), false, true);
        } else if (t == NodeType::CONSTANT) {
            trainables.emplace_back(n.get(), false, false);
        }
    }
    if (trainables.empty()) return;

    size_t ns = samples.size();

    // ---- Pre-build reverse structures (once per train() call) ----
    // rev_adj_node[dst] = list of predecessor nodes (for topological sort)
    // input_source[dst][port] = source node for that input port (nullptr if unconnected)
    // Built from the already-cached rev_adj_ + node_idx_ (ensure_caches() call above)
    std::unordered_map<Node*, std::vector<Node*>> rev_adj_node;
    std::unordered_map<Node*, std::vector<Node*>> input_source;
    // recurrent_inputs[dst_node_ptr][port] = true if connection is recurrent
    std::unordered_map<Node*, std::vector<bool>> recurrent_inputs;
    for (const auto& kv : rev_adj_) {
        for (const auto* conn : kv.second) {
            auto si = node_idx_.find(conn->src_node);
            auto di = node_idx_.find(conn->dst_node);
            if (si == node_idx_.end() || di == node_idx_.end()) continue;
            Node* src = nodes_[si->second].get();
            Node* dst = nodes_[di->second].get();
            rev_adj_node[dst].push_back(src);
            auto& srcs = input_source[dst];
            if (conn->dst_port >= srcs.size()) srcs.resize(conn->dst_port + 1, nullptr);
            srcs[conn->dst_port] = src;
            auto& recs = recurrent_inputs[dst];
            if (conn->dst_port >= recs.size()) recs.resize(conn->dst_port + 1, false);
            recs[conn->dst_port] = conn->is_recurrent;
        }
    }

    // ---- Build reverse topological order once using iterative DFS ----
    struct DFState { Node* node; size_t child_idx; };
    std::vector<Node*> rev_order;
    {
        std::unordered_set<Node*> visited;
        for (auto& n : nodes_) {
            if (n->get_type() == NodeType::OUTPUT && !visited.count(n.get())) {
                std::vector<DFState> stack;
                stack.push_back({n.get(), 0});
                visited.insert(n.get());
                while (!stack.empty()) {
                    auto& top = stack.back();
                    auto& preds = rev_adj_node[top.node];
                    if (top.child_idx < preds.size()) {
                        Node* child = preds[top.child_idx];
                        top.child_idx++;
                        if (!child || visited.count(child) ||
                            child->get_type() == NodeType::INPUT) continue;
                        visited.insert(child);
                        stack.push_back({child, 0});
                    } else {
                        rev_order.push_back(top.node);
                        stack.pop_back();
                    }
                }
            }
        }
    }

    // ---- Pre-allocate gradient accumulators (reused across epochs) ----
    std::unordered_map<Node*, std::vector<double>> acc_dw;
    std::unordered_map<Node*, double> acc_db;
    std::unordered_map<Node*, double> acc_dv;
    std::unordered_map<Node*, std::vector<double>> vel_w;
    std::unordered_map<Node*, double> vel_b;
    std::unordered_map<Node*, double> vel_c;
    // Adam second-moment maps (used only when cfg.use_adam = true)
    std::unordered_map<Node*, std::vector<double>> adam_v_w;
    std::unordered_map<Node*, double> adam_v_b;
    std::unordered_map<Node*, double> adam_v_c;
    for (auto& tb : trainables) {
        if (tb.is_neuron) {
            auto* nn = static_cast<NeuronNode*>(tb.node);
            acc_dw[tb.node] = std::vector<double>(nn->get_num_weights(), 0.0);
            acc_db[tb.node] = 0.0;
            vel_w[tb.node] = std::vector<double>(nn->get_num_weights(), 0.0);
            vel_b[tb.node] = 0.0;
            adam_v_w[tb.node] = std::vector<double>(nn->get_num_weights(), 0.0);
            adam_v_b[tb.node] = 0.0;
        } else if (tb.is_output) {
            acc_dw[tb.node] = std::vector<double>(1, 0.0);
            acc_db[tb.node] = 0.0;
            vel_w[tb.node] = std::vector<double>(1, 0.0);
            vel_b[tb.node] = 0.0;
            adam_v_w[tb.node] = std::vector<double>(1, 0.0);
            adam_v_b[tb.node] = 0.0;
        } else {
            acc_dv[tb.node] = 0.0;
            vel_c[tb.node] = 0.0;
            adam_v_c[tb.node] = 0.0;
        }
    }

    // Within-SGD early stopping state
    double es_best_loss = 1e18;
    int    es_patience_left = cfg.early_stop_patience;

    // ---- Mini-batch setup ----
    // batch_size=0 means full-batch (process all ns samples per step).
    // Otherwise, each epoch iterates over ceil(ns/batch_size) mini-batches,
    // giving ns/batch_size gradient steps per epoch.
    int bs = (cfg.batch_size > 0)
           ? std::min(cfg.batch_size, static_cast<int>(ns))
           : static_cast<int>(ns);
    int num_batches = (static_cast<int>(ns) + bs - 1) / bs;

    // Per-node per-sample buffers (allocated ONCE at batch capacity)
    std::vector<std::vector<Value>> node_outputs(nodes_.size());
    std::vector<std::vector<Value>> node_grads(nodes_.size());
    for (size_t ni = 0; ni < nodes_.size(); ++ni) {
        node_outputs[ni].resize(bs);
        node_grads[ni].resize(bs, 0.0);
    }

    // Adam timestep — non-static so it resets each train() call (correct
    // bias correction when moments start at zero).
    int adam_t = 0;

    // ---- Persistent per-thread clones (parallel forward pass) ----
    // Created ONCE per train() call; weights synced from the main graph
    // each batch via copy_state_to (cheap, no allocation). Replacing the
    // old clone-per-batch approach cuts dominant cost on high-dim graphs
    // (CIFAR: 1044 nodes × 10 threads × 125 batches of deep clones/epoch).
    int max_threads = (config::EVOLUTION_PARALLEL && !has_recurrent_connections())
        ? std::min(config::EVOLUTION_NUM_THREADS, static_cast<int>(ns))
        : 1;
    std::vector<std::unique_ptr<Graph>> thread_graphs;
    for (int t = 0; t < max_threads; ++t) {
        thread_graphs.push_back(clone());
    }
    // Sync trainable weights from main graph into all clones.
    // Clones preserve node order, so index i in clone == index i in main.
    // PERF: only nodes with TRAINABLE state can change during training —
    // NEURON/LINEAR (weights/bias), OUTPUT (scale/bias), CONSTANT (value).
    // Skipping INPUT and stateless arithmetic nodes avoids ~1000 virtual
    // copy calls per batch on image-shaped graphs (1024 INPUTs of 1044
    // nodes on CIFAR).
    std::vector<size_t> sync_idx;
    sync_idx.reserve(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
        auto t = nodes_[i]->get_type();
        if (t == NodeType::NEURON || t == NodeType::LINEAR
            || t == NodeType::OUTPUT || t == NodeType::CONSTANT) {
            sync_idx.push_back(i);
        }
    }
    auto sync_clones = [&]() {
        for (auto& tg : thread_graphs) {
            const auto& cnodes = tg->get_nodes();
            for (size_t i : sync_idx) {
                if (i < cnodes.size()) {
                    nodes_[i]->copy_state_to(cnodes[i].get());
                }
            }
        }
    };

    // Watchdog start (0 = disabled)
    const auto wd_start = std::chrono::steady_clock::now();
    bool watchdog_fired = false;

    for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
        double epoch_loss = 0.0;
        // Wall-clock watchdog: abort at epoch boundary when over budget.
        if (cfg.watchdog_seconds > 0
            && std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - wd_start).count()
               >= cfg.watchdog_seconds) {
            watchdog_fired = true;
            break;
        }
        // Reset recurrent state at the start of each epoch so temporal
        // tasks (running parity/sum) start fresh from sample 0.
        if (epoch > 0) reset_recurrent_state();

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int batch_start = batch_idx * bs;
            int batch_end   = std::min(batch_start + bs, static_cast<int>(ns));
            int actual_bs   = batch_end - batch_start;

        // Zero out accumulators for this batch
        for (auto& tb : trainables) {
            if (tb.is_neuron) {
                auto& dw = acc_dw[tb.node];
                std::fill(dw.begin(), dw.end(), 0.0);
                acc_db[tb.node] = 0.0;
            } else if (tb.is_output) {
                acc_dw[tb.node][0] = 0.0;
                acc_db[tb.node] = 0.0;
            } else {
                acc_dv[tb.node] = 0.0;
            }
        }

        // Zero node_grads for this batch
        for (size_t ni = 0; ni < nodes_.size(); ++ni) {
            std::fill(node_grads[ni].begin(), node_grads[ni].begin() + actual_bs, 0.0);
        }

        // ---- Batched forward pass: execute batch samples, capture outputs ----
        // Force sequential for recurrent graphs — parallel threads break
        // temporal state continuity (each thread clone starts with delay_buffer=0).
        int num_threads = (max_threads > 1)
            ? std::min(max_threads, actual_bs)
            : 1;

        if (num_threads <= 1) {
            for (int si = 0; si < actual_bs; ++si) {
                const auto& sample = samples[batch_start + si];

                if (!cfg.input_data_to_graph.empty()) {
                    for (const auto& kv : sample.inputs) {
                        auto map_it = cfg.input_data_to_graph.find(kv.first);
                        if (map_it != cfg.input_data_to_graph.end()) {
                            set_input_value(map_it->second, kv.second);
                        }
                    }
                } else {
                    for (auto& n : nodes_) {
                        if (n->get_type() == NodeType::INPUT) {
                            auto it = sample.inputs.find(n->get_id());
                            if (it != sample.inputs.end()) {
                                set_input_value(n->get_id(), it->second);
                            }
                        }
                    }
                }
                execute();

                for (size_t ni = 0; ni < nodes_.size(); ++ni) {
                    auto& n = nodes_[ni];
                    auto t = n->get_type();
                      if (t == NodeType::OUTPUT) {
                          Value raw = dynamic_cast<OutputNode*>(n.get())->get_value();
                          if (cfg.loss_type == LossType::MSE
                              || cfg.loss_type == LossType::SOFTMAX_CE) {
                              node_outputs[ni][si] = raw;
                          } else {
                              Value sig = 1.0 / (1.0 + std::exp(-raw));
                              node_outputs[ni][si] = sig;
                          }
                      } else if (n->get_num_outputs() > 0) {
                          node_outputs[ni][si] = n->get_output(0);
                    } else {
                        node_outputs[ni][si] = 0.0;
                    }
                }
            }
        } else {
            // Sync weights into persistent clones (they were created with
            // pre-training weights; each batch's update must propagate).
            sync_clones();

            int chunk_size = (actual_bs + num_threads - 1) / num_threads;
            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    int start = t * chunk_size;
                    int end   = std::min(start + chunk_size, actual_bs);
                    if (start >= end) return;
                    Graph* thread_graph = thread_graphs[t].get();
                    auto input_map = cfg.input_data_to_graph;
                    auto lt = cfg.loss_type;
                    for (int si = start; si < end; ++si) {
                        const auto& sample = samples[batch_start + si];
                        if (!input_map.empty()) {
                            for (const auto& kv : sample.inputs) {
                                auto map_it = input_map.find(kv.first);
                                if (map_it != input_map.end()) {
                                    thread_graph->set_input_value(map_it->second, kv.second);
                                }
                            }
                        } else {
                            for (auto& n : nodes_) {
                                if (n->get_type() == NodeType::INPUT) {
                                    auto it = sample.inputs.find(n->get_id());
                                    if (it != sample.inputs.end()) {
                                        thread_graph->set_input_value(n->get_id(), it->second);
                                    }
                                }
                            }
                        }
                        thread_graph->execute();
                        for (size_t ni = 0; ni < nodes_.size(); ++ni) {
                            auto& n = nodes_[ni];
                            auto t2 = n->get_type();
                            if (t2 == NodeType::OUTPUT) {
                                Node* tn = thread_graph->get_node(n->get_id());
                                Value raw = dynamic_cast<OutputNode*>(tn)->get_value();
                                if (lt == LossType::MSE || lt == LossType::SOFTMAX_CE) {
                                    node_outputs[ni][si] = raw;
                                } else {
                                    Value sig = 1.0 / (1.0 + std::exp(-raw));
                                    node_outputs[ni][si] = sig;
                                }
                            } else if (n->get_num_outputs() > 0) {
                                Node* tn = thread_graph->get_node(n->get_id());
                                node_outputs[ni][si] = tn->get_output(0);
                            } else {
                                node_outputs[ni][si] = 0.0;
                            }
                        }
                    }
                });
            }
            for (auto& th : threads) th.join();
        }

        // ---- Batched backward pass ----
        // Seed loss gradients on output nodes
        // Collect the OUTPUT node indices once (for softmax coupling).
        std::vector<size_t> sm_out_idxs;
        if (cfg.loss_type == LossType::SOFTMAX_CE) {
            for (auto& kv : samples[batch_start].targets) {
                uint64_t gid = kv.first;
                if (!cfg.output_data_to_graph.empty()) {
                    auto mi = cfg.output_data_to_graph.find(kv.first);
                    if (mi == cfg.output_data_to_graph.end()) continue;
                    gid = mi->second;
                }
                auto oi = node_idx_.find(gid);
                if (oi != node_idx_.end()) sm_out_idxs.push_back(oi->second);
            }
        }

        if (cfg.loss_type == LossType::SOFTMAX_CE) {
            // Softmax cross-entropy: p = softmax(logits) over ALL outputs
            // jointly; seed grad_c = p_c - y_c. The coupling (pushing down
            // rivals when one output fires) is what makes context statistics
            // learnable — independent one-vs-rest gradients cannot express
            // "u is likely BECAUSE q fired and others are not".
            for (int si = 0; si < actual_bs; ++si) {
                const auto& sample = samples[batch_start + si];
                double mx = -1e300;
                for (size_t oi_ : sm_out_idxs) {
                    mx = std::max(mx, static_cast<double>(node_outputs[oi_][si]));
                }
                double denom = 0.0;
                for (size_t oi_ : sm_out_idxs) {
                    denom += std::exp(static_cast<double>(node_outputs[oi_][si]) - mx);
                }
                for (size_t oi_ : sm_out_idxs) {
                    uint64_t gid2 = nodes_[oi_]->get_id();
                    // target for this output: reverse-map graph id -> data key
                    Value y = 0.0;
                    for (const auto& kv : sample.targets) {
                        auto mi = cfg.output_data_to_graph.find(kv.first);
                        if (mi != cfg.output_data_to_graph.end()
                            && mi->second == gid2) { y = kv.second; break; }
                    }
                    Value p = static_cast<Value>(
                        std::exp(static_cast<double>(node_outputs[oi_][si]) - mx) / denom);
                    Value grad = p - y;
                    if (grad > cfg.gradient_clip) grad = cfg.gradient_clip;
                    else if (grad < -cfg.gradient_clip) grad = -cfg.gradient_clip;
                    node_grads[oi_][si] = grad;
                }
            }
        } else
        for (auto& kv : samples[batch_start].targets) {
            uint64_t data_key = kv.first;
            uint64_t graph_node_id = data_key;
            if (!cfg.output_data_to_graph.empty()) {
                auto map_it = cfg.output_data_to_graph.find(data_key);
                if (map_it == cfg.output_data_to_graph.end()) continue;
                graph_node_id = map_it->second;
            }
            Node* tgt_node = get_node(graph_node_id);
            if (!tgt_node || tgt_node->get_type() != NodeType::OUTPUT) continue;
            auto oi = node_idx_.find(graph_node_id);
            if (oi == node_idx_.end()) continue;
            size_t out_idx = oi->second;
            for (int si = 0; si < actual_bs; ++si) {
                const auto& sample = samples[batch_start + si];
                auto tit = sample.targets.find(data_key);
                if (tit == sample.targets.end()) continue;
                Value sig_out = node_outputs[out_idx][si];
                Value grad = sig_out - tit->second;
                if (grad > cfg.gradient_clip) grad = cfg.gradient_clip;
                else if (grad < -cfg.gradient_clip) grad = -cfg.gradient_clip;
                node_grads[out_idx][si] = grad;
            }
        }

        // Backward propagation through reverse topological order
        for (auto rit = rev_order.rbegin(); rit != rev_order.rend(); ++rit) {
            Node* v = *rit;
            auto vi_it = node_idx_.find(v->get_id());
            if (vi_it == node_idx_.end()) continue;
            size_t vi = vi_it->second;
            auto vt = v->get_type();

            auto src_it = input_source.find(v);
            const std::vector<Node*>* preds = (src_it != input_source.end())
                ? &src_it->second : nullptr;

            for (int si = 0; si < actual_bs; ++si) {
                Value vgrad = node_grads[vi][si];
                if (std::abs(vgrad) < config::GRADIENT_ZERO_THRESHOLD) continue;

                auto add_grad = [&](size_t i, Value g) {
                    Node* pred = (preds && i < preds->size()) ? (*preds)[i] : nullptr;
                    if (pred && pred->get_type() != NodeType::INPUT) {
                        auto pi_it = node_idx_.find(pred->get_id());
                        if (pi_it != node_idx_.end())
                            node_grads[pi_it->second][si] += g;
                    }
                };
                auto pred_output = [&](size_t port) -> Value {
                    Node* pred = (preds && port < preds->size()) ? (*preds)[port] : nullptr;
                    if (!pred) return 0.0;
                    auto pi_it = node_idx_.find(pred->get_id());
                    if (pi_it == node_idx_.end()) return 0.0;
                    auto rec_it = recurrent_inputs.find(v);
                    bool recur = false;
                    if (rec_it != recurrent_inputs.end() && port < rec_it->second.size())
                        recur = rec_it->second[port];
                    if (recur && si > 0)
                        return node_outputs[pi_it->second][si - 1];
                    return node_outputs[pi_it->second][si];
                };

                switch (vt) {
                case NodeType::NEURON: {
                    auto* nn = static_cast<NeuronNode*>(v);
                    Value out_val = node_outputs[vi][si];
                    Value dtanh = 1.0 - out_val * out_val;
                    acc_db[nn] += vgrad * dtanh;
                    for (size_t i = 0; i < nn->get_num_weights(); ++i) {
                        Value xi = pred_output(i);
                        acc_dw[nn][i] += vgrad * dtanh * xi;
                    }
                    for (size_t i = 0; i < v->get_num_inputs(); ++i) {
                        Value w = nn->get_weight(i);
                        add_grad(i, vgrad * dtanh * w);
                    }
                    break;
                }
                case NodeType::LINEAR: {
                    auto* ln = static_cast<NeuronNode*>(v);
                    acc_db[ln] += vgrad;
                    for (size_t i = 0; i < ln->get_num_weights(); ++i) {
                        Value xi = pred_output(i);
                        acc_dw[ln][i] += vgrad * xi;
                    }
                    for (size_t i = 0; i < v->get_num_inputs(); ++i) {
                        Value w = ln->get_weight(i);
                        add_grad(i, vgrad * w);
                    }
                    break;
                }
                case NodeType::CONSTANT:
                    acc_dv[v] += vgrad;
                    break;
                case NodeType::INPUT:
                case NodeType::SINK:
                    break;
                case NodeType::ADD:
                    for (size_t i = 0; i < v->get_num_inputs(); ++i)
                        add_grad(i, vgrad);
                    break;
                case NodeType::SUBTRACT:
                    add_grad(0, vgrad);
                    add_grad(1, -vgrad);
                    break;
                case NodeType::NEGATE:
                    add_grad(0, -vgrad);
                    break;
                case NodeType::OUTPUT: {
                    auto* on = static_cast<OutputNode*>(v);
                    Value in_val = pred_output(0);
                    acc_dw[on][0] += vgrad * in_val;
                    acc_db[on] += vgrad;
                    add_grad(0, vgrad * on->get_scale());
                    break;
                }
                case NodeType::MULTIPLY: {
                    Value a_out = pred_output(0);
                    Value b_out = pred_output(1);
                    add_grad(0, vgrad * b_out);
                    add_grad(1, vgrad * a_out);
                    break;
                }
                case NodeType::DIVIDE: {
                    Value a_out = pred_output(0);
                    Value b_out = pred_output(1);
                    if (std::abs(b_out) > config::BACKWARD_DIV_EPSILON) {
                        add_grad(0, vgrad / b_out);
                        add_grad(1, vgrad * (-a_out / (b_out * b_out)));
                    }
                    break;
                }
                case NodeType::RELU: {
                    Value x_out = pred_output(0);
                    add_grad(0, (x_out > 0.0) ? vgrad : 0.0);
                    break;
                }
                case NodeType::SIGMOID: {
                    Value out_val = node_outputs[vi][si];
                    add_grad(0, vgrad * out_val * (1.0 - out_val));
                    break;
                }
                case NodeType::TANH: {
                    Value out_val = node_outputs[vi][si];
                    add_grad(0, vgrad * (1.0 - out_val * out_val));
                    break;
                }
                case NodeType::SIN: {
                    Value in_val = pred_output(0);
                    add_grad(0, vgrad * std::cos(in_val));
                    break;
                }
                default: {
                    std::vector<Value> saved_inputs(v->get_num_inputs());
                    for (size_t i = 0; i < v->get_num_inputs(); ++i)
                        saved_inputs[i] = v->get_input(i);
                    Value saved_output = (v->get_num_outputs() > 0) ? v->get_output(0) : 0.0;

                    if (preds) {
                        for (size_t i = 0; i < v->get_num_inputs() && i < preds->size(); ++i)
                            v->set_input(i, pred_output(i));
                    }
                    if (v->get_num_outputs() > 0)
                        v->set_output(0, node_outputs[vi][si]);

                    auto gradients = v->backward_input_grads(vgrad);
                    for (size_t i = 0; i < v->get_num_inputs() && i < gradients.size(); ++i)
                        add_grad(i, gradients[i]);

                    for (size_t i = 0; i < v->get_num_inputs(); ++i)
                        v->set_input(i, saved_inputs[i]);
                    if (v->get_num_outputs() > 0)
                        v->set_output(0, saved_output);
                    break;
                }
                }
            }
        }

        // ---- Apply gradients (Adam or momentum SGD) ----
        double lr   = cfg.learning_rate;
        double mu   = cfg.momentum;
        double wd   = cfg.weight_decay;
        double nscale = 1.0 / static_cast<double>(actual_bs);

        if (cfg.use_adam) {
            adam_t++;
            double b1 = cfg.adam_beta1, b2 = cfg.adam_beta2, eps = cfg.adam_eps;
            double bc1 = 1.0 - std::pow(b1, adam_t);
            double bc2 = 1.0 - std::pow(b2, adam_t);

            for (auto& tb : trainables) {
                if (tb.is_neuron) {
                    auto* nn = static_cast<NeuronNode*>(tb.node);
                    auto& dw = acc_dw[tb.node];
                    auto& m_w = vel_w[tb.node];
                    auto& v_w = adam_v_w[tb.node];
                    auto& m_b = vel_b[tb.node];
                    auto& v_b = adam_v_b[tb.node];
                    for (size_t i = 0; i < nn->get_num_weights(); ++i) {
                        double g = nscale * dw[i] + wd * nn->get_weight(i);
                        m_w[i] = b1 * m_w[i] + (1.0 - b1) * g;
                        v_w[i] = b2 * v_w[i] + (1.0 - b2) * g * g;
                        nn->set_weight(i, nn->get_weight(i) - lr * (m_w[i] / bc1) / (std::sqrt(v_w[i] / bc2) + eps));
                    }
                    double gb = nscale * acc_db[tb.node] + wd * nn->get_bias();
                    m_b = b1 * m_b + (1.0 - b1) * gb;
                    v_b = b2 * v_b + (1.0 - b2) * gb * gb;
                    nn->set_bias(nn->get_bias() - lr * (m_b / bc1) / (std::sqrt(v_b / bc2) + eps));
                } else if (tb.is_output) {
                    auto* on = static_cast<OutputNode*>(tb.node);
                    auto& m_w = vel_w[tb.node];
                    auto& v_w = adam_v_w[tb.node];
                    auto& m_b = vel_b[tb.node];
                    auto& v_b = adam_v_b[tb.node];
                    double gs = nscale * acc_dw[tb.node][0] + wd * on->get_scale();
                    m_w[0] = b1 * m_w[0] + (1.0 - b1) * gs;
                    v_w[0] = b2 * v_w[0] + (1.0 - b2) * gs * gs;
                    on->set_scale(on->get_scale() - lr * (m_w[0] / bc1) / (std::sqrt(v_w[0] / bc2) + eps));
                    double gb = nscale * acc_db[tb.node] + wd * on->get_bias();
                    m_b = b1 * m_b + (1.0 - b1) * gb;
                    v_b = b2 * v_b + (1.0 - b2) * gb * gb;
                    on->set_bias(on->get_bias() - lr * (m_b / bc1) / (std::sqrt(v_b / bc2) + eps));
                } else {
                    auto* cn = static_cast<ConstantNode*>(tb.node);
                    auto& m_c = vel_c[tb.node];
                    auto& v_c = adam_v_c[tb.node];
                    double gv = nscale * acc_dv[tb.node] + wd * cn->get_value();
                    m_c = b1 * m_c + (1.0 - b1) * gv;
                    v_c = b2 * v_c + (1.0 - b2) * gv * gv;
                    cn->set_value(cn->get_value() - lr * (m_c / bc1) / (std::sqrt(v_c / bc2) + eps));
                }
            }
        } else {
            for (auto& tb : trainables) {
                if (tb.is_neuron) {
                    auto* nn = static_cast<NeuronNode*>(tb.node);
                    auto& dw = acc_dw[tb.node];
                    auto& vw = vel_w[tb.node];
                    auto& vb = vel_b[tb.node];
                    for (size_t i = 0; i < nn->get_num_weights(); ++i) {
                        double avg_grad = nscale * dw[i];
                        double w = nn->get_weight(i);
                        vw[i] = mu * vw[i] + avg_grad + wd * w;
                        nn->set_weight(i, w - lr * vw[i]);
                    }
                    double avg_db = nscale * acc_db[tb.node];
                    vb = mu * vb + avg_db + wd * nn->get_bias();
                    nn->set_bias(nn->get_bias() - lr * vb);
                } else if (tb.is_output) {
                    auto* on = static_cast<OutputNode*>(tb.node);
                    double avg_ds = nscale * acc_dw[tb.node][0];
                    double s = on->get_scale();
                    vel_w[tb.node][0] = mu * vel_w[tb.node][0] + avg_ds + wd * s;
                    on->set_scale(s - lr * vel_w[tb.node][0]);
                    double avg_db = nscale * acc_db[tb.node];
                    vel_b[tb.node] = mu * vel_b[tb.node] + avg_db + wd * on->get_bias();
                    on->set_bias(on->get_bias() - lr * vel_b[tb.node]);
                } else {
                    auto* cn = static_cast<ConstantNode*>(tb.node);
                    double avg_dv = nscale * acc_dv[tb.node];
                    vel_c[tb.node] = mu * vel_c[tb.node] + avg_dv + wd * cn->get_value();
                    cn->set_value(cn->get_value() - lr * vel_c[tb.node]);
                }
            }
        }

        // Accumulate this batch's loss into epoch_loss
        if (cfg.loss_type == LossType::SOFTMAX_CE && !sm_out_idxs.empty()) {
            // Per-sample softmax-CE over the joint output set (raw logits
            // in node_outputs). Mirrors the gradient seeding.
            double batch_loss_acc = 0.0;
            for (int si = 0; si < actual_bs; ++si) {
                double mx = -1e300;
                for (size_t oi_ : sm_out_idxs) {
                    mx = std::max(mx, static_cast<double>(node_outputs[oi_][si]));
                }
                double denom = 0.0;
                for (size_t oi_ : sm_out_idxs) {
                    denom += std::exp(static_cast<double>(node_outputs[oi_][si]) - mx);
                }
                double lse = mx + std::log(denom);   // log-sum-exp
                // -log p_true = lse - logit_true
                for (size_t oi_ : sm_out_idxs) {
                    uint64_t gid2 = nodes_[oi_]->get_id();
                    for (const auto& kv : samples[batch_start + si].targets) {
                        auto mi = cfg.output_data_to_graph.find(kv.first);
                        if (mi != cfg.output_data_to_graph.end()
                            && mi->second == gid2 && kv.second > 0.5) {
                            batch_loss_acc += lse - static_cast<double>(node_outputs[oi_][si]);
                            break;
                        }
                    }
                }
            }
            epoch_loss += batch_loss_acc;
        } else {
            double batch_loss = 0.0;
            for (int si = 0; si < actual_bs; ++si) {
                for (auto& kv : samples[batch_start + si].targets) {
                    auto map_it = cfg.output_data_to_graph.find(kv.first);
                    if (map_it == cfg.output_data_to_graph.end()) continue;
                    auto oi = node_idx_.find(map_it->second);
                    if (oi == node_idx_.end()) continue;
                    Value pred = node_outputs[oi->second][si];
                    Value target = kv.second;
                    if (cfg.loss_type == LossType::SOFTMAX_CE) {
                        continue;
                    } else if (cfg.loss_type == LossType::BCE) {
                        Value sig = 1.0 / (1.0 + std::exp(-pred));
                        constexpr double bce_eps = 1e-9;
                        double clamped = std::max(bce_eps, std::min(1.0 - bce_eps, sig));
                        batch_loss += -(target * std::log(clamped) + (1.0 - target) * std::log(1.0 - clamped));
                    } else {
                        Value diff = pred - target;
                        batch_loss += diff * diff;
                    }
                }
            }
            epoch_loss += batch_loss;
        }

        } // end mini-batch loop

        // ---- Within-SGD early stopping (per-epoch, not per-batch) ----
        epoch_loss /= static_cast<double>(ns);
        if (cfg.early_stop_patience > 0) {
            if (epoch_loss < es_best_loss - cfg.early_stop_min_improvement) {
                es_best_loss = epoch_loss;
                es_patience_left = cfg.early_stop_patience;
            } else {
                es_patience_left--;
                if (es_patience_left <= 0) {
                    break;
                }
            }
        }
    }
    if (watchdog_fired) {
        Logger::info("Train watchdog fired: aborted after "
                    + std::to_string(cfg.watchdog_seconds) + "s wall (live-lock guard)");
    }
    set_all_dirty();
}

// ============================================================================
// Error attribution (perturbation-based blame analysis)
// ============================================================================
std::vector<Graph::ErrorAttributionResult> Graph::compute_error_attribution(
    const std::vector<SampleIODesc>& samples,
    Value epsilon,
    int max_candidates,
    unsigned int seed,
    const std::unordered_map<uint64_t, uint64_t>& input_data_to_graph,
    const std::unordered_map<uint64_t, uint64_t>& output_data_to_graph)
{
    // Collect candidate nodes (skip IO/sink, must have outputs)
    std::vector<Node*> candidates;
    for (auto& n : nodes_) {
        auto t = n->get_type();
        if (t != NodeType::INPUT && t != NodeType::OUTPUT &&
            t != NodeType::SINK && n->get_num_outputs() > 0) {
            candidates.push_back(n.get());
        }
    }
    if (candidates.empty()) return {};

    // Randomly subsample candidates when max_candidates > 0 and less than total
    // Greatly reduces O(samples × candidates) graph executions per call.
    if (max_candidates > 0 && max_candidates < static_cast<int>(candidates.size())) {
        std::mt19937 cand_rng(seed);
        std::shuffle(candidates.begin(), candidates.end(), cand_rng);
        candidates.resize(max_candidates);
    }

    // Accumulate per-node error sums
    std::unordered_map<uint64_t, Value> sum_base;
    std::unordered_map<uint64_t, Value> sum_pert;
    int sample_count = 0;

    int ns = static_cast<int>(samples.size());
    int num_threads = config::EVOLUTION_PARALLEL
        ? std::min(config::EVOLUTION_NUM_THREADS, ns)
        : 1;

    if (num_threads <= 1) {
        for (const auto& sample : samples) {
            sample_count++;

            // Reset recurrent state for independent sample evaluation
            reset_recurrent_state();

            // --- Baseline (no perturbation on any node) ---
            for (auto& n : nodes_) n->set_perturbation(0.0);
            if (!input_data_to_graph.empty()) {
                for (const auto& kv : sample.inputs) {
                    auto it = input_data_to_graph.find(kv.first);
                    if (it != input_data_to_graph.end()) {
                        set_input_value(it->second, kv.second);
                    }
                }
            } else {
                for (auto& n : nodes_) {
                    if (n->get_type() == NodeType::INPUT) {
                        auto it = sample.inputs.find(n->get_id());
                        if (it != sample.inputs.end())
                            set_input_value(n->get_id(), it->second);
                    }
                }
            }
            execute();

            Value base_error = 0.0;
            if (!output_data_to_graph.empty()) {
                for (const auto& [data_key, target] : sample.targets) {
                    auto it = output_data_to_graph.find(data_key);
                    if (it == output_data_to_graph.end()) continue;
                    Value diff = get_output_value(it->second) - target;
                    base_error += diff * diff;
                }
            } else {
                for (const auto& [id, target] : sample.targets) {
                    const Node* out_node = get_node(id);
                    if (!out_node || out_node->get_type() != NodeType::OUTPUT) continue;
                    Value diff = get_output_value(id) - target;
                    base_error += diff * diff;
                }
            }
            base_error /= static_cast<Value>(sample.targets.size());

            // --- Perturb each candidate one at a time ---
            for (Node* cand : candidates) {
                // Reset recurrent state so each candidate starts from same baseline
                reset_recurrent_state();

                // Clear all perturbations, set only this candidate's
                for (auto& n : nodes_) n->set_perturbation(0.0);
                cand->set_perturbation(epsilon);
                mark_transitively_dirty(cand->get_id());

                // Re-evaluate
                if (!input_data_to_graph.empty()) {
                    for (const auto& kv : sample.inputs) {
                        auto it = input_data_to_graph.find(kv.first);
                        if (it != input_data_to_graph.end()) {
                            set_input_value(it->second, kv.second);
                        }
                    }
                } else {
                    for (auto& n : nodes_) {
                        if (n->get_type() == NodeType::INPUT) {
                            auto it = sample.inputs.find(n->get_id());
                            if (it != sample.inputs.end())
                                set_input_value(n->get_id(), it->second);
                        }
                    }
                }
                execute();

                Value pert_error = 0.0;
                if (!output_data_to_graph.empty()) {
                    for (const auto& [data_key, target] : sample.targets) {
                        auto it = output_data_to_graph.find(data_key);
                        if (it == output_data_to_graph.end()) continue;
                        Value diff = get_output_value(it->second) - target;
                        pert_error += diff * diff;
                    }
                } else {
                    for (const auto& [id, target] : sample.targets) {
                        const Node* out_node = get_node(id);
                        if (!out_node || out_node->get_type() != NodeType::OUTPUT) continue;
                        Value diff = get_output_value(id) - target;
                        pert_error += diff * diff;
                    }
                }
                pert_error /= static_cast<Value>(sample.targets.size());

                sum_base[cand->get_id()] += base_error;
                sum_pert[cand->get_id()] += pert_error;
            }
        }
    } else {
            int chunk_size = (ns + num_threads - 1) / num_threads;
            std::vector<int> t_sample_counts(num_threads, 0);
            std::vector<std::unordered_map<uint64_t, Value>> t_sum_base(num_threads);
            std::vector<std::unordered_map<uint64_t, Value>> t_sum_pert(num_threads);
            std::vector<std::thread> threads;

            std::vector<uint64_t> cand_ids;
            for (Node* cand : candidates) cand_ids.push_back(cand->get_id());

            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    int start = t * chunk_size;
                    int end   = std::min(start + chunk_size, ns);
                    if (start >= end) return;
                    auto tg = clone();
                    auto& local_base = t_sum_base[t];
                    auto& local_pert = t_sum_pert[t];
                    int sc = 0;

                    for (int si = start; si < end; ++si) {
                        const auto& sample = samples[si];
                        sc++;

                        tg->reset_recurrent_state();

                        // --- Baseline ---
                        for (auto& n : nodes_) {
                            tg->get_node(n->get_id())->set_perturbation(0.0);
                        }
                        if (!input_data_to_graph.empty()) {
                            for (const auto& kv : sample.inputs) {
                                auto it = input_data_to_graph.find(kv.first);
                                if (it != input_data_to_graph.end()) {
                                    tg->set_input_value(it->second, kv.second);
                                }
                            }
                        } else {
                            for (auto& n : nodes_) {
                                if (n->get_type() == NodeType::INPUT) {
                                    auto it = sample.inputs.find(n->get_id());
                                    if (it != sample.inputs.end())
                                        tg->set_input_value(n->get_id(), it->second);
                                }
                            }
                        }
                        tg->execute();

                        Value base_error = 0.0;
                        if (!output_data_to_graph.empty()) {
                            for (const auto& [data_key, target] : sample.targets) {
                                auto it = output_data_to_graph.find(data_key);
                                if (it == output_data_to_graph.end()) continue;
                                Value diff = tg->get_output_value(it->second) - target;
                                base_error += diff * diff;
                            }
                        } else {
                            for (const auto& [id, target] : sample.targets) {
                                const Node* out_node = tg->get_node(id);
                                if (!out_node || out_node->get_type() != NodeType::OUTPUT) continue;
                                Value diff = tg->get_output_value(id) - target;
                                base_error += diff * diff;
                            }
                        }
                        base_error /= static_cast<Value>(sample.targets.size());

                        // --- Perturb each candidate ---
                        for (uint64_t cid : cand_ids) {
                            tg->reset_recurrent_state();

                            for (auto& n : nodes_) {
                                tg->get_node(n->get_id())->set_perturbation(0.0);
                            }
                            tg->get_node(cid)->set_perturbation(epsilon);
                            tg->mark_transitively_dirty(cid);

                            if (!input_data_to_graph.empty()) {
                                for (const auto& kv : sample.inputs) {
                                    auto it = input_data_to_graph.find(kv.first);
                                    if (it != input_data_to_graph.end()) {
                                        tg->set_input_value(it->second, kv.second);
                                    }
                                }
                            } else {
                                for (auto& n : nodes_) {
                                    if (n->get_type() == NodeType::INPUT) {
                                        auto it = sample.inputs.find(n->get_id());
                                        if (it != sample.inputs.end())
                                            tg->set_input_value(n->get_id(), it->second);
                                    }
                                }
                            }
                            tg->execute();

                            Value pert_error = 0.0;
                            if (!output_data_to_graph.empty()) {
                                for (const auto& [data_key, target] : sample.targets) {
                                    auto it = output_data_to_graph.find(data_key);
                                    if (it == output_data_to_graph.end()) continue;
                                    Value diff = tg->get_output_value(it->second) - target;
                                    pert_error += diff * diff;
                                }
                            } else {
                                for (const auto& [id, target] : sample.targets) {
                                    const Node* out_node = tg->get_node(id);
                                    if (!out_node || out_node->get_type() != NodeType::OUTPUT) continue;
                                    Value diff = tg->get_output_value(id) - target;
                                    pert_error += diff * diff;
                                }
                            }
                            pert_error /= static_cast<Value>(sample.targets.size());

                            local_base[cid] += base_error;
                            local_pert[cid] += pert_error;
                        }
                    }
                    t_sample_counts[t] = sc;
                });
            }
            for (auto& th : threads) th.join();

            for (int t = 0; t < num_threads; ++t) {
                sample_count += t_sample_counts[t];
                for (const auto& [id, val] : t_sum_base[t]) {
                    sum_base[id] += val;
                }
                for (const auto& [id, val] : t_sum_pert[t]) {
                    sum_pert[id] += val;
                }
            }
        }

    // Build result list, sorted by |blame| descending
    std::vector<ErrorAttributionResult> results;
    double n = static_cast<double>(sample_count);
    for (const auto& [id, base_sum] : sum_base) {
        ErrorAttributionResult r;
        r.node_id = id;
        r.base_error = base_sum / n;
        r.perturbed_error = sum_pert[id] / n;
        r.blame = r.perturbed_error - r.base_error;
        results.push_back(r);
    }

    std::sort(results.begin(), results.end(),
              [](const ErrorAttributionResult& a, const ErrorAttributionResult& b) {
                  return std::abs(a.blame) > std::abs(b.blame);
              });

    return results;
}

// ============================================================================
// Node-type replacement search (uses error attribution)
// ============================================================================
std::vector<Graph::NodeImprovementResult> Graph::search_improvements(
    const std::vector<SampleIODesc>& samples,
    const std::vector<ErrorAttributionResult>& attribution,
    int top_k,
    bool non_nn_only,
    std::unique_ptr<Graph>* best_trial,
    unsigned int seed)
{
    std::vector<NodeImprovementResult> results;
    Value best_improvement = 0.0;  // track best for clone capture
    if (samples.empty() || attribution.empty()) return results;

    size_t num_outputs = samples[0].targets.size();
    if (num_outputs == 0) return results;

    // ---- constants (from constants.h) ----
    constexpr int NUM_SUBSETS       = config::SEARCH_IMPROVEMENT_NUM_SUBSETS;
    constexpr int NUM_INSERT_TRIALS = config::SEARCH_IMPROVEMENT_NUM_INSERT_TRIALS;

    // ---- select top-k nodes by absolute |blame| (not just negative) ----
    std::vector<const ErrorAttributionResult*> blameworthy;
    for (const auto& r : attribution) {
        blameworthy.push_back(&r);
    }
    if (blameworthy.empty()) return results;
    std::sort(blameworthy.begin(), blameworthy.end(),
              [](const ErrorAttributionResult* a, const ErrorAttributionResult* b) {
                  return std::abs(a->blame) > std::abs(b->blame);
              });
    if (static_cast<int>(blameworthy.size()) > top_k)
        blameworthy.resize(static_cast<size_t>(top_k));

    // ---- baseline MSE ----
    Value baseline_error = 0.0;
    for (const auto& sample : samples) {
        for (auto& n : nodes_) n->set_perturbation(0.0);
        for (auto& n : nodes_) {
            if (n->get_type() == NodeType::INPUT) {
                auto it = sample.inputs.find(n->get_id());
                if (it != sample.inputs.end())
                    set_input_value(n->get_id(), it->second);
            }
        }
        reset_recurrent_state();
        execute();
        for (const auto& [id, target] : sample.targets) {
            const Node* out_node = get_node(id);
            if (!out_node || out_node->get_type() != NodeType::OUTPUT) continue;
            Value diff = get_output_value(id) - target;
            baseline_error += diff * diff;
        }
    }
    baseline_error /= static_cast<Value>(samples.size() * num_outputs);

    // ---- candidate-type builder: (num_inputs) �?types ----
    auto build_candidates = [non_nn_only](size_t ni) -> std::vector<NodeType> {
        std::vector<NodeType> out;
        if (ni == 1) {
            out.push_back(NodeType::NEGATE);
            out.push_back(NodeType::NOT);
            out.push_back(NodeType::ABSENT);
            if (!non_nn_only) {
                out.push_back(NodeType::RELU);
                out.push_back(NodeType::SIGMOID);
                out.push_back(NodeType::TANH);
                out.push_back(NodeType::NEURON);
            }
        } else if (ni == 2) {
            out.push_back(NodeType::ADD);
            out.push_back(NodeType::SUBTRACT);
            out.push_back(NodeType::MULTIPLY);
            out.push_back(NodeType::DIVIDE);
            out.push_back(NodeType::AND);
            out.push_back(NodeType::OR);
            out.push_back(NodeType::XOR);
            out.push_back(NodeType::EQUAL);
            out.push_back(NodeType::NOT_EQUAL);
            out.push_back(NodeType::GREATER);
            out.push_back(NodeType::LESS);
            out.push_back(NodeType::GREATER_EQUAL);
            out.push_back(NodeType::LESS_EQUAL);
            out.push_back(NodeType::NEURON);
        } else if (ni == 3) {
            out.push_back(NodeType::IF);
            out.push_back(NodeType::IFELSE);
            out.push_back(NodeType::NEURON);
        }
        return out;
    };

    // ---- analytical node type scorer (forward-only, no graph execution) ----
    auto eval_candidate = [](NodeType type, const std::vector<Value>& inputs) -> Value {
        auto in = [&](size_t i) -> Value { return i < inputs.size() ? inputs[i] : 0.0; };
        auto safe_div = [](Value a, Value b) -> Value {
            return std::abs(b) < config::BACKWARD_DIV_EPSILON ? 0.0 : a / b;
        };
        switch (type) {
        case NodeType::ADD:            return in(0) + in(1);
        case NodeType::SUBTRACT:       return in(0) - in(1);
        case NodeType::MULTIPLY:       return in(0) * in(1);
        case NodeType::DIVIDE:         return safe_div(in(0), in(1));
        case NodeType::NEGATE:         return -in(0);
        case NodeType::RELU:           { Value x = in(0); return x > 0.0 ? x : 0.0; }
        case NodeType::SIGMOID:        return 1.0 / (1.0 + std::exp(-in(0)));
        case NodeType::TANH:           return std::tanh(in(0));
        case NodeType::IF:             return in(0) > 0.0 ? in(1) : 0.0;
        case NodeType::IFELSE:         return in(0) > 0.0 ? in(1) : 0.0;
        case NodeType::EQUAL:          return (in(0) == in(1)) ? 1.0 : 0.0;
        case NodeType::NOT_EQUAL:      return (in(0) != in(1)) ? 1.0 : 0.0;
        case NodeType::GREATER:        return (in(0) > in(1)) ? 1.0 : 0.0;
        case NodeType::LESS:           return (in(0) < in(1)) ? 1.0 : 0.0;
        case NodeType::GREATER_EQUAL:  return (in(0) >= in(1)) ? 1.0 : 0.0;
        case NodeType::LESS_EQUAL:     return (in(0) <= in(1)) ? 1.0 : 0.0;
        case NodeType::AND:            return (in(0) > 0.0 && in(1) > 0.0) ? 1.0 : 0.0;
        case NodeType::OR:             return (in(0) > 0.0 || in(1) > 0.0) ? 1.0 : 0.0;
        case NodeType::NOT:            return (in(0) > 0.0) ? 0.0 : 1.0;
        case NodeType::XOR:            return ((in(0) > 0.0) != (in(1) > 0.0)) ? 1.0 : 0.0;
        case NodeType::SINK:
        case NodeType::ABSENT:         return 0.0;
        case NodeType::CONSTANT:       return 0.0;
        case NodeType::NEURON:         {
            // Xavier-like weighted sum: sum(inputs) / max(1, sqrt(n))
            Value sum = 0.0;
            for (const auto& v : inputs) sum += v;
            Value n = static_cast<Value>(std::max(size_t(1), inputs.size()));
            return sum / std::sqrt(n);
        }
        default:                       return 0.0;
        }
    };

    // ---- RNG ----
    std::mt19937 rng(seed);

    auto rand_idx = [&rng](size_t n) -> size_t {
        if (n == 0) return 0;
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        return dist(rng);
    };
    auto rand_bool = [&rng]() -> bool {
        std::uniform_int_distribution<int> dist(0, 1);
        return dist(rng) == 0;
    };

    // ---- pre-compute baseline error on a quick sample subset ----
    constexpr int    TRIAL_SUBSET = config::SEARCH_IMPROVEMENT_TRIAL_SUBSET;
    int quick_count = std::min(TRIAL_SUBSET, static_cast<int>(samples.size()));
    Value baseline_subset_error = 0.0;
    if (quick_count < static_cast<int>(samples.size())) {
        for (int si = 0; si < quick_count; ++si) {
            const auto& sample = samples[si];
            for (auto& n : nodes_) n->set_perturbation(0.0);
            for (auto& n : nodes_) {
                if (n->get_type() == NodeType::INPUT) {
                    auto it = sample.inputs.find(n->get_id());
                    if (it != sample.inputs.end())
                        set_input_value(n->get_id(), it->second);
                }
            }
                    reset_recurrent_state();
                    execute();
                    for (const auto& [id, target] : sample.targets) {
                        const Node* out_node = get_node(id);
                        if (!out_node || out_node->get_type() != NodeType::OUTPUT) continue;
                        Value diff = get_output_value(id) - target;
                        baseline_subset_error += diff * diff;            }
        }
        baseline_subset_error /= static_cast<Value>(quick_count * num_outputs);
    }

    // ---- evaluate a trial graph (two-stage: subset pre-screen, then full eval) ----
    auto evaluate_trial = [&](Graph& trial) -> std::pair<bool, Value> {
        // Stage 1: quick pre-screen on sample subset (skip most bad trials early)
        if (quick_count < static_cast<int>(samples.size())) {
            Value quick_error = 0.0;
            for (int si = 0; si < quick_count; ++si) {
                const auto& sample = samples[si];
                for (auto& n : trial.get_nodes()) {
                    if (n->get_type() == NodeType::INPUT) {
                        auto it = sample.inputs.find(n->get_id());
                        if (it != sample.inputs.end())
                            trial.set_input_value(n->get_id(), it->second);
                    }
                }
                try {
                    trial.reset_recurrent_state();
                    trial.execute();
                } catch (...) {
                    return {false, 0.0};
                }
                for (const auto& [id, target] : sample.targets) {
                    const Node* out_node = trial.get_node(id);
                    if (!out_node || out_node->get_type() != NodeType::OUTPUT) continue;
                    Value diff = trial.get_output_value(id) - target;
                    quick_error += diff * diff;
                }
            }
            quick_error /= static_cast<Value>(quick_count * num_outputs);
            // Skip full evaluation if trial is no better than baseline on subset
            if (quick_error >= baseline_subset_error) {
                return {false, 0.0};
            }
        }

        // Stage 2: full evaluation
        Value trial_error = 0.0;
        for (const auto& sample : samples) {
            for (auto& n : trial.get_nodes()) {
                if (n->get_type() == NodeType::INPUT) {
                    auto it = sample.inputs.find(n->get_id());
                    if (it != sample.inputs.end())
                        trial.set_input_value(n->get_id(), it->second);
                }
            }
            try {
                trial.reset_recurrent_state();
                trial.execute();
            } catch (...) {
                return {false, 0.0};
            }
            for (const auto& [id, target] : sample.targets) {
                const Node* out_node = trial.get_node(id);
                if (!out_node || out_node->get_type() != NodeType::OUTPUT) continue;
                Value diff = trial.get_output_value(id) - target;
                trial_error += diff * diff;
            }
        }
        trial_error /= static_cast<Value>(samples.size() * num_outputs);
        return {true, trial_error};
    };

    // ---- collect node IDs by category from a Graph ----
    auto collect_input_ids = [](Graph& g) -> std::vector<uint64_t> {
        std::vector<uint64_t> ids;
        for (auto& n : g.get_nodes()) {
            if (n->get_type() == NodeType::INPUT)
                ids.push_back(n->get_id());
        }
        return ids;
    };

    // ---- build pool for extra insertion inputs ----
    // (N's sources + non-IO-non-CONSTANT nodes + INPUT nodes, excluding N)
    auto build_fill_pool = [&](Graph& g, uint64_t exclude_id,
                                const std::vector<uint64_t>& source_ids) -> std::vector<uint64_t> {
        std::unordered_set<uint64_t> seen;
        std::vector<uint64_t> pool;
        for (auto sid : source_ids) {
            if (sid != exclude_id && seen.insert(sid).second)
                pool.push_back(sid);
        }
        for (auto& n : g.get_nodes()) {
            auto t = n->get_type();
            if (t == NodeType::INPUT) {
                uint64_t id = n->get_id();
                if (seen.insert(id).second) pool.push_back(id);
            } else if (t != NodeType::OUTPUT && t != NodeType::SINK && t != NodeType::CONSTANT) {
                uint64_t id = n->get_id();
                if (id != exclude_id && seen.insert(id).second) pool.push_back(id);
            }
        }
        return pool;
    };

    // ---- baseline clone ----
    auto baseline = clone();

    // ---- parallelization decision ----
    int num_blame_nodes = static_cast<int>(blameworthy.size());
    int num_threads = config::EVOLUTION_PARALLEL
        ? std::min(config::EVOLUTION_NUM_THREADS, num_blame_nodes)
        : 1;

    // ---- iterate blameworthy nodes ----
    if (num_threads <= 1) {
    for (const auto* attr : blameworthy) {
        uint64_t node_id = attr->node_id;
        Node* original = get_node(node_id);
        if (!original) continue;

        auto orig_type = original->get_type();
        size_t num_outputs = original->get_num_outputs();

        // Skip unmodifiable nodes
        if (orig_type == NodeType::INPUT  || orig_type == NodeType::OUTPUT ||
            orig_type == NodeType::SINK || num_outputs != 1)
            continue;

        // Record incoming / outgoing connections
        struct InInfo  { uint64_t src_node; size_t src_port; size_t dst_port; };
        struct OutInfo { size_t src_port; uint64_t dst_node; size_t dst_port; };
        std::vector<InInfo> incoming;
        std::vector<OutInfo> outgoing;
        for (const auto& c : connections_) {
            if (c.dst_node == node_id)
                incoming.push_back({c.src_node, c.src_port, c.dst_port});
            if (c.src_node == node_id)
                outgoing.push_back({c.src_port, c.dst_node, c.dst_port});
        }
        size_t source_n = incoming.size();

        // Collect source IDs for pool building
        std::vector<uint64_t> source_ids;
        for (const auto& inc : incoming) source_ids.push_back(inc.src_node);

        // ---- analytical pre-screen: score all candidate types via forward MSE,
        //      keep only top-3 for the expensive trial loops ----
        // execute baseline on one sample to collect this node's I/O values
        // (also used by INSERT pre-screen below)
        for (auto& n : baseline->get_nodes()) n->set_perturbation(0.0);
        for (auto& n : baseline->get_nodes()) {
            if (n->get_type() == NodeType::INPUT) {
                auto it = samples[0].inputs.find(n->get_id());
                if (it != samples[0].inputs.end())
                    baseline->set_input_value(n->get_id(), it->second);
            }
        }
        baseline->reset_recurrent_state();
        baseline->execute();
        Value node_out = baseline->get_any_node_output(node_id);
        std::vector<Value> node_ins;
        for (const auto& inc : incoming)
            node_ins.push_back(baseline->get_any_node_output(inc.src_node));

        std::vector<NodeType> top_candidates;
        {
            // score every (target_n, cand_type) by analytical MSE, keep best per type
            std::unordered_map<NodeType, double> type_score;
            for (size_t target_n = 1; target_n <= 3; ++target_n) {
                auto candidates = build_candidates(target_n);
                for (NodeType ct : candidates) {
                    if (ct == orig_type) continue;
                    size_t en = std::min(size_t(target_n), node_ins.size());
                    std::vector<Value> ein(node_ins.begin(), node_ins.begin() + en);
                    ein.resize(target_n, 0.0);
                    Value pred = eval_candidate(ct, ein);
                    double mse = (pred - node_out) * (pred - node_out);
                    auto it = type_score.find(ct);
                    if (it == type_score.end() || mse < it->second)
                        type_score[ct] = mse;
                }
            }
            // keep top-3 (or fewer if not enough candidates)
            std::vector<std::pair<NodeType, double>> ranked;
            for (const auto& [t, s] : type_score) ranked.push_back({t, s});
            std::sort(ranked.begin(), ranked.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
            for (size_t i = 0; i < std::min(size_t(3), ranked.size()); ++i)
                top_candidates.push_back(ranked[i].first);
        }

        // =============================================================
        // CONSTANT special case: wire each downstream to random INPUT
        // =============================================================
        if (orig_type == NodeType::CONSTANT) {
            if (outgoing.empty()) continue;

            std::vector<uint64_t> input_ids;
            for (auto& n : nodes_) {
                if (n->get_type() == NodeType::INPUT)
                    input_ids.push_back(n->get_id());
            }
            if (input_ids.empty()) continue;

            for (int t = 0; t < NUM_SUBSETS; ++t) {
                auto trial_ptr = baseline->clone();
                auto& trial = *trial_ptr;
                trial.remove_node(node_id);

                bool ok = true;
                for (const auto& out : outgoing) {
                    uint64_t rand_inp = input_ids[rand_idx(input_ids.size())];
                    if (!trial.add_connection(rand_inp, 0, out.dst_node, out.dst_port)) {
                        ok = false; break;
                    }
                }
                if (!ok) continue;

                auto [eval_ok, trial_error] = evaluate_trial(trial);
                if (!eval_ok) continue;

                NodeImprovementResult r;
                r.node_id        = node_id;
                r.operation      = SearchOperation::REPLACE;
                r.original_type  = orig_type;
                r.new_type       = NodeType::INPUT;
                r.original_error = baseline_error;
                r.new_error      = trial_error;
                r.improvement    = baseline_error - trial_error;
                results.push_back(r);
                if (best_trial && r.improvement > best_improvement) {
                    best_improvement = r.improvement;
                    *best_trial = trial.clone();
                }
            }
            continue;
        }

        if (source_n == 0) continue;

        // =============================================================
        // REPLACE �?try other node types with subset/fill strategy
        // =============================================================
        for (size_t target_n = 1; target_n <= 3; ++target_n) {
            auto candidates = build_candidates(target_n);
            for (NodeType cand_type : candidates) {
                if (cand_type == orig_type) continue;
                // skip candidates the analytical pre-screen ranked below top-3
                if (std::find(top_candidates.begin(), top_candidates.end(), cand_type) == top_candidates.end()) continue;

                if (source_n >= target_n) {
                    // ---- subset trials ----
                    for (int t = 0; t < NUM_SUBSETS; ++t) {
                        std::vector<InInfo> subset = incoming;
                        std::shuffle(subset.begin(), subset.end(), rng);
                        subset.resize(target_n);

                        auto trial_ptr = baseline->clone();
                        auto& trial = *trial_ptr;
                        trial.remove_node(node_id);
                        auto new_node = NodeRegistry::instance().create(cand_type, node_id, "");
                        if (!new_node) continue;
                        trial.add_node_existing(std::move(new_node));

                        for (const auto& inc : subset)
                            trial.add_connection(inc.src_node, inc.src_port, node_id, inc.dst_port);
                        for (const auto& out : outgoing)
                            trial.add_connection(node_id, out.src_port, out.dst_node, out.dst_port);

                        auto [ok, trial_error] = evaluate_trial(trial);
                        if (!ok) continue;

                        NodeImprovementResult r;
                        r.node_id        = node_id;
                        r.operation      = SearchOperation::REPLACE;
                        r.original_type  = orig_type;
                        r.new_type       = cand_type;
                        r.original_error = baseline_error;
                        r.new_error      = trial_error;
                        r.improvement    = baseline_error - trial_error;
                        results.push_back(r);
                        if (best_trial && r.improvement > best_improvement) {
                            best_improvement = r.improvement;
                            *best_trial = trial.clone();
                        }
                    }
                } else {
                    // ---- fill trials: strategy A (non-IO) + strategy B (INPUT) ----
                    size_t need_n = target_n - source_n;

                    // Pre-collect pools from baseline
                    std::vector<uint64_t> pool_a, pool_b;
                    for (auto& n : nodes_) {
                        auto t = n->get_type();
                        uint64_t id = n->get_id();
                        if (id == node_id) continue;
                        if (t == NodeType::INPUT) {
                            pool_b.push_back(id);
                        } else if (t != NodeType::OUTPUT && t != NodeType::SINK &&
                                   t != NodeType::CONSTANT) {
                            pool_a.push_back(id);
                        }
                    }

                    for (int strategy = 0; strategy < 2; ++strategy) {
                        const auto& fill_pool = (strategy == 0) ? pool_a : pool_b;
                        if (fill_pool.empty()) continue;

                        for (int t = 0; t < NUM_SUBSETS; ++t) {
                            auto trial_ptr = baseline->clone();
                            auto& trial = *trial_ptr;
                            trial.remove_node(node_id);
                            auto new_node = NodeRegistry::instance().create(cand_type, node_id, "");
                            if (!new_node) continue;
                            trial.add_node_existing(std::move(new_node));

                            for (const auto& inc : incoming)
                                trial.add_connection(inc.src_node, inc.src_port,
                                                     node_id, inc.dst_port);

                            for (size_t fi = 0; fi < need_n; ++fi) {
                                uint64_t fill_id = fill_pool[rand_idx(fill_pool.size())];
                                trial.add_connection(fill_id, 0, node_id,
                                                     incoming.size() + fi);
                            }

                            for (const auto& out : outgoing)
                                trial.add_connection(node_id, out.src_port, out.dst_node, out.dst_port);

                            auto [ok, trial_error] = evaluate_trial(trial);
                            if (!ok) continue;

                            NodeImprovementResult r;
                            r.node_id        = node_id;
                            r.operation      = SearchOperation::REPLACE;
                            r.original_type  = orig_type;
                            r.new_type       = cand_type;
                            r.original_error = baseline_error;
                            r.new_error      = trial_error;
                            r.improvement    = baseline_error - trial_error;
                            results.push_back(r);
                            if (best_trial && r.improvement > best_improvement) {
                                best_improvement = r.improvement;
                                *best_trial = trial.clone();
                            }
                        }
                    }
                }
            }
        }

        // =============================================================
        // INSERT - intercept connection near N (pre- or post-processing)
        // =============================================================

        // analytical pre-screen for INSERT: rank candidate types by how
        // non-disruptive they are when placed between N and downstream.
        std::vector<NodeType> insert_top_candidates;
        {
            std::unordered_map<NodeType, double> ins_type_score;
            for (size_t target_n = 1; target_n <= 3; ++target_n) {
                auto candidates = build_candidates(target_n);
                for (NodeType ct : candidates) {
                    if (ct == orig_type) continue;
                    std::vector<Value> ein;
                    ein.push_back(node_out);
                    while (ein.size() < target_n) ein.push_back(0.0);
                    Value pred = eval_candidate(ct, ein);
                    double mse = (pred - node_out) * (pred - node_out);
                    auto it = ins_type_score.find(ct);
                    if (it == ins_type_score.end() || mse < it->second)
                        ins_type_score[ct] = mse;
                }
            }
            std::vector<std::pair<NodeType, double>> ins_ranked;
            for (const auto& [t, s] : ins_type_score) ins_ranked.push_back({t, s});
            std::sort(ins_ranked.begin(), ins_ranked.end(),
                      [](const auto& a, const auto& b) { return a.second < b.second; });
            ins_ranked.push_back({orig_type, 0.0});  // always include self-type
            for (size_t i = 0; i < std::min(size_t(3), ins_ranked.size()); ++i)
                insert_top_candidates.push_back(ins_ranked[i].first);
        }

        for (size_t target_n = 1; target_n <= 3; ++target_n) {
            auto candidates = build_candidates(target_n);
            for (NodeType cand_type : candidates) {
                // skip candidates the analytical pre-screen ranked below top-3
                if (std::find(insert_top_candidates.begin(), insert_top_candidates.end(), cand_type) == insert_top_candidates.end()) continue;
                for (int t = 0; t < NUM_INSERT_TRIALS; ++t) {
                    bool is_post = rand_bool() && !outgoing.empty();
                    if (!is_post && incoming.empty()) continue;

                    auto trial_ptr = baseline->clone();
                    auto& trial = *trial_ptr;

                    std::vector<uint64_t> fill_pool = build_fill_pool(trial, node_id, source_ids);

                    if (is_post) {
                        // POST: intercept N �?downstream
                        const auto& out = outgoing[rand_idx(outgoing.size())];
                        trial.remove_connection(node_id, out.src_port,
                                                out.dst_node, out.dst_port);

                        uint64_t new_id = trial.add_node(cand_type, "");
                        trial.add_connection(node_id, 0, new_id, 0);
                        trial.add_connection(new_id, 0, out.dst_node, out.dst_port);

                        for (size_t pi = 1; pi < target_n && !fill_pool.empty(); ++pi)
                            trial.add_connection(fill_pool[rand_idx(fill_pool.size())], 0,
                                                 new_id, pi);
                    } else {
                        // PRE: intercept source �?N
                        const auto& inc = incoming[rand_idx(incoming.size())];
                        trial.remove_connection(inc.src_node, inc.src_port,
                                                node_id, inc.dst_port);

                        uint64_t new_id = trial.add_node(cand_type, "");
                        trial.add_connection(inc.src_node, inc.src_port, new_id, 0);
                        trial.add_connection(new_id, 0, node_id, inc.dst_port);

                        for (size_t pi = 1; pi < target_n && !fill_pool.empty(); ++pi)
                            trial.add_connection(fill_pool[rand_idx(fill_pool.size())], 0,
                                                 new_id, pi);
                    }

                    auto [ok, trial_error] = evaluate_trial(trial);
                    if (!ok) continue;

                    NodeImprovementResult r;
                    r.node_id        = node_id;
                    r.operation      = SearchOperation::INSERT;
                    r.original_type  = orig_type;
                    r.new_type       = cand_type;
                    r.original_error = baseline_error;
                    r.new_error      = trial_error;
                    r.improvement    = baseline_error - trial_error;
                    results.push_back(r);
                    if (best_trial && r.improvement > best_improvement) {
                        best_improvement = r.improvement;
                        *best_trial = trial.clone();
                    }
                }
            }
        }

        // =============================================================
        // REWIRE_INPUT �?try alternative source nodes for existing inputs
        // =============================================================
        {
            // Build candidate source pool: include all nodes except the
            // current blameworthy node (skip SINK since SINK has no output).
            // This includes INPUT, OUTPUT, CONSTANT, and all computation/
            // internal nodes.  More diverse than build_fill_pool which excludes
            // OUTPUT and CONSTANT.
            std::unordered_set<uint64_t> rew_seen;
            std::vector<uint64_t> rew_pool;
            for (auto& n : baseline->get_nodes()) {
                auto t = n->get_type();
                uint64_t id = n->get_id();
                if (id == node_id) continue;
                if (t == NodeType::SINK) continue;
                if (rew_seen.insert(id).second)
                    rew_pool.push_back(id);
            }
            if (rew_pool.empty()) continue;

            for (size_t pi = 0; pi < incoming.size(); ++pi) {
                uint64_t old_src = incoming[pi].src_node;

                for (int t = 0; t < NUM_SUBSETS; ++t) {
                    // Pick a random alternative source (not the original)
                    uint64_t new_src = old_src;
                    int attempts = 0;
                    while (new_src == old_src && attempts < config::SEARCH_REWIRE_MAX_ATTEMPTS) {
                        new_src = rew_pool[rand_idx(rew_pool.size())];
                        ++attempts;
                    }
                    if (new_src == old_src) continue;  // pool exhausted

                    auto trial_ptr = baseline->clone();
                    auto& trial = *trial_ptr;

                    // Rewire: remove old incoming connection, add from alternative
                    trial.remove_connection(old_src, incoming[pi].src_port,
                                            node_id, incoming[pi].dst_port);
                    if (!trial.add_connection(new_src, 0,
                                              node_id, incoming[pi].dst_port))
                        continue;

                    auto [ok, trial_error] = evaluate_trial(trial);
                    if (!ok) continue;

                    NodeImprovementResult r;
                    r.node_id        = node_id;
                    r.operation      = SearchOperation::REWIRE_INPUT;
                    r.original_type  = orig_type;
                    r.new_type       = orig_type;  // node type unchanged
                    r.original_error = baseline_error;
                    r.new_error      = trial_error;
                    r.improvement    = baseline_error - trial_error;
                    results.push_back(r);
                    if (best_trial && r.improvement > best_improvement) {
                        best_improvement = r.improvement;
                        *best_trial = trial.clone();
                    }
                }
            }
        }

        /*
         * =============================================================
         * Option 2 (future): Target-based internal node fitting
         * =============================================================
         *
         * The REWIRE_INPUT trial above is lightweight but blind — it tries
         * random alternative sources and evaluates on all samples.  A more
         * directed approach would derive approximate *target outputs* for
         * internal nodes, then use cheap node-level fitting as a pre-filter
         * before expensive graph-level evaluation:
         *
         * 1. Forward pass: record all intermediate activations.
         * 2. Backward pass: propagate output MSE through the graph using
         *    the chain-rule gradients that every node type already implements
         *    (backward_input_grads).  For an internal node N, accumulate
         *    the blamed gradient across all downstream paths to get dL/dy(N).
         * 3. Target derivation: compute a pseudo-target
         *      target(N) = y(N) - lr * dL/dy(N)
         *    where y(N) is N's current output and lr is a small step size.
         * 4. Node fitting (per-node, cheap): for each candidate node type,
         *    fit its parameters to approximate target(N) given N's inputs.
         *    This is O(num_candidates * node_eval) instead of O(graph_eval).
         * 5. Graph verification (expensive, but selective): only the
         *    top-K candidates that passed the cheap node-fitting filter are
         *    evaluated on the full graph with real data.
         *
         * This approach needs ~400-550 additional lines and raises challenges:
         *   - Fan-out: a single node drives multiple downstream paths; the
         *     blame accumulation must be convex-weighted by importance.
         *   - Non-invertibility: not all node types have an invertible
         *     activation, so the "pseudo-target" is a linear approximation
         *     that may not be achievable — but serves as a useful search bias.
         *   - If the node's inputs themselves are sub-optimal, the fitted
         *     target may be unattainable.  This is mitigated by step 5
         *     (graph-level verification rejects false positives).
         *
         * The existing blame framework (structure_grad_map_ and connection_structure_grad_map_)
         * provides dL/dy for output nodes; extending it to all internal nodes
         * via topological backward propagation is the main implementation
         * effort.
         */

        // =============================================================
        // DELETE �?remove node and bypass (cartesian product in×out)
        // =============================================================
        {
            auto trial_ptr = baseline->clone();
            auto& trial = *trial_ptr;
            trial.remove_node(node_id);

            for (const auto& inc : incoming) {
                for (const auto& out : outgoing) {
                    trial.add_connection(inc.src_node, inc.src_port,
                                         out.dst_node, out.dst_port);
                }
            }

            auto [ok, trial_error] = evaluate_trial(trial);
            if (ok) {
                NodeImprovementResult r;
                r.node_id        = node_id;
                r.operation      = SearchOperation::DELETE_BYPASS;
                r.original_type  = orig_type;
                r.new_type       = orig_type;
                r.original_error = baseline_error;
                r.new_error      = trial_error;
                r.improvement    = baseline_error - trial_error;
                results.push_back(r);
                if (best_trial && r.improvement > best_improvement) {
                    best_improvement = r.improvement;
                    *best_trial = trial.clone();
                }
            }
        }

        // =============================================================
        // REPLACE_INSERT �?replace N AND insert near N in same trial
        // =============================================================
        {
            for (size_t repl_tn = 1; repl_tn <= 3; ++repl_tn) {
                auto repl_cands = build_candidates(repl_tn);
                std::shuffle(repl_cands.begin(), repl_cands.end(), rng);
                for (size_t ri = 0; ri < static_cast<size_t>(NUM_SUBSETS) && ri < repl_cands.size(); ++ri) {
                    NodeType repl_type = repl_cands[ri];
                    if (repl_type == orig_type) continue;

                    for (size_t ins_tn = 1; ins_tn <= 3; ++ins_tn) {
                        auto ins_cands = build_candidates(ins_tn);
                        std::shuffle(ins_cands.begin(), ins_cands.end(), rng);
                        for (size_t ii = 0; ii < static_cast<size_t>(NUM_INSERT_TRIALS) && ii < ins_cands.size(); ++ii) {
                            NodeType ins_type = ins_cands[ii];
                            auto trial_ptr = baseline->clone();
                            auto& trial = *trial_ptr;

                            // --- replace node_id ---
                            trial.remove_node(node_id);
                            auto new_node = NodeRegistry::instance().create(repl_type, node_id, "");
                            if (!new_node) continue;
                            trial.add_node_existing(std::move(new_node));

                            if (source_n >= repl_tn) {
                                // subset wiring
                                std::vector<InInfo> subset = incoming;
                                std::shuffle(subset.begin(), subset.end(), rng);
                                subset.resize(repl_tn);
                                for (const auto& inc : subset)
                                    trial.add_connection(inc.src_node, inc.src_port,
                                                         node_id, inc.dst_port);
                            } else {
                                // fill wiring (INPUT pool)
                                for (const auto& inc : incoming)
                                    trial.add_connection(inc.src_node, inc.src_port,
                                                         node_id, inc.dst_port);
                                size_t need_n = repl_tn - source_n;
                                auto inp_ids = collect_input_ids(trial);
                                for (size_t fi = 0; fi < need_n && !inp_ids.empty(); ++fi)
                                    trial.add_connection(inp_ids[rand_idx(inp_ids.size())], 0,
                                                         node_id, source_n + fi);
                            }
                            for (const auto& out : outgoing)
                                trial.add_connection(node_id, out.src_port,
                                                     out.dst_node, out.dst_port);

                            // --- recalc incoming and outgoing from trial graph after replacement ---
                            std::vector<InInfo> trial_incoming;
                            std::vector<OutInfo> trial_outgoing;
                            for (const auto& c : trial.get_connections()) {
                                if (c.src_node == node_id)
                                    trial_outgoing.push_back({c.src_port, c.dst_node, c.dst_port});
                                if (c.dst_node == node_id)
                                    trial_incoming.push_back({c.src_node, c.src_port, c.dst_port});
                            }
                            if (trial_incoming.empty() || trial_outgoing.empty()) continue;

                            // PRE or POST mode (random, same as standalone INSERT)
                            bool is_post = rand_bool();
                            if (is_post) {
                                // POST: intercept node_id → downstream
                                const auto& out_conn = trial_outgoing[rand_idx(trial_outgoing.size())];
                                trial.remove_connection(node_id, out_conn.src_port,
                                                        out_conn.dst_node, out_conn.dst_port);
                                uint64_t new_id = trial.add_node(ins_type, "");
                                trial.add_connection(node_id, 0, new_id, 0);
                                trial.add_connection(new_id, 0, out_conn.dst_node, out_conn.dst_port);

                                std::vector<uint64_t> fill_pool =
                                    build_fill_pool(trial, node_id, source_ids);
                                for (size_t pi = 1; pi < ins_tn && !fill_pool.empty(); ++pi)
                                    trial.add_connection(fill_pool[rand_idx(fill_pool.size())], 0,
                                                         new_id, pi);
                            } else {
                                // PRE: intercept upstream → node_id
                                const auto& inc_conn = trial_incoming[rand_idx(trial_incoming.size())];
                                trial.remove_connection(inc_conn.src_node, inc_conn.src_port,
                                                        node_id, inc_conn.dst_port);
                                uint64_t new_id = trial.add_node(ins_type, "");
                                trial.add_connection(inc_conn.src_node, inc_conn.src_port, new_id, 0);
                                trial.add_connection(new_id, 0, node_id, inc_conn.dst_port);

                                std::vector<uint64_t> fill_pool =
                                    build_fill_pool(trial, node_id, source_ids);
                                for (size_t pi = 1; pi < ins_tn && !fill_pool.empty(); ++pi)
                                    trial.add_connection(fill_pool[rand_idx(fill_pool.size())], 0,
                                                         new_id, pi);
                            }

                            auto [ok, trial_error] = evaluate_trial(trial);
                            if (!ok) continue;

                            NodeImprovementResult r;
                            r.node_id        = node_id;
                            r.operation      = SearchOperation::REPLACE_INSERT;
                            r.original_type  = orig_type;
                            r.new_type       = repl_type;
                            r.original_error = baseline_error;
                            r.new_error      = trial_error;
                            r.improvement    = baseline_error - trial_error;
                            results.push_back(r);
                            if (best_trial && r.improvement > best_improvement) {
                                best_improvement = r.improvement;
                                *best_trial = trial.clone();
                            }
                        }
                    }
                }
            }
        }

        // =============================================================
        // DELETE_INSERT �?delete N AND insert near former position
        // =============================================================
        {
            for (size_t ins_tn = 1; ins_tn <= 3; ++ins_tn) {
                auto ins_cands = build_candidates(ins_tn);
                std::shuffle(ins_cands.begin(), ins_cands.end(), rng);
                for (size_t ii = 0; ii < static_cast<size_t>(NUM_INSERT_TRIALS) && ii < ins_cands.size(); ++ii) {
                    NodeType ins_type = ins_cands[ii];
                    if (incoming.empty() || outgoing.empty()) continue;

                    auto trial_ptr = baseline->clone();
                    auto& trial = *trial_ptr;
                    trial.remove_node(node_id);

                    // Choose src and dst before bypass so we skip this pair
                    const auto& src = incoming[rand_idx(incoming.size())];
                    const auto& dst = outgoing[rand_idx(outgoing.size())];
                    size_t src_idx = static_cast<size_t>(&src - incoming.data());
                    size_t dst_idx = static_cast<size_t>(&dst - outgoing.data());

                    // Bypass: cartesian product, skipping the (src, dst) pair
                    for (size_t inc_i = 0; inc_i < incoming.size(); ++inc_i) {
                        const auto& inc = incoming[inc_i];
                        for (size_t out_i = 0; out_i < outgoing.size(); ++out_i) {
                            const auto& out = outgoing[out_i];
                            if (inc_i == src_idx && out_i == dst_idx) continue;
                            trial.add_connection(inc.src_node, inc.src_port,
                                                 out.dst_node, out.dst_port);
                        }
                    }

                    // Wire src -> new_id -> dst directly (no remove_connection needed)
                    uint64_t new_id = trial.add_node(ins_type, "");
                    trial.add_connection(src.src_node, src.src_port, new_id, 0);
                    trial.add_connection(new_id, 0, dst.dst_node, dst.dst_port);

                    std::vector<uint64_t> fill_pool =
                        build_fill_pool(trial, node_id, source_ids);
                    for (size_t pi = 1; pi < ins_tn && !fill_pool.empty(); ++pi)
                        trial.add_connection(fill_pool[rand_idx(fill_pool.size())], 0,
                                             new_id, pi);

                    auto [ok, trial_error] = evaluate_trial(trial);
                    if (!ok) continue;

                    NodeImprovementResult r;
                    r.node_id        = node_id;
                    r.operation      = SearchOperation::DELETE_INSERT;
                    r.original_type  = orig_type;
                    r.new_type       = ins_type;
                    r.original_error = baseline_error;
                    r.new_error      = trial_error;
                    r.improvement    = baseline_error - trial_error;
                    results.push_back(r);
                    if (best_trial && r.improvement > best_improvement) {
                        best_improvement = r.improvement;
                        *best_trial = trial.clone();
                    }
                }
            }
        }
    }  // end for-loop (sequential)

    } else {
        // ====================================================================
        // PARALLEL PATH: distribute blameworthy nodes across threads
        // ====================================================================
        int chunk_size = (num_blame_nodes + num_threads - 1) / num_threads;

        std::vector<std::vector<NodeImprovementResult>> thread_results(num_threads);
        std::vector<double> thread_best_improvements(num_threads, 0.0);
        std::vector<std::unique_ptr<Graph>> thread_best_trials(num_threads);
        std::vector<std::thread> threads;

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                int start = t * chunk_size;
                int end   = std::min(start + chunk_size, num_blame_nodes);
                if (start >= end) return;

                std::mt19937 local_rng(seed + t * config::THREAD_RNG_SEED_MULTIPLIER);
                auto thread_baseline = baseline->clone();
                auto& tr = thread_results[t];
                auto& local_best_improvement = thread_best_improvements[t];
                auto& local_best_trial = thread_best_trials[t];

                auto local_rand_idx = [&local_rng](size_t n) -> size_t {
                    if (n == 0) return 0;
                    std::uniform_int_distribution<size_t> dist(0, n - 1);
                    return dist(local_rng);
                };
                auto local_rand_bool = [&local_rng]() -> bool {
                    std::uniform_int_distribution<int> dist(0, 1);
                    return dist(local_rng) == 0;
                };

                for (int bi = start; bi < end; ++bi) {
                    const auto* attr = blameworthy[bi];
                uint64_t node_id = attr->node_id;
                Node* original = get_node(node_id);
                if (!original) continue;

                auto orig_type = original->get_type();
                size_t num_outputs = original->get_num_outputs();

                // Skip unmodifiable nodes
                if (orig_type == NodeType::INPUT  || orig_type == NodeType::OUTPUT ||
                    orig_type == NodeType::SINK || num_outputs != 1)
                    continue;

                // Record incoming / outgoing connections
                struct InInfo  { uint64_t src_node; size_t src_port; size_t dst_port; };
                struct OutInfo { size_t src_port; uint64_t dst_node; size_t dst_port; };
                std::vector<InInfo> incoming;
                std::vector<OutInfo> outgoing;
                for (const auto& c : connections_) {
                    if (c.dst_node == node_id)
                        incoming.push_back({c.src_node, c.src_port, c.dst_port});
                    if (c.src_node == node_id)
                        outgoing.push_back({c.src_port, c.dst_node, c.dst_port});
                }
                size_t source_n = incoming.size();

                // Collect source IDs for pool building
                std::vector<uint64_t> source_ids;
                for (const auto& inc : incoming) source_ids.push_back(inc.src_node);

                // ---- analytical pre-screen: score all candidate types via forward MSE,
                //      keep only top-3 for the expensive trial loops ----
                for (auto& n : thread_baseline->get_nodes()) n->set_perturbation(0.0);
                for (auto& n : thread_baseline->get_nodes()) {
                    if (n->get_type() == NodeType::INPUT) {
                        auto it = samples[0].inputs.find(n->get_id());
                        if (it != samples[0].inputs.end())
                            thread_baseline->set_input_value(n->get_id(), it->second);
                    }
                }
                thread_baseline->reset_recurrent_state();
                thread_baseline->execute();
                Value node_out = thread_baseline->get_any_node_output(node_id);
                std::vector<Value> node_ins;
                for (const auto& inc : incoming)
                    node_ins.push_back(thread_baseline->get_any_node_output(inc.src_node));

                std::vector<NodeType> top_candidates;
                {
                    std::unordered_map<NodeType, double> type_score;
                    for (size_t target_n = 1; target_n <= 3; ++target_n) {
                        auto candidates = build_candidates(target_n);
                        for (NodeType ct : candidates) {
                            if (ct == orig_type) continue;
                            size_t en = std::min(size_t(target_n), node_ins.size());
                            std::vector<Value> ein(node_ins.begin(), node_ins.begin() + en);
                            ein.resize(target_n, 0.0);
                            Value pred = eval_candidate(ct, ein);
                            double mse = (pred - node_out) * (pred - node_out);
                            auto it = type_score.find(ct);
                            if (it == type_score.end() || mse < it->second)
                                type_score[ct] = mse;
                        }
                    }
                    std::vector<std::pair<NodeType, double>> ranked;
                    for (const auto& [t, s] : type_score) ranked.push_back({t, s});
                    std::sort(ranked.begin(), ranked.end(),
                              [](const auto& a, const auto& b) { return a.second < b.second; });
                    for (size_t i = 0; i < std::min(size_t(3), ranked.size()); ++i)
                        top_candidates.push_back(ranked[i].first);
                }

                // =============================================================
                // CONSTANT special case: wire each downstream to random INPUT
                // =============================================================
                if (orig_type == NodeType::CONSTANT) {
                    if (outgoing.empty()) continue;

                    std::vector<uint64_t> input_ids;
                    for (auto& n : nodes_) {
                        if (n->get_type() == NodeType::INPUT)
                            input_ids.push_back(n->get_id());
                    }
                    if (input_ids.empty()) continue;

                    for (int t = 0; t < NUM_SUBSETS; ++t) {
                        auto trial_ptr = thread_baseline->clone();
                        auto& trial = *trial_ptr;
                        trial.remove_node(node_id);

                        bool ok = true;
                        for (const auto& out : outgoing) {
                            uint64_t rand_inp = input_ids[local_rand_idx(input_ids.size())];
                            if (!trial.add_connection(rand_inp, 0, out.dst_node, out.dst_port)) {
                                ok = false; break;
                            }
                        }
                        if (!ok) continue;

                        auto [eval_ok, trial_error] = evaluate_trial(trial);
                        if (!eval_ok) continue;

                        NodeImprovementResult r;
                        r.node_id        = node_id;
                        r.operation      = SearchOperation::REPLACE;
                        r.original_type  = orig_type;
                        r.new_type       = NodeType::INPUT;
                        r.original_error = baseline_error;
                        r.new_error      = trial_error;
                        r.improvement    = baseline_error - trial_error;
                        tr.push_back(r);
                        if (r.improvement > local_best_improvement) {
                            local_best_improvement = r.improvement;
                            local_best_trial = trial.clone();
                        }
                    }
                    continue;
                }

                if (source_n == 0) continue;

                // =============================================================
                // REPLACE — try other node types with subset/fill strategy
                // =============================================================
                for (size_t target_n = 1; target_n <= 3; ++target_n) {
                    auto candidates = build_candidates(target_n);
                    for (NodeType cand_type : candidates) {
                        if (cand_type == orig_type) continue;
                        if (std::find(top_candidates.begin(), top_candidates.end(), cand_type) == top_candidates.end()) continue;

                        if (source_n >= target_n) {
                            // ---- subset trials ----
                            for (int t = 0; t < NUM_SUBSETS; ++t) {
                                std::vector<InInfo> subset = incoming;
                                std::shuffle(subset.begin(), subset.end(), local_rng);
                                subset.resize(target_n);

                                auto trial_ptr = thread_baseline->clone();
                                auto& trial = *trial_ptr;
                                trial.remove_node(node_id);
                                auto new_node = NodeRegistry::instance().create(cand_type, node_id, "");
                                if (!new_node) continue;
                                trial.add_node_existing(std::move(new_node));

                                for (const auto& inc : subset)
                                    trial.add_connection(inc.src_node, inc.src_port, node_id, inc.dst_port);
                                for (const auto& out : outgoing)
                                    trial.add_connection(node_id, out.src_port, out.dst_node, out.dst_port);

                                auto [ok, trial_error] = evaluate_trial(trial);
                                if (!ok) continue;

                                NodeImprovementResult r;
                                r.node_id        = node_id;
                                r.operation      = SearchOperation::REPLACE;
                                r.original_type  = orig_type;
                                r.new_type       = cand_type;
                                r.original_error = baseline_error;
                                r.new_error      = trial_error;
                                r.improvement    = baseline_error - trial_error;
                                tr.push_back(r);
                                if (r.improvement > local_best_improvement) {
                                    local_best_improvement = r.improvement;
                                    local_best_trial = trial.clone();
                                }
                            }
                        } else {
                            // ---- fill trials: strategy A (non-IO) + strategy B (INPUT) ----
                            size_t need_n = target_n - source_n;

                            std::vector<uint64_t> pool_a, pool_b;
                            for (auto& n : nodes_) {
                                auto t = n->get_type();
                                uint64_t id = n->get_id();
                                if (id == node_id) continue;
                                if (t == NodeType::INPUT) {
                                    pool_b.push_back(id);
                                } else if (t != NodeType::OUTPUT && t != NodeType::SINK &&
                                           t != NodeType::CONSTANT) {
                                    pool_a.push_back(id);
                                }
                            }

                            for (int strategy = 0; strategy < 2; ++strategy) {
                                const auto& fill_pool = (strategy == 0) ? pool_a : pool_b;
                                if (fill_pool.empty()) continue;

                                for (int t = 0; t < NUM_SUBSETS; ++t) {
                                    auto trial_ptr = thread_baseline->clone();
                                    auto& trial = *trial_ptr;
                                    trial.remove_node(node_id);
                                    auto new_node = NodeRegistry::instance().create(cand_type, node_id, "");
                                    if (!new_node) continue;
                                    trial.add_node_existing(std::move(new_node));

                                    for (const auto& inc : incoming)
                                        trial.add_connection(inc.src_node, inc.src_port,
                                                             node_id, inc.dst_port);

                                    for (size_t fi = 0; fi < need_n; ++fi) {
                                        uint64_t fill_id = fill_pool[local_rand_idx(fill_pool.size())];
                                        trial.add_connection(fill_id, 0, node_id,
                                                             incoming.size() + fi);
                                    }

                                    for (const auto& out : outgoing)
                                        trial.add_connection(node_id, out.src_port, out.dst_node, out.dst_port);

                                    auto [ok, trial_error] = evaluate_trial(trial);
                                    if (!ok) continue;

                                    NodeImprovementResult r;
                                    r.node_id        = node_id;
                                    r.operation      = SearchOperation::REPLACE;
                                    r.original_type  = orig_type;
                                    r.new_type       = cand_type;
                                    r.original_error = baseline_error;
                                    r.new_error      = trial_error;
                                    r.improvement    = baseline_error - trial_error;
                                    tr.push_back(r);
                                    if (r.improvement > local_best_improvement) {
                                        local_best_improvement = r.improvement;
                                        local_best_trial = trial.clone();
                                    }
                                }
                            }
                        }
                    }
                }

                // =============================================================
                // INSERT - intercept connection near N (pre- or post-processing)
                // =============================================================
                std::vector<NodeType> insert_top_candidates;
                {
                    std::unordered_map<NodeType, double> ins_type_score;
                    for (size_t target_n = 1; target_n <= 3; ++target_n) {
                        auto candidates = build_candidates(target_n);
                        for (NodeType ct : candidates) {
                            if (ct == orig_type) continue;
                            std::vector<Value> ein;
                            ein.push_back(node_out);
                            while (ein.size() < target_n) ein.push_back(0.0);
                            Value pred = eval_candidate(ct, ein);
                            double mse = (pred - node_out) * (pred - node_out);
                            auto it = ins_type_score.find(ct);
                            if (it == ins_type_score.end() || mse < it->second)
                                ins_type_score[ct] = mse;
                        }
                    }
                    std::vector<std::pair<NodeType, double>> ins_ranked;
                    for (const auto& [t, s] : ins_type_score) ins_ranked.push_back({t, s});
                    std::sort(ins_ranked.begin(), ins_ranked.end(),
                              [](const auto& a, const auto& b) { return a.second < b.second; });
                    ins_ranked.push_back({orig_type, 0.0});
                    for (size_t i = 0; i < std::min(size_t(3), ins_ranked.size()); ++i)
                        insert_top_candidates.push_back(ins_ranked[i].first);
                }

                for (size_t target_n = 1; target_n <= 3; ++target_n) {
                    auto candidates = build_candidates(target_n);
                    for (NodeType cand_type : candidates) {
                        if (std::find(insert_top_candidates.begin(), insert_top_candidates.end(), cand_type) == insert_top_candidates.end()) continue;
                        for (int t = 0; t < NUM_INSERT_TRIALS; ++t) {
                            bool is_post = local_rand_bool() && !outgoing.empty();
                            if (!is_post && incoming.empty()) continue;

                            auto trial_ptr = thread_baseline->clone();
                            auto& trial = *trial_ptr;

                            std::vector<uint64_t> fill_pool = build_fill_pool(trial, node_id, source_ids);

                            if (is_post) {
                                const auto& out = outgoing[local_rand_idx(outgoing.size())];
                                trial.remove_connection(node_id, out.src_port,
                                                        out.dst_node, out.dst_port);

                                uint64_t new_id = trial.add_node(cand_type, "");
                                trial.add_connection(node_id, 0, new_id, 0);
                                trial.add_connection(new_id, 0, out.dst_node, out.dst_port);

                                for (size_t pi = 1; pi < target_n && !fill_pool.empty(); ++pi)
                                    trial.add_connection(fill_pool[local_rand_idx(fill_pool.size())], 0,
                                                         new_id, pi);
                            } else {
                                const auto& inc = incoming[local_rand_idx(incoming.size())];
                                trial.remove_connection(inc.src_node, inc.src_port,
                                                        node_id, inc.dst_port);

                                uint64_t new_id = trial.add_node(cand_type, "");
                                trial.add_connection(inc.src_node, inc.src_port, new_id, 0);
                                trial.add_connection(new_id, 0, node_id, inc.dst_port);

                                for (size_t pi = 1; pi < target_n && !fill_pool.empty(); ++pi)
                                    trial.add_connection(fill_pool[local_rand_idx(fill_pool.size())], 0,
                                                         new_id, pi);
                            }

                            auto [ok, trial_error] = evaluate_trial(trial);
                            if (!ok) continue;

                            NodeImprovementResult r;
                            r.node_id        = node_id;
                            r.operation      = SearchOperation::INSERT;
                            r.original_type  = orig_type;
                            r.new_type       = cand_type;
                            r.original_error = baseline_error;
                            r.new_error      = trial_error;
                            r.improvement    = baseline_error - trial_error;
                            tr.push_back(r);
                            if (r.improvement > local_best_improvement) {
                                local_best_improvement = r.improvement;
                                local_best_trial = trial.clone();
                            }
                        }
                    }
                }

                // =============================================================
                // REWIRE_INPUT — try alternative source nodes for existing inputs
                // =============================================================
                {
                    std::unordered_set<uint64_t> rew_seen;
                    std::vector<uint64_t> rew_pool;
                    for (auto& n : thread_baseline->get_nodes()) {
                        auto t = n->get_type();
                        uint64_t id = n->get_id();
                        if (id == node_id) continue;
                        if (t == NodeType::SINK) continue;
                        if (rew_seen.insert(id).second)
                            rew_pool.push_back(id);
                    }
                    if (rew_pool.empty()) continue;

                    for (size_t pi = 0; pi < incoming.size(); ++pi) {
                        uint64_t old_src = incoming[pi].src_node;

                        for (int t = 0; t < NUM_SUBSETS; ++t) {
                            uint64_t new_src = old_src;
                            int attempts = 0;
                            while (new_src == old_src && attempts < config::SEARCH_REWIRE_MAX_ATTEMPTS) {
                                new_src = rew_pool[local_rand_idx(rew_pool.size())];
                                ++attempts;
                            }
                            if (new_src == old_src) continue;

                            auto trial_ptr = thread_baseline->clone();
                            auto& trial = *trial_ptr;

                            trial.remove_connection(old_src, incoming[pi].src_port,
                                                    node_id, incoming[pi].dst_port);
                            if (!trial.add_connection(new_src, 0,
                                                      node_id, incoming[pi].dst_port))
                                continue;

                            auto [ok, trial_error] = evaluate_trial(trial);
                            if (!ok) continue;

                            NodeImprovementResult r;
                            r.node_id        = node_id;
                            r.operation      = SearchOperation::REWIRE_INPUT;
                            r.original_type  = orig_type;
                            r.new_type       = orig_type;
                            r.original_error = baseline_error;
                            r.new_error      = trial_error;
                            r.improvement    = baseline_error - trial_error;
                            tr.push_back(r);
                            if (r.improvement > local_best_improvement) {
                                local_best_improvement = r.improvement;
                                local_best_trial = trial.clone();
                            }
                        }
                    }
                }

                // =============================================================
                // DELETE — remove node and bypass (cartesian product in×out)
                // =============================================================
                {
                    auto trial_ptr = thread_baseline->clone();
                    auto& trial = *trial_ptr;
                    trial.remove_node(node_id);

                    for (const auto& inc : incoming) {
                        for (const auto& out : outgoing) {
                            trial.add_connection(inc.src_node, inc.src_port,
                                                 out.dst_node, out.dst_port);
                        }
                    }

                    auto [ok, trial_error] = evaluate_trial(trial);
                    if (ok) {
                        NodeImprovementResult r;
                        r.node_id        = node_id;
                        r.operation      = SearchOperation::DELETE_BYPASS;
                        r.original_type  = orig_type;
                        r.new_type       = orig_type;
                        r.original_error = baseline_error;
                        r.new_error      = trial_error;
                        r.improvement    = baseline_error - trial_error;
                        tr.push_back(r);
                        if (r.improvement > local_best_improvement) {
                            local_best_improvement = r.improvement;
                            local_best_trial = trial.clone();
                        }
                    }
                }

                // =============================================================
                // REPLACE_INSERT — replace N AND insert near N in same trial
                // =============================================================
                {
                    for (size_t repl_tn = 1; repl_tn <= 3; ++repl_tn) {
                        auto repl_cands = build_candidates(repl_tn);
                        std::shuffle(repl_cands.begin(), repl_cands.end(), local_rng);
                        for (size_t ri = 0; ri < static_cast<size_t>(NUM_SUBSETS) && ri < repl_cands.size(); ++ri) {
                            NodeType repl_type = repl_cands[ri];
                            if (repl_type == orig_type) continue;

                            for (size_t ins_tn = 1; ins_tn <= 3; ++ins_tn) {
                                auto ins_cands = build_candidates(ins_tn);
                                std::shuffle(ins_cands.begin(), ins_cands.end(), local_rng);
                                for (size_t ii = 0; ii < static_cast<size_t>(NUM_INSERT_TRIALS) && ii < ins_cands.size(); ++ii) {
                                    NodeType ins_type = ins_cands[ii];
                                    auto trial_ptr = thread_baseline->clone();
                                    auto& trial = *trial_ptr;

                                    trial.remove_node(node_id);
                                    auto new_node = NodeRegistry::instance().create(repl_type, node_id, "");
                                    if (!new_node) continue;
                                    trial.add_node_existing(std::move(new_node));

                                    if (source_n >= repl_tn) {
                                        std::vector<InInfo> subset = incoming;
                                        std::shuffle(subset.begin(), subset.end(), local_rng);
                                        subset.resize(repl_tn);
                                        for (const auto& inc : subset)
                                            trial.add_connection(inc.src_node, inc.src_port,
                                                                 node_id, inc.dst_port);
                                    } else {
                                        for (const auto& inc : incoming)
                                            trial.add_connection(inc.src_node, inc.src_port,
                                                                 node_id, inc.dst_port);
                                        size_t need_n = repl_tn - source_n;
                                        auto inp_ids = collect_input_ids(trial);
                                        for (size_t fi = 0; fi < need_n && !inp_ids.empty(); ++fi)
                                            trial.add_connection(inp_ids[local_rand_idx(inp_ids.size())], 0,
                                                                 node_id, source_n + fi);
                                    }
                                    for (const auto& out : outgoing)
                                        trial.add_connection(node_id, out.src_port,
                                                             out.dst_node, out.dst_port);

                                    // recalc incoming and outgoing from trial graph after replacement
                                    std::vector<InInfo> trial_incoming;
                                    std::vector<OutInfo> trial_outgoing;
                                    for (const auto& c : trial.get_connections()) {
                                        if (c.src_node == node_id)
                                            trial_outgoing.push_back({c.src_port, c.dst_node, c.dst_port});
                                        if (c.dst_node == node_id)
                                            trial_incoming.push_back({c.src_node, c.src_port, c.dst_port});
                                    }
                                    if (trial_incoming.empty() || trial_outgoing.empty()) continue;

                                    bool is_post = local_rand_bool();
                                    if (is_post) {
                                        const auto& out_conn = trial_outgoing[local_rand_idx(trial_outgoing.size())];
                                        trial.remove_connection(node_id, out_conn.src_port,
                                                                out_conn.dst_node, out_conn.dst_port);
                                        uint64_t new_id = trial.add_node(ins_type, "");
                                        trial.add_connection(node_id, 0, new_id, 0);
                                        trial.add_connection(new_id, 0, out_conn.dst_node, out_conn.dst_port);

                                        std::vector<uint64_t> fill_pool =
                                            build_fill_pool(trial, node_id, source_ids);
                                        for (size_t pi = 1; pi < ins_tn && !fill_pool.empty(); ++pi)
                                            trial.add_connection(fill_pool[local_rand_idx(fill_pool.size())], 0,
                                                                 new_id, pi);
                                    } else {
                                        const auto& inc_conn = trial_incoming[local_rand_idx(trial_incoming.size())];
                                        trial.remove_connection(inc_conn.src_node, inc_conn.src_port,
                                                                node_id, inc_conn.dst_port);
                                        uint64_t new_id = trial.add_node(ins_type, "");
                                        trial.add_connection(inc_conn.src_node, inc_conn.src_port, new_id, 0);
                                        trial.add_connection(new_id, 0, node_id, inc_conn.dst_port);

                                        std::vector<uint64_t> fill_pool =
                                            build_fill_pool(trial, node_id, source_ids);
                                        for (size_t pi = 1; pi < ins_tn && !fill_pool.empty(); ++pi)
                                            trial.add_connection(fill_pool[local_rand_idx(fill_pool.size())], 0,
                                                                 new_id, pi);
                                    }

                                    auto [ok, trial_error] = evaluate_trial(trial);
                                    if (!ok) continue;

                                    NodeImprovementResult r;
                                    r.node_id        = node_id;
                                    r.operation      = SearchOperation::REPLACE_INSERT;
                                    r.original_type  = orig_type;
                                    r.new_type       = repl_type;
                                    r.original_error = baseline_error;
                                    r.new_error      = trial_error;
                                    r.improvement    = baseline_error - trial_error;
                                    tr.push_back(r);
                                    if (r.improvement > local_best_improvement) {
                                        local_best_improvement = r.improvement;
                                        local_best_trial = trial.clone();
                                    }
                                }
                            }
                        }
                    }
                }

                // =============================================================
                // DELETE_INSERT — delete N AND insert near former position
                // =============================================================
                {
                    for (size_t ins_tn = 1; ins_tn <= 3; ++ins_tn) {
                        auto ins_cands = build_candidates(ins_tn);
                        std::shuffle(ins_cands.begin(), ins_cands.end(), local_rng);
                        for (size_t ii = 0; ii < static_cast<size_t>(NUM_INSERT_TRIALS) && ii < ins_cands.size(); ++ii) {
                            NodeType ins_type = ins_cands[ii];
                            if (incoming.empty() || outgoing.empty()) continue;

                            auto trial_ptr = thread_baseline->clone();
                            auto& trial = *trial_ptr;
                            trial.remove_node(node_id);

                            const auto& src = incoming[local_rand_idx(incoming.size())];
                            const auto& dst = outgoing[local_rand_idx(outgoing.size())];
                            size_t src_idx = static_cast<size_t>(&src - incoming.data());
                            size_t dst_idx = static_cast<size_t>(&dst - outgoing.data());

                            for (size_t inc_i = 0; inc_i < incoming.size(); ++inc_i) {
                                const auto& inc = incoming[inc_i];
                                for (size_t out_i = 0; out_i < outgoing.size(); ++out_i) {
                                    const auto& out = outgoing[out_i];
                                    if (inc_i == src_idx && out_i == dst_idx) continue;
                                    trial.add_connection(inc.src_node, inc.src_port,
                                                         out.dst_node, out.dst_port);
                                }
                            }

                            uint64_t new_id = trial.add_node(ins_type, "");
                            trial.add_connection(src.src_node, src.src_port, new_id, 0);
                            trial.add_connection(new_id, 0, dst.dst_node, dst.dst_port);

                            std::vector<uint64_t> fill_pool =
                                build_fill_pool(trial, node_id, source_ids);
                            for (size_t pi = 1; pi < ins_tn && !fill_pool.empty(); ++pi)
                                trial.add_connection(fill_pool[local_rand_idx(fill_pool.size())], 0,
                                                     new_id, pi);

                            auto [ok, trial_error] = evaluate_trial(trial);
                            if (!ok) continue;

                            NodeImprovementResult r;
                            r.node_id        = node_id;
                            r.operation      = SearchOperation::DELETE_INSERT;
                            r.original_type  = orig_type;
                            r.new_type       = ins_type;
                            r.original_error = baseline_error;
                            r.new_error      = trial_error;
                            r.improvement    = baseline_error - trial_error;
                            tr.push_back(r);
                            if (r.improvement > local_best_improvement) {
                                local_best_improvement = r.improvement;
                                local_best_trial = trial.clone();
                            }
                        }
                    }
                }
                }  // for bi
            });
        }

        for (auto& th : threads) th.join();

        // Merge all thread results into main results, pick best trial
        for (int t = 0; t < num_threads; ++t) {
            auto& tr = thread_results[t];
            results.insert(results.end(),
                std::make_move_iterator(tr.begin()),
                std::make_move_iterator(tr.end()));
            if (best_trial && thread_best_improvements[t] > best_improvement) {
                best_improvement = thread_best_improvements[t];
                *best_trial = std::move(thread_best_trials[t]);
            }
        }
    }  // end else

    // Sort by improvement descending (best improvements first)
    std::sort(results.begin(), results.end(),
              [](const NodeImprovementResult& a, const NodeImprovementResult& b) {
                  return a.improvement > b.improvement;
              });

    return results;
}

// ============================================================================
// Structural hashing (for duplicate detection)
// ============================================================================
size_t Graph::compute_structural_hash() const {
    if (!hash_dirty_) return cached_hash_;

    size_t h = 0;
    auto hash_combine = [](size_t& seed, size_t val) {
        seed ^= val + config::STRUCTURAL_HASH_GOLDEN_RATIO + (seed << 6) + (seed >> 2);
    };

    // Hash node types and value-carrying state in stored order
    // (node order is stable across clone_graph)
    for (const auto& n : nodes_) {
        hash_combine(h, static_cast<size_t>(n->get_type()));

        if (n->get_type() == NodeType::CONSTANT) {
            auto* cn = dynamic_cast<const ConstantNode*>(n.get());
            if (cn) {
                hash_combine(h, std::hash<double>{}(cn->get_value()));
            }
        } else if (n->get_type() == NodeType::NEURON) {
            auto* nn = dynamic_cast<const NeuronNode*>(n.get());
            if (nn) {
                hash_combine(h, std::hash<double>{}(nn->get_bias()));
                for (auto w : nn->get_weights()) {
                    hash_combine(h, std::hash<double>{}(w));
                }
            }
        }
    }

    // Hash connections in stored order
    for (const auto& c : connections_) {
        hash_combine(h, static_cast<size_t>(c.src_node));
        hash_combine(h, static_cast<size_t>(c.src_port));
        hash_combine(h, static_cast<size_t>(c.dst_node));
        hash_combine(h, static_cast<size_t>(c.dst_port));
    }

    cached_hash_ = h;
    hash_dirty_ = false;
    return h;
}

// ============================================================================
// Constant output detection (for targeted mutation)
// ============================================================================
bool Graph::is_output_constant(const std::vector<double>& sigmoid_outputs) {
    if (sigmoid_outputs.empty()) return false;

    // Check 1: same sign — all predictions on same side of 0.5
    bool all_above = true, all_below = true;
    double mean = 0.0;
    for (double p : sigmoid_outputs) {
        if (p < config::CONSTANT_OUTPUT_SAME_SIGN_THRESHOLD) all_above = false;
        if (p > config::CONSTANT_OUTPUT_SAME_SIGN_THRESHOLD) all_below = false;
        mean += p;
    }
    if (all_above || all_below) return true;

    // Check 2: variance below threshold
    mean /= static_cast<double>(sigmoid_outputs.size());
    double var = 0.0;
    for (double p : sigmoid_outputs) {
        double d = p - mean;
        var += d * d;
    }
    var /= static_cast<double>(sigmoid_outputs.size());
    return var < config::CONSTANT_OUTPUT_VARIANCE_THRESHOLD;
}

void Graph::set_constant_outputs(const std::vector<uint64_t>& outputs) {
    constant_outputs_.clear();
    constant_outputs_.insert(outputs.begin(), outputs.end());
}

std::vector<uint64_t> Graph::get_ancestors(uint64_t node_id) const {
    ensure_caches();
    std::vector<uint64_t> result;
    std::unordered_set<uint64_t> visited;

    std::function<void(uint64_t)> dfs = [&](uint64_t nid) {
        if (!visited.insert(nid).second) return;
        result.push_back(nid);
        auto it = rev_adj_.find(nid);
        if (it != rev_adj_.end()) {
            for (const auto* conn : it->second) {
                dfs(conn->src_node);
            }
        }
    };

    dfs(node_id);
    return result;
}

// ============================================================================
// Subgraph extraction — used by Building Block Library
// ============================================================================
std::unique_ptr<Graph> Graph::extract_subgraph(uint64_t output_node_id,
                                               std::vector<uint64_t>& out_input_ids) const {
    auto ancestors = get_ancestors(output_node_id);
    std::unordered_set<uint64_t> ancestor_set(ancestors.begin(), ancestors.end());

    // Collect INPUT node IDs and internal nodes
    out_input_ids.clear();
    std::vector<uint64_t> internal_ids;
    for (uint64_t id : ancestors) {
        const Node* n = get_node(id);
        if (!n) continue;
        if (n->get_type() == NodeType::INPUT) {
            out_input_ids.push_back(id);
        } else if (n->get_type() != NodeType::OUTPUT) {
            internal_ids.push_back(id);
        }
    }

    // Build the subgraph
    auto sub = std::make_unique<Graph>();
    sub->set_next_id(next_id_);

    // Clone the OUTPUT node
    const Node* out_node = get_node(output_node_id);
    if (out_node) {
        sub->add_node_existing(out_node->clone());
    }

    // Clone all INPUT nodes
    for (uint64_t id : out_input_ids) {
        const Node* n = get_node(id);
        if (n) {
            sub->add_node_existing(n->clone());
        }
    }

    // Clone all internal (non-INPUT, non-OUTPUT) nodes
    for (uint64_t id : internal_ids) {
        const Node* n = get_node(id);
        if (n) {
            sub->add_node_existing(n->clone());
        }
    }

    // Copy connections where both src and dst are in the ancestor set
    for (const auto& conn : connections_) {
        if (ancestor_set.count(conn.src_node) && ancestor_set.count(conn.dst_node)) {
            sub->add_connection_existing(conn);
        }
    }

    return sub;
}

// ============================================================================
// Size diagnostics
// ============================================================================
std::vector<uint64_t> Graph::compute_reachable_ids() const {
    ensure_caches();
    std::unordered_set<uint64_t> reachable;

    // Start from OUTPUT nodes
    for (const auto& n : nodes_) {
        if (n->get_type() == NodeType::OUTPUT) {
            reachable.insert(n->get_id());
        }
    }
    if (reachable.empty()) return {};

    // BFS backwards through reverse adjacency
    std::vector<uint64_t> frontier(reachable.begin(), reachable.end());
    size_t head = 0;
    while (head < frontier.size()) {
        uint64_t current = frontier[head++];
        auto it = rev_adj_.find(current);
        if (it == rev_adj_.end()) continue;
        for (const auto* conn : it->second) {
            if (reachable.insert(conn->src_node).second) {
                frontier.push_back(conn->src_node);
            }
        }
    }

    return std::vector<uint64_t>(reachable.begin(), reachable.end());
}

double Graph::compute_utilization_ratio() const {
    if (nodes_.empty()) return 1.0;
    auto reachable = compute_reachable_ids();
    return static_cast<double>(reachable.size()) / static_cast<double>(nodes_.size());
}

double Graph::compute_saturation_ratio() const {
    ensure_caches();

    int total_internal = 0;
    int saturated_count = 0;

    for (const auto& n : nodes_) {
        auto t = n->get_type();
        // Skip I/O nodes — they don't represent graph capacity
        if (t == NodeType::INPUT || t == NodeType::OUTPUT) continue;

        int num_inputs = static_cast<int>(n->get_num_inputs());
        if (num_inputs == 0) continue;

        total_internal++;

        // Count how many of this node's input ports have incoming connections
        std::set<size_t> connected_ports;
        auto it = rev_adj_.find(n->get_id());
        if (it != rev_adj_.end()) {
            for (const auto* conn : it->second) {
                connected_ports.insert(conn->dst_port);
            }
        }

        if (static_cast<int>(connected_ports.size()) >= num_inputs) {
            saturated_count++;
        }
    }

    if (total_internal == 0) return 0.0;
    return static_cast<double>(saturated_count) / static_cast<double>(total_internal);
}

// ============================================================================
// Connectivity diagnostics — check if OUTPUT nodes have paths from INPUT/CONSTANT
// ============================================================================
bool Graph::has_input_to_output_path() const {
    ensure_caches();

    // Collect source nodes (INPUT and CONSTANT — data originates here)
    std::vector<uint64_t> sources;
    for (const auto& n : nodes_) {
        auto t = n->get_type();
        if (t == NodeType::INPUT || t == NodeType::CONSTANT) {
            sources.push_back(n->get_id());
        }
    }
    if (sources.empty()) return false;

    // BFS forward from all source nodes along connections
    std::unordered_set<uint64_t> reachable;
    std::vector<uint64_t> frontier(sources.begin(), sources.end());
    reachable.insert(frontier.begin(), frontier.end());

    size_t head = 0;
    while (head < frontier.size()) {
        uint64_t current = frontier[head++];
        auto it = src_adj_.find(current);
        if (it == src_adj_.end()) continue;
        for (const auto* conn : it->second) {
            if (reachable.insert(conn->dst_node).second) {
                frontier.push_back(conn->dst_node);
            }
        }
    }

    // Every OUTPUT must be reachable from some source
    for (const auto& n : nodes_) {
        if (n->get_type() == NodeType::OUTPUT) {
            if (reachable.find(n->get_id()) == reachable.end())
                return false;
        }
    }
    return true;
}

int Graph::count_reachable_outputs() const {
    ensure_caches();

    // Collect source nodes
    std::vector<uint64_t> sources;
    for (const auto& n : nodes_) {
        auto t = n->get_type();
        if (t == NodeType::INPUT || t == NodeType::CONSTANT) {
            sources.push_back(n->get_id());
        }
    }
    if (sources.empty()) return 0;

    // BFS forward
    std::unordered_set<uint64_t> reachable;
    std::vector<uint64_t> frontier(sources.begin(), sources.end());
    reachable.insert(frontier.begin(), frontier.end());

    size_t head = 0;
    while (head < frontier.size()) {
        uint64_t current = frontier[head++];
        auto it = src_adj_.find(current);
        if (it == src_adj_.end()) continue;
        for (const auto* conn : it->second) {
            if (reachable.insert(conn->dst_node).second) {
                frontier.push_back(conn->dst_node);
            }
        }
    }

    int count = 0;
    for (const auto& n : nodes_) {
        if (n->get_type() == NodeType::OUTPUT) {
            if (reachable.find(n->get_id()) != reachable.end())
                count++;
        }
    }
    return count;
}

// ============================================================================
// to_expression — compile graph DAG to a readable math+ternary formula
// ============================================================================
std::string Graph::to_expression(uint64_t output_node_id) const {
    // Name INPUT nodes x0, x1, ...
    std::unordered_map<uint64_t, std::string> input_names;
    size_t idx = 0;
    for (const auto& n : nodes_) {
        if (n->get_type() == NodeType::INPUT)
            input_names[n->get_id()] = "x" + std::to_string(idx++);
    }

    // Memo: node_id → expression string (for single-output nodes)
    std::unordered_map<uint64_t, std::string> memo;

    // Recursive expression builder
    std::function<std::string(uint64_t, size_t)> build =
        [&](uint64_t node_id, size_t port) -> std::string {

        // For IFELSE (2 outputs), don't memo — different ports give different exprs
        const Node* n = get_node(node_id);
        if (!n) return "0";
        bool multi_out = (n->get_type() == NodeType::IFELSE);
        if (!multi_out) {
            auto it = memo.find(node_id);
            if (it != memo.end()) return it->second;
        }

        // Helper: expression of the source feeding input port i
        auto in_expr = [&](size_t i) -> std::string {
            for (const auto& c : connections_) {
                if (c.dst_node == node_id && c.dst_port == i
                    && !c.is_recurrent)  // skip recurrent (can't represent inline)
                    return build(c.src_node, c.src_port);
            }
            return "0";
        };

        auto fmt_num = [](double v) -> std::string {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << v;
            return ss.str();
        };

        std::string result;
        switch (n->get_type()) {
            case NodeType::INPUT:
                result = input_names.count(node_id) ? input_names[node_id] : "x?";
                break;
            case NodeType::CONSTANT:
                result = fmt_num(static_cast<const ConstantNode*>(n)->get_value());
                break;
            case NodeType::NEURON: {
                auto* nn = static_cast<const NeuronNode*>(n);
                std::ostringstream ss;
                ss << "tanh(";
                bool first = true;
                for (size_t i = 0; i < nn->get_num_weights(); ++i) {
                    double w = nn->get_weight(i);
                    if (std::abs(w) < 1e-8) continue;
                    if (!first && w >= 0) ss << "+";
                    ss << std::fixed << std::setprecision(3) << w << "*" << in_expr(i);
                    first = false;
                }
                double b = nn->get_bias();
                if (std::abs(b) > 1e-8) {
                    if (!first && b >= 0) ss << "+";
                    ss << std::fixed << std::setprecision(3) << b;
                }
                if (first) ss << "0";
                ss << ")";
                result = ss.str();
                break;
            }
            case NodeType::LINEAR: {
                // Same as NEURON but without the tanh() wrapper
                auto* ln = static_cast<const NeuronNode*>(n);
                std::ostringstream ss;
                ss << "(";
                bool first = true;
                for (size_t i = 0; i < ln->get_num_weights(); ++i) {
                    double w = ln->get_weight(i);
                    if (std::abs(w) < 1e-8) continue;
                    if (!first && w >= 0) ss << "+";
                    ss << std::fixed << std::setprecision(3) << w << "*" << in_expr(i);
                    first = false;
                }
                double b = ln->get_bias();
                if (std::abs(b) > 1e-8) {
                    if (!first && b >= 0) ss << "+";
                    ss << std::fixed << std::setprecision(3) << b;
                }
                if (first) ss << "0";
                ss << ")";
                result = ss.str();
                break;
            }
            case NodeType::OUTPUT: {
                auto* on = static_cast<const OutputNode*>(n);
                double sc = on->get_scale(), bi = on->get_bias();
                std::ostringstream ss;
                if (std::abs(sc - 1.0) > 1e-8)
                    ss << std::fixed << std::setprecision(3) << sc << "*";
                ss << in_expr(0);
                if (std::abs(bi) > 1e-8) {
                    if (bi >= 0) ss << "+";
                    ss << std::fixed << std::setprecision(3) << bi;
                }
                result = ss.str();
                break;
            }
            case NodeType::ADD:    result = "(" + in_expr(0) + "+" + in_expr(1) + ")"; break;
            case NodeType::SUBTRACT: result = "(" + in_expr(0) + "-" + in_expr(1) + ")"; break;
            case NodeType::MULTIPLY: result = "(" + in_expr(0) + "*" + in_expr(1) + ")"; break;
            case NodeType::DIVIDE:   result = "(" + in_expr(0) + "/" + in_expr(1) + ")"; break;
            case NodeType::NEGATE:   result = "(-" + in_expr(0) + ")"; break;
            case NodeType::TANH:     result = "tanh(" + in_expr(0) + ")"; break;
            case NodeType::SIN:      result = "sin(" + in_expr(0) + ")"; break;
            case NodeType::SIGMOID:  result = "sigmoid(" + in_expr(0) + ")"; break;
            case NodeType::RELU:     result = "relu(" + in_expr(0) + ")"; break;
            case NodeType::GREATER:  result = "(" + in_expr(0) + ">" + in_expr(1) + "?1:0)"; break;
            case NodeType::LESS:     result = "(" + in_expr(0) + "<" + in_expr(1) + "?1:0)"; break;
            case NodeType::GREATER_EQUAL: result = "(" + in_expr(0) + ">=" + in_expr(1) + "?1:0)"; break;
            case NodeType::LESS_EQUAL:    result = "(" + in_expr(0) + "<=" + in_expr(1) + "?1:0)"; break;
            case NodeType::EQUAL:    result = "(" + in_expr(0) + "==" + in_expr(1) + "?1:0)"; break;
            case NodeType::NOT_EQUAL: result = "(" + in_expr(0) + "!=" + in_expr(1) + "?1:0)"; break;
            case NodeType::AND: result = "(" + in_expr(0) + "&&" + in_expr(1) + "?1:0)"; break;
            case NodeType::OR:  result = "(" + in_expr(0) + "||" + in_expr(1) + "?1:0)"; break;
            case NodeType::XOR: result = "(" + in_expr(0) + "!=" + in_expr(1) + "?1:0)"; break;
            case NodeType::IFELSE:
                if (port == 0) result = "(" + in_expr(0) + "?" + in_expr(1) + ":0)";
                else           result = "(" + in_expr(0) + "?0:" + in_expr(1) + ")";
                break;
            case NodeType::SINK:   result = "0"; break;
            case NodeType::ABSENT: result = "absent(" + in_expr(0) + ")"; break;
            default: result = "?";
        }

        if (!multi_out) memo[node_id] = result;
        return result;
    };

    return build(output_node_id, 0);
}

} // namespace aria
