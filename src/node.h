#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cmath>

namespace aria {

// ============================================================================
// Value type used throughout the graph
// ============================================================================
using Value = double;

// ============================================================================
// Node types 鈥?add new types here for extensibility
// ============================================================================
enum class NodeType : uint32_t {
    // IO
    INPUT,
    OUTPUT,
    CONSTANT,
    SINK,

    // Arithmetic
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    NEGATE,

    // Neural network
    NEURON,
    RELU,
    SIGMOID,
    TANH,

    SIN,     // 1 input, 1 output 鈥?sin(x) activation

    LINEAR,  // N inputs, 1 output 鈥?identity: w路x + b (no activation, for BCE)

    // Control flow
    IF,      // 2 inputs (condition, value), 1 output 鈥?false 鈫?output 0 (sink-like)
    IFELSE,  // 2 inputs (condition, value), 2 outputs 鈥?true-out / false-out
    MUX,     // 3 inputs (condition, a, b), 1 output — a if cond else b (inverse of IFELSE)

    // Logic & Comparison
    EQUAL,         // 2 inputs, 1 output 鈥?1 if input0 == input1, else 0
    NOT_EQUAL,     // 2 inputs, 1 output 鈥?1 if input0 != input1, else 0
    GREATER,       // 2 inputs, 1 output 鈥?1 if input0 >  input1, else 0
    LESS,          // 2 inputs, 1 output 鈥?1 if input0 <  input1, else 0
    GREATER_EQUAL, // 2 inputs, 1 output 鈥?1 if input0 >= input1, else 0
    LESS_EQUAL,    // 2 inputs, 1 output 鈥?1 if input0 <= input1, else 0
    AND,           // 2 inputs, 1 output 鈥?1 if both non-zero,  else 0
    OR,            // 2 inputs, 1 output 鈥?1 if either non-zero, else 0
    NOT,           // 1 input,  1 output 鈥?1 if input == 0,    else 0
    XOR,           // 2 inputs, 1 output 鈥?1 if boolean values differ, else 0

    // Meta / topology sensing
            ABSENT,        // 1 input,  1 output 鈥?1 if connected but no signal, else 0
};

// Convert NodeType to/from string (for serialization)
const char* node_type_to_string(NodeType type);
NodeType node_type_from_string(const std::string& str);

// ============================================================================
// Base Node class
// ============================================================================
class Node {
public:
    Node(uint64_t id, NodeType type, const std::string& name,
         size_t num_inputs, size_t num_outputs);
    virtual ~Node() = default;

    // Identity
    uint64_t get_id() const { return id_; }
    void set_id(uint64_t id) { id_ = id; }
    NodeType get_type() const { return type_; }
    const std::string& get_name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    // I/O count
    size_t get_num_inputs() const { return inputs_.size(); }
    size_t get_num_outputs() const { return outputs_.size(); }
    virtual size_t get_min_inputs() const = 0;

    // Input/output access
    void set_input(size_t index, Value value);
    Value get_input(size_t index) const;
    Value get_output(size_t index) const;
    void set_output(size_t index, Value val) { outputs_[index] = val; }
    bool is_input_filled(size_t index) const;

    // Perturbation 鈥?for error-attribution analysis.
    // When non-zero, Graph::execute() adds this to each output after the node executes.
    void set_perturbation(Value p) { perturbation_ = p; }
    Value get_perturbation() const { return perturbation_; }

    // Structural connection tracking (set during graph construction, NOT reset)
    bool is_input_connected(size_t index) const;
    void mark_input_connected(size_t index);
    void clear_input_connected(size_t index);

    // Check if all inputs have been written (at least once)
    bool has_all_inputs() const;

    // Dirty flag for execution tracking
    bool is_dirty() const { return dirty_; }
    void clear_dirty() { dirty_ = false; }
    void mark_dirty() { dirty_ = true; }

    // Reset for next execution pass
    virtual void reset();

    // Execute 鈥?compute outputs from inputs
    virtual void execute() = 0;

    // Deep-copy this node (including node-specific state like weights/bias/value).
    virtual std::unique_ptr<Node> clone() const = 0;

    // Copy node-specific state (weights, bias, constants) to another node
    // Used during crossover to preserve learned parameters
    virtual void copy_state_to(Node* /*target*/) const {}

    // Copy runtime output vector from another node.
    // Enables incremental evaluation: after Graph::clone(), output caches are
    // preserved; mutations mark only affected downstream nodes dirty.
    void copy_outputs_from(const Node& src) { outputs_ = src.outputs_; }

    // Gradient / importance tracking (backpropagation-inspired)
    // Returns gradients for each input port given the gradient of the output.
    virtual std::vector<Value> backward_input_grads(Value output_grad);
    void accumulate_grad(Value g) { grad_ += g; }
    void zero_grad() { grad_ = 0.0; }
    Value get_grad() const { return grad_; }

    // Serialization helpers
    virtual void serialize_extra(std::string& /*json*/) const {}
    virtual void deserialize_extra(const std::string& /*key*/, const std::string& /*value*/) {}

protected:
    uint64_t id_;
    NodeType type_;
    std::string name_;
    std::vector<Value> inputs_;
    std::vector<Value> outputs_;
    std::vector<bool> input_filled_;
    std::vector<bool> input_connected_;
    bool dirty_ = true;
    Value grad_ = 0.0;
    Value perturbation_ = 0.0;
};

// ============================================================================
// IO Nodes
// ============================================================================

class InputNode : public Node {
public:
    InputNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 0; }
    void set_value(Value value);
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
    void copy_state_to(Node* target) const override;
};

class OutputNode : public Node {
public:
    OutputNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }
    Value get_value() const;
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;

    // Trainable output scale/bias (output = scale * input + bias)
    void set_scale(Value s) { scale_ = s; }
    Value get_scale() const { return scale_; }
    void set_bias(Value b) { bias_ = b; }
    Value get_bias() const { return bias_; }

    void copy_state_to(Node* target) const override;
    void serialize_extra(std::string& json) const override;
    void deserialize_extra(const std::string& key, const std::string& value) override;

private:
    Value scale_ = 1.0;
    Value bias_  = 0.0;
};

class ConstantNode : public Node {
public:
    ConstantNode(uint64_t id, const std::string& name, Value initial_value = 0.0);
    size_t get_min_inputs() const override { return 0; }
    void set_value(Value value);
    Value get_value() const;
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
    void copy_state_to(Node* target) const override;
    void serialize_extra(std::string& json) const override;
    void deserialize_extra(const std::string& key, const std::string& value) override;
private:
    Value value_;
};

class SinkNode : public Node {
public:
    SinkNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

// ============================================================================
// Arithmetic Nodes
// ============================================================================

class AddNode : public Node {
public:
    AddNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class SubtractNode : public Node {
public:
    SubtractNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class MultiplyNode : public Node {
public:
    MultiplyNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class DivideNode : public Node {
public:
    DivideNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class NegateNode : public Node {
public:
    NegateNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

// ============================================================================
// Neural Network Nodes
// ============================================================================

class NeuronNode : public Node {
public:
    NeuronNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }

    // Neuron: output = tanh(bias + sum(weight_i * input_i))
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;

    // Weight/bias management
    void set_weight(size_t index, Value w);
    Value get_weight(size_t index) const;
    void set_bias(Value b) { bias_ = b; }
    Value get_bias() const { return bias_; }
    size_t get_num_weights() const { return weights_.size(); }
    const std::vector<Value>& get_weights() const { return weights_; }

    // Resize inputs (called by Graph when connections change)
    virtual void set_input_count(size_t count);

    void copy_state_to(Node* target) const override;
    void serialize_extra(std::string& json) const override;
    void deserialize_extra(const std::string& key, const std::string& value) override;

protected:
    // Protected constructor for subclasses (LinearNode) that need a different NodeType
    NeuronNode(uint64_t id, const std::string& name, NodeType type);

    std::vector<Value> weights_;
    Value bias_ = 0.0;
};

// ============================================================================
// LinearNode 鈥?same as NEURON but WITHOUT the tanh activation.
// output = bias + sum(weight_i * input_i)   [identity activation]
// Used as the starter node for BCE loss (logistic-regression equivalent).
// The tanh in NEURON creates double-saturation (tanh + sigmoid) that kills
// gradients on high-dimensional classification; LINEAR avoids this.
// ============================================================================
class LinearNode : public NeuronNode {
public:
    LinearNode(uint64_t id, const std::string& name);
    void execute() override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
    std::unique_ptr<Node> clone() const override;
    // Override init: use 1/sqrt(fan_in) instead of Xavier's sqrt(6/fan_in).
    // LINEAR output is unbounded, so Xavier (tuned for tanh) makes pre-sigmoid
    // values too large 鈫?sigmoid saturation 鈫?vanishing gradients. The smaller
    // scale keeps 蟽(w路x+b) 鈮?0.5 initially so gradients flow.
    void set_input_count(size_t count) override;
};

class ReLUNode : public Node {
public:
    ReLUNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class SigmoidNode : public Node {
public:
    SigmoidNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class TanhNode : public Node {
public:
    TanhNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class SinNode : public Node {
public:
    SinNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

// ============================================================================
// Control Flow Nodes
// ============================================================================

class IfNode : public Node {
public:
    IfNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class IfElseNode : public Node {
public:
    IfElseNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
private:
    int forward_active_branch_ = 0;  // 0 = true branch (output[0]), 1 = false branch (output[1])
};

// ============================================================================
// MultiplexerNode — the INVERSE of IFELSE.
// IFELSE: (condition, value) → 2 outputs (true-out / false-out) — used to
// SPLIT a signal by condition downstream.
// MUX: (condition, a, b) → 1 output = a if condition else b — used to
// SELECT between two already-computed signals. This is the shape that
// learns piecewise/k-switching functions (W[sel] — a row-selected weight,
// the same primitive a bigram model needs).
// ============================================================================
class MultiplexerNode : public Node {
public:
    MultiplexerNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 3; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
private:
    int forward_active_branch_ = 0;  // 0 = a (condition true), 1 = b
};

// ============================================================================
// Logic & Comparison Nodes
// ============================================================================

class EqualNode : public Node {
public:
    EqualNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class NotEqualNode : public Node {
public:
    NotEqualNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class GreaterNode : public Node {
public:
    GreaterNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class LessNode : public Node {
public:
    LessNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class GreaterEqualNode : public Node {
public:
    GreaterEqualNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class LessEqualNode : public Node {
public:
    LessEqualNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class AndNode : public Node {
public:
    AndNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class OrNode : public Node {
public:
    OrNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class NotNode : public Node {
public:
    NotNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class XorNode : public Node {
public:
    XorNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 2; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

class AbsentNode : public Node {
public:
    AbsentNode(uint64_t id, const std::string& name);
    size_t get_min_inputs() const override { return 1; }
    void execute() override;
    std::unique_ptr<Node> clone() const override;
    std::vector<Value> backward_input_grads(Value output_grad) override;
};

// ============================================================================
// Node factory 鈥?for extensible node creation
// ============================================================================
using NodeFactory = std::function<std::unique_ptr<Node>(uint64_t, const std::string&)>;

class NodeRegistry {
public:
    static NodeRegistry& instance();

    // Register a factory for a node type
    void register_type(NodeType type, NodeFactory factory);

    // Create a node of the given type
    std::unique_ptr<Node> create(NodeType type, uint64_t id, const std::string& name) const;

private:
    NodeRegistry();
    std::unordered_map<NodeType, NodeFactory> factories_;
};

// Register all built-in types (called once at startup)
void register_builtin_node_types();

} // namespace aria
