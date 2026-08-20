#include "node.h"
#include "constants.h"
#include <stdexcept>
#include <cmath>
#include <cstdlib>

namespace aria {

// ============================================================================
// NodeType 鈫?string conversion
// ============================================================================
const char* node_type_to_string(NodeType type) {
    switch (type) {
        case NodeType::INPUT:    return "INPUT";
        case NodeType::OUTPUT:   return "OUTPUT";
        case NodeType::CONSTANT: return "CONSTANT";
        case NodeType::SINK:     return "SINK";
        case NodeType::ADD:      return "ADD";
        case NodeType::SUBTRACT: return "SUBTRACT";
        case NodeType::MULTIPLY: return "MULTIPLY";
        case NodeType::DIVIDE:   return "DIVIDE";
        case NodeType::NEGATE:   return "NEGATE";
        case NodeType::NEURON:   return "NEURON";
        case NodeType::RELU:     return "RELU";
        case NodeType::SIGMOID:  return "SIGMOID";
        case NodeType::TANH:     return "TANH";
    case NodeType::SIN:      return "SIN";
    case NodeType::LINEAR:   return "LINEAR";
        case NodeType::IF:           return "IF";
        case NodeType::IFELSE:       return "IFELSE";
        case NodeType::MUX:        return "MUX";
        case NodeType::EQUAL:        return "EQUAL";
        case NodeType::NOT_EQUAL:    return "NOT_EQUAL";
        case NodeType::GREATER:      return "GREATER";
        case NodeType::LESS:         return "LESS";
        case NodeType::GREATER_EQUAL:return "GREATER_EQUAL";
        case NodeType::LESS_EQUAL:   return "LESS_EQUAL";
        case NodeType::AND:          return "AND";
        case NodeType::OR:           return "OR";
        case NodeType::NOT:          return "NOT";
        case NodeType::XOR:          return "XOR";
        case NodeType::ABSENT:       return "ABSENT";
    }
    return "UNKNOWN";
}

NodeType node_type_from_string(const std::string& str) {
    if (str == "INPUT")    return NodeType::INPUT;
    if (str == "OUTPUT")   return NodeType::OUTPUT;
    if (str == "CONSTANT") return NodeType::CONSTANT;
    if (str == "SINK")     return NodeType::SINK;
    if (str == "ADD")      return NodeType::ADD;
    if (str == "SUBTRACT") return NodeType::SUBTRACT;
    if (str == "MULTIPLY") return NodeType::MULTIPLY;
    if (str == "DIVIDE")   return NodeType::DIVIDE;
    if (str == "NEGATE")   return NodeType::NEGATE;
    if (str == "NEURON")   return NodeType::NEURON;
    if (str == "RELU")     return NodeType::RELU;
    if (str == "SIGMOID")  return NodeType::SIGMOID;
    if (str == "TANH")     return NodeType::TANH;
    if (str == "SIN")      return NodeType::SIN;
    if (str == "LINEAR")   return NodeType::LINEAR;
    if (str == "IF")           return NodeType::IF;
    if (str == "IFELSE")       return NodeType::IFELSE;
    if (str == "MUX")        return NodeType::MUX;
    if (str == "EQUAL")        return NodeType::EQUAL;
    if (str == "NOT_EQUAL")    return NodeType::NOT_EQUAL;
    if (str == "GREATER")      return NodeType::GREATER;
    if (str == "LESS")         return NodeType::LESS;
    if (str == "GREATER_EQUAL")return NodeType::GREATER_EQUAL;
    if (str == "LESS_EQUAL")   return NodeType::LESS_EQUAL;
    if (str == "AND")          return NodeType::AND;
    if (str == "OR")           return NodeType::OR;
    if (str == "NOT")          return NodeType::NOT;
    if (str == "XOR")          return NodeType::XOR;
    if (str == "ABSENT")       return NodeType::ABSENT;
    throw std::runtime_error("Unknown NodeType string: " + str);
}

// ============================================================================
// Node base class
// ============================================================================
Node::Node(uint64_t id, NodeType type, const std::string& name,
           size_t num_inputs, size_t num_outputs)
    : id_(id), type_(type), name_(name),
      inputs_(num_inputs, 0.0),
      outputs_(num_outputs, 0.0),
      input_filled_(num_inputs, false),
      input_connected_(num_inputs, false),
      dirty_(true)
{}

void Node::set_input(size_t index, Value value) {
    if (index >= inputs_.size()) {
        throw std::out_of_range("Node::set_input index out of range");
    }
    inputs_[index] = value;
    input_filled_[index] = true;
}

Value Node::get_input(size_t index) const {
    if (index >= inputs_.size()) {
        throw std::out_of_range("Node::get_input index out of range");
    }
    return inputs_[index];
}

Value Node::get_output(size_t index) const {
    if (index >= outputs_.size()) {
        throw std::out_of_range("Node::get_output index out of range");
    }
    return outputs_[index];
}

bool Node::is_input_filled(size_t index) const {
    if (index >= input_filled_.size()) return false;
    return input_filled_[index];
}

bool Node::is_input_connected(size_t index) const {
    if (index >= input_connected_.size()) return false;
    return input_connected_[index];
}

void Node::mark_input_connected(size_t index) {
    if (index >= input_connected_.size()) {
        input_connected_.resize(index + 1, false);
    }
    input_connected_[index] = true;
}

void Node::clear_input_connected(size_t index) {
    if (index < input_connected_.size()) {
        input_connected_[index] = false;
    }
}

bool Node::has_all_inputs() const {
    for (bool f : input_filled_) {
        if (!f) return false;
    }
    return true;
}

void Node::reset() {
    for (size_t i = 0; i < input_filled_.size(); ++i) {
        input_filled_[i] = false;
    }
    dirty_ = true;
}

std::vector<Value> Node::backward_input_grads(Value /*output_grad*/) {
    // Default: no gradient propagation (terminals, sinks)
    return {};
}

// ============================================================================
// InputNode
// ============================================================================
InputNode::InputNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::INPUT, name, 0, 1) {}

void InputNode::set_value(Value value) {
    outputs_[0] = value;
    dirty_ = false;
}

void InputNode::execute() {
    // No computation needed 鈥?value is set externally via set_value()
}

std::vector<Value> InputNode::backward_input_grads(Value /*output_grad*/) {
    // Input node has no inputs 鈥?gradient stops here
    return {};
}

std::unique_ptr<Node> InputNode::clone() const {
    return std::make_unique<InputNode>(id_, name_);
}

void InputNode::copy_state_to(Node* target) const {
    if (auto* t = dynamic_cast<InputNode*>(target)) {
        t->outputs_[0] = outputs_[0];
    }
}

// ============================================================================
// OutputNode
// ============================================================================
OutputNode::OutputNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::OUTPUT, name, 1, 0), scale_(config::OUTPUT_DEFAULT_SCALE), bias_(config::OUTPUT_DEFAULT_BIAS) {}

Value OutputNode::get_value() const {
    return scale_ * inputs_[0] + bias_;
}

void OutputNode::execute() {
    // output = scale * input + bias (accessible via get_value())
}

std::vector<Value> OutputNode::backward_input_grads(Value output_grad) {
    // 鈭俹ut/鈭俰n = scale
    return {output_grad * scale_};
}

std::unique_ptr<Node> OutputNode::clone() const {
    auto node = std::make_unique<OutputNode>(id_, name_);
    node->scale_ = scale_;
    node->bias_  = bias_;
    return node;
}

void OutputNode::copy_state_to(Node* target) const {
    if (auto* t = dynamic_cast<OutputNode*>(target)) {
        t->scale_ = scale_;
        t->bias_  = bias_;
    }
}

void OutputNode::serialize_extra(std::string& json) const {
    json += ",\"scale\":" + std::to_string(scale_);
    json += ",\"bias\":" + std::to_string(bias_);
}

void OutputNode::deserialize_extra(const std::string& key, const std::string& value) {
    if (key == "scale") {
        scale_ = std::stod(value);
    } else if (key == "bias") {
        bias_ = std::stod(value);
    }
}

// ============================================================================
// ConstantNode
// ============================================================================
ConstantNode::ConstantNode(uint64_t id, const std::string& name, Value initial_value)
    : Node(id, NodeType::CONSTANT, name, 0, 1), value_(initial_value)
{
    outputs_[0] = value_;
    dirty_ = false;
}

void ConstantNode::set_value(Value value) {
    value_ = value;
    outputs_[0] = value_;
}

Value ConstantNode::get_value() const { return value_; }

void ConstantNode::execute() {
    outputs_[0] = value_;
}

std::vector<Value> ConstantNode::backward_input_grads(Value /*output_grad*/) {
    // Constant has no inputs 鈥?gradient stops here
    return {};
}

std::unique_ptr<Node> ConstantNode::clone() const {
    auto c = std::make_unique<ConstantNode>(id_, name_);
    c->set_value(value_);
    return c;
}

void ConstantNode::copy_state_to(Node* target) const {
    if (auto* t = dynamic_cast<ConstantNode*>(target)) {
        t->set_value(value_);
    }
}

void ConstantNode::serialize_extra(std::string& json) const {
    json += ",\"value\":";
    json += std::to_string(value_);
}

void ConstantNode::deserialize_extra(const std::string& key, const std::string& value) {
    if (key == "value") {
        set_value(std::stod(value));
    }
}

// ============================================================================
// SinkNode
// ============================================================================
SinkNode::SinkNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::SINK, name, 1, 0) {}

void SinkNode::execute() {
    // Sink absorbs its input 鈥?nothing to output
}

std::vector<Value> SinkNode::backward_input_grads(Value /*output_grad*/) {
    // Sink produces no output 鈥?no gradient to propagate
    return {};
}

std::unique_ptr<Node> SinkNode::clone() const {
    return std::make_unique<SinkNode>(id_, name_);
}

// ============================================================================
// AddNode
// ============================================================================
AddNode::AddNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::ADD, name, 2, 1) {}

void AddNode::execute() {
    outputs_[0] = inputs_[0] + inputs_[1];
}

std::vector<Value> AddNode::backward_input_grads(Value output_grad) {
    // 鈭?a+b)/鈭俛 = 1,  鈭?a+b)/鈭俠 = 1
    return {output_grad, output_grad};
}

std::unique_ptr<Node> AddNode::clone() const {
    return std::make_unique<AddNode>(id_, name_);
}

// ============================================================================
// SubtractNode
// ============================================================================
SubtractNode::SubtractNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::SUBTRACT, name, 2, 1) {}

void SubtractNode::execute() {
    outputs_[0] = inputs_[0] - inputs_[1];
}

std::vector<Value> SubtractNode::backward_input_grads(Value output_grad) {
    // 鈭?a-b)/鈭俛 = 1,  鈭?a-b)/鈭俠 = -1
    return {output_grad, -output_grad};
}

std::unique_ptr<Node> SubtractNode::clone() const {
    return std::make_unique<SubtractNode>(id_, name_);
}

// ============================================================================
// MultiplyNode
// ============================================================================
MultiplyNode::MultiplyNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::MULTIPLY, name, 2, 1) {}

void MultiplyNode::execute() {
    outputs_[0] = inputs_[0] * inputs_[1];
}

std::vector<Value> MultiplyNode::backward_input_grads(Value output_grad) {
    // 鈭?a*b)/鈭俛 = b,  鈭?a*b)/鈭俠 = a
    Value a = inputs_[0];
    Value b = inputs_[1];
    return {output_grad * b, output_grad * a};
}

std::unique_ptr<Node> MultiplyNode::clone() const {
    return std::make_unique<MultiplyNode>(id_, name_);
}

// ============================================================================
// DivideNode
// ============================================================================
DivideNode::DivideNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::DIVIDE, name, 2, 1) {}

void DivideNode::execute() {
    // Epsilon-magnitude clamp: near-zero denominators produce huge outputs
    // (z-scored inputs live near 0). Clamp |b| to a small floor, preserving
    // sign 鈥?keeps output finite and gradients bounded (mirrors the
    // BACKWARD_DIV_EPSILON guard in train()).
    Value b = inputs_[1];
    Value bmag = std::abs(b);
    if (bmag < config::DIVIDE_FORWARD_EPSILON) {
        b = (b >= 0.0) ? config::DIVIDE_FORWARD_EPSILON : -config::DIVIDE_FORWARD_EPSILON;
    }
    outputs_[0] = inputs_[0] / b;
}

std::vector<Value> DivideNode::backward_input_grads(Value output_grad) {
    // 鈭?a/b)/鈭俛 = 1/b,  鈭?a/b)/鈭俠 = -a/b虏
    Value a = inputs_[0];
    Value b = inputs_[1];
    if (b == 0.0) return {0.0, 0.0};
    return {output_grad / b, output_grad * (-a / (b * b))};
}

std::unique_ptr<Node> DivideNode::clone() const {
    return std::make_unique<DivideNode>(id_, name_);
}

// ============================================================================
// NegateNode
// ============================================================================
NegateNode::NegateNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::NEGATE, name, 1, 1) {}

void NegateNode::execute() {
    outputs_[0] = -inputs_[0];
}

std::vector<Value> NegateNode::backward_input_grads(Value output_grad) {
    // 鈭?-x)/鈭倄 = -1
    return {-output_grad};
}

std::unique_ptr<Node> NegateNode::clone() const {
    return std::make_unique<NegateNode>(id_, name_);
}

// ============================================================================
// NeuronNode
// ============================================================================
NeuronNode::NeuronNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::NEURON, name, 0, 1),
      weights_(),
      bias_(config::NEURON_DEFAULT_BIAS)
{}

NeuronNode::NeuronNode(uint64_t id, const std::string& name, NodeType type)
    : Node(id, type, name, 0, 1),
      weights_(),
      bias_(config::NEURON_DEFAULT_BIAS)
{}

void NeuronNode::set_input_count(size_t count) {
    // Minimum 1 input
    if (count < config::NEURON_MIN_INPUT_COUNT) count = config::NEURON_MIN_INPUT_COUNT;
    inputs_.resize(count, 0.0);
    input_filled_.resize(count, false);
    input_connected_.resize(count, false);
    // Preserve existing weights, extend with Xavier/Glorot uniform init
    size_t old_size = weights_.size();
    weights_.resize(count);
    // Xavier: U[-sqrt(6/n), +sqrt(6/n)] for tanh-activated neurons
    // Scale = gain * sqrt(6 / fan_in)
    double xavier_scale = config::WEIGHT_INIT_XAVIER_GAIN
                          * std::sqrt(6.0 / static_cast<double>(count));
    for (size_t i = old_size; i < count; ++i) {
        weights_[i] = static_cast<Value>(
            (static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0) * xavier_scale);
    }
}

void NeuronNode::set_weight(size_t index, Value w) {
    if (index >= weights_.size()) {
        weights_.resize(index + 1, 0.0);
    }
    weights_[index] = w;
}

Value NeuronNode::get_weight(size_t index) const {
    if (index >= weights_.size()) return 0.0;
    return weights_[index];
}

void NeuronNode::execute() {
    Value sum = bias_;
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Value w = (i < weights_.size()) ? weights_[i] : 0.0;
        sum += w * inputs_[i];
    }
    outputs_[0] = std::tanh(sum);
}

std::vector<Value> NeuronNode::backward_input_grads(Value output_grad) {
    // Neuron: output = tanh(bias + 危 w_i * x_i)
    // dtanh/dz = 1 - tanh虏(z) = 1 - output虏
    // 鈭俹ut/鈭倄_i = dtanh/dz * w_i = (1 - output虏) * w_i
    Value dtanh = 1.0 - outputs_[0] * outputs_[0];
    std::vector<Value> grads;
    grads.reserve(inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Value w = (i < weights_.size()) ? weights_[i] : 0.0;
        grads.push_back(output_grad * dtanh * w);
    }
    return grads;
}

std::unique_ptr<Node> NeuronNode::clone() const {
    auto n = std::make_unique<NeuronNode>(id_, name_);
    n->set_input_count(weights_.size());
    for (size_t i = 0; i < weights_.size(); ++i) {
        n->set_weight(i, weights_[i]);
    }
    n->set_bias(bias_);
    return n;
}

void NeuronNode::copy_state_to(Node* target) const {
    if (auto* t = dynamic_cast<NeuronNode*>(target)) {
        t->weights_ = weights_;
        t->bias_ = bias_;
    }
}

void NeuronNode::serialize_extra(std::string& json) const {
    // Serialize weights as JSON array
    json += ",\"weights\":[";
    for (size_t i = 0; i < weights_.size(); ++i) {
        if (i > 0) json += ",";
        json += std::to_string(weights_[i]);
    }
    json += "]";
    // Serialize bias
    json += ",\"bias\":";
    json += std::to_string(bias_);
}

void NeuronNode::deserialize_extra(const std::string& key, const std::string& value) {
    if (key == "bias") {
        bias_ = std::stod(value);
    }
    // "weights" is handled by the serializer (array parsing)
}

// ============================================================================
// ReLUNode
// ============================================================================
ReLUNode::ReLUNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::RELU, name, 1, 1) {}

void ReLUNode::execute() {
    outputs_[0] = (inputs_[0] > 0.0) ? inputs_[0] : 0.0;
}

std::vector<Value> ReLUNode::backward_input_grads(Value output_grad) {
    // LeakyReLU: 鈭?鈭倄 = 1 if x > 0, else RELU_LEAKY_SLOPE
    // Uses a small positive gradient for negative inputs to prevent gradient
    // vanishing, which would otherwise kill all upstream training.
    return {(inputs_[0] > 0.0) ? output_grad : output_grad * config::RELU_LEAKY_SLOPE};
}

std::unique_ptr<Node> ReLUNode::clone() const {
    return std::make_unique<ReLUNode>(id_, name_);
}

// ============================================================================
// SigmoidNode
// ============================================================================
SigmoidNode::SigmoidNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::SIGMOID, name, 1, 1) {}

void SigmoidNode::execute() {
    outputs_[0] = 1.0 / (1.0 + std::exp(-inputs_[0]));
}

std::vector<Value> SigmoidNode::backward_input_grads(Value output_grad) {
    // 鈭傁?x)/鈭倄 = 蟽(x) * (1 - 蟽(x)) = output * (1 - output)
    Value s = outputs_[0];
    return {output_grad * s * (1.0 - s)};
}

std::unique_ptr<Node> SigmoidNode::clone() const {
    return std::make_unique<SigmoidNode>(id_, name_);
}

// ============================================================================
// TanhNode
// ============================================================================
TanhNode::TanhNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::TANH, name, 1, 1) {}

void TanhNode::execute() {
    outputs_[0] = std::tanh(inputs_[0]);
}

std::vector<Value> TanhNode::backward_input_grads(Value output_grad) {
    // 鈭倀anh(x)/鈭倄 = 1 - tanh虏(x) = 1 - output虏
    Value t = outputs_[0];
    return {output_grad * (1.0 - t * t)};
}

std::unique_ptr<Node> TanhNode::clone() const {
    return std::make_unique<TanhNode>(id_, name_);
}

// ============================================================================
// SinNode
// ============================================================================
SinNode::SinNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::SIN, name, 1, 1) {}

void SinNode::execute() {
    outputs_[0] = std::sin(inputs_[0]);
}

std::vector<Value> SinNode::backward_input_grads(Value output_grad) {
    // 鈭俿in(x)/鈭倄 = cos(x)
    return {output_grad * std::cos(inputs_[0])};
}

std::unique_ptr<Node> SinNode::clone() const {
    return std::make_unique<SinNode>(id_, name_);
}

// ============================================================================
// LinearNode 鈥?identity activation (w路x + b, no tanh)
// ============================================================================
LinearNode::LinearNode(uint64_t id, const std::string& name)
    : NeuronNode(id, name, NodeType::LINEAR) {}

void LinearNode::set_input_count(size_t count) {
    // Smaller init than NeuronNode's Xavier: 1/sqrt(fan_in) vs sqrt(6/fan_in).
    // Keeps initial 蟽(w路x+b) 鈮?0.5 on high-dimensional inputs, preventing
    // sigmoid saturation that kills BCE gradients.
    if (count < config::NEURON_MIN_INPUT_COUNT) count = config::NEURON_MIN_INPUT_COUNT;
    inputs_.resize(count, 0.0);
    input_filled_.resize(count, false);
    input_connected_.resize(count, false);
    size_t old_size = weights_.size();
    weights_.resize(count);
    double linear_scale = 1.0 / std::sqrt(static_cast<double>(count));
    for (size_t i = old_size; i < count; ++i) {
        weights_[i] = static_cast<Value>(
            (static_cast<double>(std::rand()) / RAND_MAX * 2.0 - 1.0) * linear_scale);
    }
}

void LinearNode::execute() {
    Value sum = bias_;
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Value w = (i < weights_.size()) ? weights_[i] : 0.0;
        sum += w * inputs_[i];
    }
    outputs_[0] = sum;  // no tanh 鈥?identity activation
}

std::vector<Value> LinearNode::backward_input_grads(Value output_grad) {
    // 鈭俹ut/鈭倄_i = w_i  (no tanh derivative)
    std::vector<Value> grads;
    grads.reserve(inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
        Value w = (i < weights_.size()) ? weights_[i] : 0.0;
        grads.push_back(output_grad * w);
    }
    return grads;
}

std::unique_ptr<Node> LinearNode::clone() const {
    auto n = std::make_unique<LinearNode>(id_, name_);
    n->set_input_count(weights_.size());
    for (size_t i = 0; i < weights_.size(); ++i)
        n->set_weight(i, weights_[i]);
    n->set_bias(bias_);
    return n;
}

// ============================================================================
// Control Flow Nodes
// ============================================================================

IfNode::IfNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::IF, name, 2, 1) {}

void IfNode::execute() {
    // input[0] = condition, input[1] = value
    outputs_[0] = (inputs_[0] != 0.0) ? inputs_[1] : 0.0;
}

std::vector<Value> IfNode::backward_input_grads(Value output_grad) {
    // STE: use sigmoid to smooth the condition gate
    // soft(cond, val) 鈮?sigmoid(cond/T) * val
    Value temp = config::LOGIC_STE_TEMPERATURE;
    Value sig = 1.0 / (1.0 + std::exp(-inputs_[0] / temp));
    // cond gradient: d/dcond [sigmoid(cond/T) * val] = sig*(1-sig)/T * val
    Value cond_grad = output_grad * sig * (1.0 - sig) / temp * inputs_[1];
    // val gradient: keep discrete forward pass behavior
    Value val_grad = (inputs_[0] != 0.0) ? output_grad : 0.0;
    return {cond_grad, val_grad};
}

std::unique_ptr<Node> IfNode::clone() const {
    return std::make_unique<IfNode>(id_, name_);
}

// ============================================================================
// MultiplexerNode — the inverse of IFELSE (select between two signals)
// ============================================================================
MultiplexerNode::MultiplexerNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::MUX, name, 3, 1) {}

void MultiplexerNode::execute() {
    // input[0] = condition, input[1] = a (true branch), input[2] = b (false)
    if (inputs_[0] != 0.0) {
        outputs_[0] = inputs_[1];
        forward_active_branch_ = 0;
    } else {
        outputs_[0] = inputs_[2];
        forward_active_branch_ = 1;
    }
}

std::vector<Value> MultiplexerNode::backward_input_grads(Value output_grad) {
    // STE (straight-through estimator) with sigmoid smoothing, mirroring
    // IfElseNode: gradients route to the active value input, plus a small
    // condition gradient pushing the selection boundary.
    Value temp = config::LOGIC_STE_TEMPERATURE;
    Value sig = 1.0 / (1.0 + std::exp(-inputs_[0] / temp));
    Value sig_deriv = sig * (1.0 - sig) / temp;

    // Condition gradient: positive pushes toward a, negative toward b
    Value cond_grad = output_grad * sig_deriv * (inputs_[1] - inputs_[2]);

    // Value gradients: active branch passes through; inactive gets the
    // smoothed minority share (keeps both branches trainable near the
    // boundary — hard-zeroing stalls the losing branch).
    Value g_a = output_grad * sig;
    Value g_b = output_grad * (1.0 - sig);
    return {cond_grad, g_a, g_b};
}

std::unique_ptr<Node> MultiplexerNode::clone() const {
    return std::make_unique<MultiplexerNode>(id_, name_);
}

IfElseNode::IfElseNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::IFELSE, name, 2, 2) {}

void IfElseNode::execute() {
    // input[0] = condition, input[1] = value
    // output[0] = true case, output[1] = false case
    if (inputs_[0] != 0.0) {
        outputs_[0] = inputs_[1];
        outputs_[1] = 0.0;
        forward_active_branch_ = 0;
    } else {
        outputs_[0] = 0.0;
        outputs_[1] = inputs_[1];
        forward_active_branch_ = 1;
    }
}

std::vector<Value> IfElseNode::backward_input_grads(Value output_grad) {
    // STE: use sigmoid to smooth the condition gate
    // Dual-output node: only the active branch passes input[1] through.
    // The inactive branch outputs 0 鈥?its gradient does NOT route to input[1].
    // Condition gradient sign depends on which branch was active.
    Value temp = config::LOGIC_STE_TEMPERATURE;
    Value sig = 1.0 / (1.0 + std::exp(-inputs_[0] / temp));
    Value sig_deriv = sig * (1.0 - sig) / temp;

    // Condition gradient: positive if true branch active, negative if false branch active
    Value cond_grad = (forward_active_branch_ == 0)
        ? output_grad * sig_deriv * inputs_[1]
        : -output_grad * sig_deriv * inputs_[1];

    // Value gradient: only the active branch's gradient flows to input[1]
    // NOTE: output_grad is aggregated across both output ports. In practice,
    // the inactive branch's downstream consumers may contribute non-zero
    // gradient even though the forward value was 0. As an approximation we
    // use the aggregate; a more precise fix would require per-port gradient
    // tracking in the Graph backward pass.
    Value val_grad = output_grad;
    return {cond_grad, val_grad};
}

std::unique_ptr<Node> IfElseNode::clone() const {
    return std::make_unique<IfElseNode>(id_, name_);
}

// ============================================================================
// Logic & Comparison Nodes
// ============================================================================

// --- Equality ---

EqualNode::EqualNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::EQUAL, name, 2, 1) {}

void EqualNode::execute() {
    outputs_[0] = (inputs_[0] == inputs_[1]) ? 1.0 : 0.0;
}

std::vector<Value> EqualNode::backward_input_grads(Value output_grad) {
    // STE: Gaussian soft-equality exp(-(a-b)虏/(2T虏))
    // gradient pushes a and b toward each other
    Value diff = inputs_[0] - inputs_[1];
    Value temp = config::LOGIC_STE_TEMPERATURE;
    Value tsq = temp * temp;
    Value soft = std::exp(-diff * diff / (2.0 * tsq));
    Value grad = output_grad * (-diff / tsq) * soft;
    return {grad, -grad};
}

std::unique_ptr<Node> EqualNode::clone() const {
    return std::make_unique<EqualNode>(id_, name_);
}

NotEqualNode::NotEqualNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::NOT_EQUAL, name, 2, 1) {}

void NotEqualNode::execute() {
    outputs_[0] = (inputs_[0] != inputs_[1]) ? 1.0 : 0.0;
}

std::vector<Value> NotEqualNode::backward_input_grads(Value output_grad) {
    // STE: complement of Gaussian equality: 1 - exp(-(a-b)虏/(2T虏))
    // gradient pushes a and b away from each other
    Value diff = inputs_[0] - inputs_[1];
    Value temp = config::LOGIC_STE_TEMPERATURE;
    Value tsq = temp * temp;
    Value soft_eq = std::exp(-diff * diff / (2.0 * tsq));
    Value grad = output_grad * (diff / tsq) * soft_eq;
    return {grad, -grad};
}

std::unique_ptr<Node> NotEqualNode::clone() const {
    return std::make_unique<NotEqualNode>(id_, name_);
}

// --- Comparisons ---

GreaterNode::GreaterNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::GREATER, name, 2, 1) {}

void GreaterNode::execute() {
    outputs_[0] = (inputs_[0] > inputs_[1]) ? 1.0 : 0.0;
}

std::vector<Value> GreaterNode::backward_input_grads(Value output_grad) {
    // STE: sigmoid((a-b)/T) 鈥?gradient pushes a up, b down
    Value diff = inputs_[0] - inputs_[1];
    Value temp = config::LOGIC_STE_TEMPERATURE;
    Value sig = 1.0 / (1.0 + std::exp(-diff / temp));
    Value grad = output_grad * sig * (1.0 - sig) / temp;
    return {grad, -grad};
}

std::unique_ptr<Node> GreaterNode::clone() const {
    return std::make_unique<GreaterNode>(id_, name_);
}

LessNode::LessNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::LESS, name, 2, 1) {}

void LessNode::execute() {
    outputs_[0] = (inputs_[0] < inputs_[1]) ? 1.0 : 0.0;
}

std::vector<Value> LessNode::backward_input_grads(Value output_grad) {
    // STE: sigmoid((b-a)/T) 鈥?gradient pushes a down, b up
    Value diff = inputs_[0] - inputs_[1];
    Value temp = config::LOGIC_STE_TEMPERATURE;
    Value sig = 1.0 / (1.0 + std::exp(-diff / temp));
    Value grad = output_grad * sig * (1.0 - sig) / temp;
    return {-grad, grad};
}

std::unique_ptr<Node> LessNode::clone() const {
    return std::make_unique<LessNode>(id_, name_);
}

GreaterEqualNode::GreaterEqualNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::GREATER_EQUAL, name, 2, 1) {}

void GreaterEqualNode::execute() {
    outputs_[0] = (inputs_[0] >= inputs_[1]) ? 1.0 : 0.0;
}

std::vector<Value> GreaterEqualNode::backward_input_grads(Value output_grad) {
    // STE: sigmoid((a-b)/T) 鈥?same soft approximation as GREATER
    Value diff = inputs_[0] - inputs_[1];
    Value temp = config::LOGIC_STE_TEMPERATURE;
    Value sig = 1.0 / (1.0 + std::exp(-diff / temp));
    Value grad = output_grad * sig * (1.0 - sig) / temp;
    return {grad, -grad};
}

std::unique_ptr<Node> GreaterEqualNode::clone() const {
    return std::make_unique<GreaterEqualNode>(id_, name_);
}

LessEqualNode::LessEqualNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::LESS_EQUAL, name, 2, 1) {}

void LessEqualNode::execute() {
    outputs_[0] = (inputs_[0] <= inputs_[1]) ? 1.0 : 0.0;
}

std::vector<Value> LessEqualNode::backward_input_grads(Value output_grad) {
    // STE: sigmoid((b-a)/T) 鈥?same soft approximation as LESS
    Value diff = inputs_[0] - inputs_[1];
    Value temp = config::LOGIC_STE_TEMPERATURE;
    Value sig = 1.0 / (1.0 + std::exp(-diff / temp));
    Value grad = output_grad * sig * (1.0 - sig) / temp;
    return {-grad, grad};
}

std::unique_ptr<Node> LessEqualNode::clone() const {
    return std::make_unique<LessEqualNode>(id_, name_);
}

// --- Logic Gates ---

AndNode::AndNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::AND, name, 2, 1) {}

void AndNode::execute() {
    outputs_[0] = (inputs_[0] != 0.0 && inputs_[1] != 0.0) ? 1.0 : 0.0;
}

std::vector<Value> AndNode::backward_input_grads(Value output_grad) {
    // STE: treat AND as soft f(a,b) = a * b
    // 鈭俧/鈭俛 = b,  鈭俧/鈭俠 = a
    return {output_grad * inputs_[1], output_grad * inputs_[0]};
}

std::unique_ptr<Node> AndNode::clone() const {
    return std::make_unique<AndNode>(id_, name_);
}

OrNode::OrNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::OR, name, 2, 1) {}

void OrNode::execute() {
    outputs_[0] = (inputs_[0] != 0.0 || inputs_[1] != 0.0) ? 1.0 : 0.0;
}

std::vector<Value> OrNode::backward_input_grads(Value output_grad) {
    // STE: treat OR as soft f(a,b) = a + b - a*b
    // 鈭俧/鈭俛 = 1 - b,  鈭俧/鈭俠 = 1 - a
    return {output_grad * (1.0 - inputs_[1]), output_grad * (1.0 - inputs_[0])};
}

std::unique_ptr<Node> OrNode::clone() const {
    return std::make_unique<OrNode>(id_, name_);
}

NotNode::NotNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::NOT, name, 1, 1) {}

void NotNode::execute() {
    outputs_[0] = (inputs_[0] == 0.0) ? 1.0 : 0.0;
}

std::vector<Value> NotNode::backward_input_grads(Value output_grad) {
    // STE: treat NOT as soft f(a) = 1 - a
    // 鈭俧/鈭俛 = -1
    return {-output_grad};
}

std::unique_ptr<Node> NotNode::clone() const {
    return std::make_unique<NotNode>(id_, name_);
}

XorNode::XorNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::XOR, name, 2, 1) {}

void XorNode::execute() {
    bool a = (inputs_[0] != 0.0);
    bool b = (inputs_[1] != 0.0);
    outputs_[0] = (a != b) ? 1.0 : 0.0;
}

std::vector<Value> XorNode::backward_input_grads(Value output_grad) {
    // STE: treat XOR as soft f(a,b) = a + b - 2ab
    // 鈭俧/鈭俛 = 1 - 2b,  鈭俧/鈭俠 = 1 - 2a
    return {output_grad * (1.0 - 2.0 * inputs_[1]),
            output_grad * (1.0 - 2.0 * inputs_[0])};
}

std::unique_ptr<Node> XorNode::clone() const {
    return std::make_unique<XorNode>(id_, name_);
}

// ============================================================================
// AbsentNode
// ============================================================================
AbsentNode::AbsentNode(uint64_t id, const std::string& name)
    : Node(id, NodeType::ABSENT, name, 1, 1) {}

void AbsentNode::execute() {
    // Connected but no signal 鈫?absent 鈫?true (1)
    // Connected and signal     鈫?present 鈫?false (0)
    // Not connected at all     鈫?false (0)
    outputs_[0] = (is_input_connected(0) && !is_input_filled(0)) ? 1.0 : 0.0;
}

std::vector<Value> AbsentNode::backward_input_grads(Value output_grad) {
    (void)output_grad;
    return {0.0};
}

std::unique_ptr<Node> AbsentNode::clone() const {
    return std::make_unique<AbsentNode>(id_, name_);
}

// ============================================================================
// NodeRegistry
// ============================================================================
NodeRegistry& NodeRegistry::instance() {
    static NodeRegistry reg;
    return reg;
}

NodeRegistry::NodeRegistry() = default;

void NodeRegistry::register_type(NodeType type, NodeFactory factory) {
    factories_[type] = std::move(factory);
}

std::unique_ptr<Node> NodeRegistry::create(NodeType type, uint64_t id, const std::string& name) const {
    auto it = factories_.find(type);
    if (it == factories_.end()) {
        throw std::runtime_error("NodeRegistry: unregistered node type");
    }
    return it->second(id, name);
}

// ============================================================================
// Register all built-in node types
// ============================================================================
void register_builtin_node_types() {
    auto& reg = NodeRegistry::instance();

    reg.register_type(NodeType::INPUT,    [](uint64_t id, const std::string& name) { return std::make_unique<InputNode>(id, name); });
    reg.register_type(NodeType::OUTPUT,   [](uint64_t id, const std::string& name) { return std::make_unique<OutputNode>(id, name); });
    reg.register_type(NodeType::CONSTANT, [](uint64_t id, const std::string& name) { return std::make_unique<ConstantNode>(id, name); });
    reg.register_type(NodeType::SINK,     [](uint64_t id, const std::string& name) { return std::make_unique<SinkNode>(id, name); });

    reg.register_type(NodeType::ADD,      [](uint64_t id, const std::string& name) { return std::make_unique<AddNode>(id, name); });
    reg.register_type(NodeType::SUBTRACT, [](uint64_t id, const std::string& name) { return std::make_unique<SubtractNode>(id, name); });
    reg.register_type(NodeType::MULTIPLY, [](uint64_t id, const std::string& name) { return std::make_unique<MultiplyNode>(id, name); });
    reg.register_type(NodeType::DIVIDE,   [](uint64_t id, const std::string& name) { return std::make_unique<DivideNode>(id, name); });
    reg.register_type(NodeType::NEGATE,   [](uint64_t id, const std::string& name) { return std::make_unique<NegateNode>(id, name); });

    reg.register_type(NodeType::NEURON,   [](uint64_t id, const std::string& name) { return std::make_unique<NeuronNode>(id, name); });
    reg.register_type(NodeType::LINEAR,   [](uint64_t id, const std::string& name) { return std::make_unique<LinearNode>(id, name); });
    reg.register_type(NodeType::RELU,     [](uint64_t id, const std::string& name) { return std::make_unique<ReLUNode>(id, name); });
    reg.register_type(NodeType::SIGMOID,  [](uint64_t id, const std::string& name) { return std::make_unique<SigmoidNode>(id, name); });
    reg.register_type(NodeType::TANH,     [](uint64_t id, const std::string& name) { return std::make_unique<TanhNode>(id, name); });
    reg.register_type(NodeType::SIN,      [](uint64_t id, const std::string& name) { return std::make_unique<SinNode>(id, name); });
    reg.register_type(NodeType::IF,       [](uint64_t id, const std::string& name) { return std::make_unique<IfNode>(id, name); });
    reg.register_type(NodeType::IFELSE,   [](uint64_t id, const std::string& name) { return std::make_unique<IfElseNode>(id, name); });
    reg.register_type(NodeType::MUX,      [](uint64_t id, const std::string& name) { return std::make_unique<MultiplexerNode>(id, name); });

    reg.register_type(NodeType::EQUAL,        [](uint64_t id, const std::string& name) { return std::make_unique<EqualNode>(id, name); });
    reg.register_type(NodeType::NOT_EQUAL,    [](uint64_t id, const std::string& name) { return std::make_unique<NotEqualNode>(id, name); });
    reg.register_type(NodeType::GREATER,      [](uint64_t id, const std::string& name) { return std::make_unique<GreaterNode>(id, name); });
    reg.register_type(NodeType::LESS,         [](uint64_t id, const std::string& name) { return std::make_unique<LessNode>(id, name); });
    reg.register_type(NodeType::GREATER_EQUAL,[](uint64_t id, const std::string& name) { return std::make_unique<GreaterEqualNode>(id, name); });
    reg.register_type(NodeType::LESS_EQUAL,   [](uint64_t id, const std::string& name) { return std::make_unique<LessEqualNode>(id, name); });
    reg.register_type(NodeType::AND,          [](uint64_t id, const std::string& name) { return std::make_unique<AndNode>(id, name); });
    reg.register_type(NodeType::OR,           [](uint64_t id, const std::string& name) { return std::make_unique<OrNode>(id, name); });
    reg.register_type(NodeType::NOT,          [](uint64_t id, const std::string& name) { return std::make_unique<NotNode>(id, name); });
    reg.register_type(NodeType::XOR,          [](uint64_t id, const std::string& name) { return std::make_unique<XorNode>(id, name); });
    reg.register_type(NodeType::ABSENT,       [](uint64_t id, const std::string& name) { return std::make_unique<AbsentNode>(id, name); });
}

} // namespace aria
