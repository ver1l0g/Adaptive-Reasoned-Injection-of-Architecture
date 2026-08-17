#include "serialize.h"
#include "constants.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <cctype>

namespace gpnn {

// ============================================================================
// JSON string writing helper
// ============================================================================
static void json_append_string(std::string& out, const std::string& s) {
    out += '"';
    for (char c : s) {
        if (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n"; }
        else if (c == '\r') { out += "\\r"; }
        else if (c == '\t') { out += "\\t"; }
        else { out += c; }
    }
    out += '"';
}

static void json_append_indent(std::string& out, int depth) {
    for (int i = 0; i < depth; ++i) out += "  ";
}

// ============================================================================
// JSON parsing helpers
// ============================================================================

// Skip whitespace
static size_t skip_ws(const std::string& s, size_t p) {
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
    return p;
}

// Find the end of a balanced JSON value starting at `start` (first char of value).
// Returns position just past the value.
static size_t find_value_end(const std::string& s, size_t start) {
    size_t p = start;
    if (p >= s.size()) return p;
    char c = s[p];
    if (c == '"') {
        ++p; // skip opening quote
        while (p < s.size()) {
            if (s[p] == '\\') { p += 2; continue; }
            if (s[p] == '"') return p + 1;
            ++p;
        }
        return p;
    }
    if (c == '{' || c == '[') {
        // Use a bracket stack to track nesting of mixed { } and [ ]
        std::vector<char> bracket_stack;
        bracket_stack.push_back(c);
        bool in_string = false;
        ++p;
        while (p < s.size() && !bracket_stack.empty()) {
            char ch = s[p];
            if (in_string) {
                if (ch == '\\') ++p; // skip escaped char
                else if (ch == '"') in_string = false;
            } else {
                if (ch == '"') in_string = true;
                else if (ch == '{' || ch == '[') bracket_stack.push_back(ch);
                else if (ch == '}' || ch == ']') {
                    char expected = (ch == '}') ? '{' : '[';
                    if (!bracket_stack.empty() && bracket_stack.back() == expected) {
                        bracket_stack.pop_back();
                    }
                }
            }
            ++p;
        }
        return p;
    }
    // Number or literal
    while (p < s.size() && !std::isspace(static_cast<unsigned char>(s[p])) &&
           s[p] != ',' && s[p] != '}' && s[p] != ']') {
        ++p;
    }
    return p;
}

// Extract a JSON string value (without surrounding quotes, handles escapes)
static std::string unescape_json_string(const std::string& raw) {
    // raw includes surrounding quotes
    std::string result;
    for (size_t i = 1; i + 1 < raw.size(); ++i) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
            switch (raw[i + 1]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += raw[i + 1]; break;
            }
            ++i;
        } else {
            result += raw[i];
        }
    }
    return result;
}

// Extract key→value pairs from a JSON object string (includes outer {})
static std::vector<std::pair<std::string, std::string>> extract_object_pairs(const std::string& obj_str) {
    std::vector<std::pair<std::string, std::string>> pairs;
    // obj_str looks like:  { ... }
    const size_t MAX_PAIRS = config::DESERIALIZE_MAX_PAIRS;
    size_t p = skip_ws(obj_str, 1); // skip opening {
    while (p < obj_str.size() - 1 && pairs.size() < MAX_PAIRS) {
        p = skip_ws(obj_str, p);
        if (p >= obj_str.size() || obj_str[p] == '}') break;

        // Expect key string
        if (obj_str[p] != '"') throw std::runtime_error("JSON: expected string key in object");
        size_t key_start = p;
        size_t key_end = find_value_end(obj_str, p);
        std::string key = unescape_json_string(obj_str.substr(key_start, key_end - key_start));

        // Skip ':'
        p = skip_ws(obj_str, key_end);
        if (p >= obj_str.size() || obj_str[p] != ':') {
            throw std::runtime_error("JSON: expected ':' after key");
        }
        p = skip_ws(obj_str, p + 1);

        // Read value
        size_t val_end = find_value_end(obj_str, p);
        std::string val = obj_str.substr(p, val_end - p);
        pairs.emplace_back(key, val);

        p = skip_ws(obj_str, val_end);
        if (p < obj_str.size() && obj_str[p] == ',') ++p;
    }
    return pairs;
}

// Extract elements from a JSON array string (includes outer [])
static std::vector<std::string> extract_array_elements(const std::string& arr_str) {
    std::vector<std::string> elements;
    const size_t MAX_ELEMENTS = config::DESERIALIZE_MAX_ELEMENTS;
    size_t p = skip_ws(arr_str, 1); // skip opening [
    while (p < arr_str.size() - 1 && elements.size() < MAX_ELEMENTS) {
        p = skip_ws(arr_str, p);
        if (p >= arr_str.size() || arr_str[p] == ']') break;

        size_t val_end = find_value_end(arr_str, p);
        elements.push_back(arr_str.substr(p, val_end - p));

        p = skip_ws(arr_str, val_end);
        if (p < arr_str.size() && arr_str[p] == ',') ++p;
    }
    return elements;
}

// Parse a JSON number string to double
static double parse_number_value(const std::string& s) {
    return std::stod(s);
}

// Parse a JSON number array string (like "[1, 2, 3]") into vector of doubles
static std::vector<double> parse_number_array(const std::string& arr_str) {
    std::vector<double> result;
    auto elements = extract_array_elements(arr_str);
    for (const auto& e : elements) {
        result.push_back(parse_number_value(e));
    }
    return result;
}

// ============================================================================
// Serialize graph to JSON string
// ============================================================================
std::string serialize_graph(const Graph& graph) {
    std::string json;
    int d = 0;

    json += "{\n";
    ++d;

    // Version
    json_append_indent(json, d);
    json += "\"version\": ";
    json += std::to_string(config::SERIALIZATION_FORMAT_VERSION);
    json += ",\n";

    // Next ID
    json_append_indent(json, d);
    json += "\"next_id\": ";
    json += std::to_string(graph.get_next_id());
    json += ",\n";

    // Nodes array
    json_append_indent(json, d);
    json += "\"nodes\": [\n";
    ++d;
    const auto& nodes = graph.get_nodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        json_append_indent(json, d);
        json += "{";
        json += "\"id\":";
        json += std::to_string(node->get_id());
        json += ",\"type\":";
        json_append_string(json, node_type_to_string(node->get_type()));
        json += ",\"name\":";
        json_append_string(json, node->get_name());

        // Node-specific extra fields
        std::string extra;
        node->serialize_extra(extra);
        if (!extra.empty()) {
            json += extra;
        }

        json += "}";
        if (i + 1 < nodes.size()) json += ",";
        json += "\n";
    }
    --d;
    json_append_indent(json, d);
    json += "],\n";

    // Connections array
    json_append_indent(json, d);
    json += "\"connections\": [\n";
    ++d;
    const auto& conns = graph.get_connections();
    for (size_t i = 0; i < conns.size(); ++i) {
        const auto& c = conns[i];
        json_append_indent(json, d);
        json += "{";
        json += "\"src_node\":" + std::to_string(c.src_node) + ",";
        json += "\"src_port\":" + std::to_string(c.src_port) + ",";
        json += "\"dst_node\":" + std::to_string(c.dst_node) + ",";
        json += "\"dst_port\":" + std::to_string(c.dst_port) + ",";
        json += "\"is_recurrent\":" + std::string(c.is_recurrent ? "true" : "false");
        json += "}";
        if (i + 1 < conns.size()) json += ",";
        json += "\n";
    }
    --d;
    json_append_indent(json, d);
    json += "]\n";

    --d;
    json_append_indent(json, d);
    json += "}\n";
    return json;
}

// ============================================================================
// Deserialize JSON string into a graph
// ============================================================================
void deserialize_graph(Graph& graph, const std::string& json) {
    graph.clear();

    // Top-level object: extract key-value pairs
    auto top_pairs = extract_object_pairs(json);

    uint64_t next_id = config::DEFAULT_GRAPH_NODE_ID_START;
    std::vector<std::string> node_raw_list;
    std::vector<std::string> conn_raw_list;

    for (const auto& [key, value] : top_pairs) {
        if (key == "version") {
            int ver = static_cast<int>(parse_number_value(value));
            if (ver != config::SERIALIZATION_FORMAT_VERSION) {
                throw std::runtime_error("Unsupported graph format version: " + value);
            }
        } else if (key == "next_id") {
            next_id = static_cast<uint64_t>(parse_number_value(value));
        } else if (key == "nodes") {
            node_raw_list = extract_array_elements(value);
        } else if (key == "connections") {
            conn_raw_list = extract_array_elements(value);
        }
    }

    // --- Create nodes ---
    struct NodeInfo {
        uint64_t id;
        NodeType type;
        std::string name;
        std::vector<std::pair<std::string, std::string>> extras;
    };
    std::vector<NodeInfo> node_infos;

    for (const auto& node_raw : node_raw_list) {
        auto props = extract_object_pairs(node_raw);
        NodeInfo info;
        info.id = 0;
        info.type = NodeType::INPUT;
        for (const auto& [k, v] : props) {
            if (k == "id") {
                info.id = static_cast<uint64_t>(parse_number_value(v));
            } else if (k == "type") {
                info.type = node_type_from_string(unescape_json_string(v));
            } else if (k == "name") {
                info.name = unescape_json_string(v);
            } else {
                info.extras.emplace_back(k, v);
            }
        }
        node_infos.push_back(std::move(info));
    }

    // Create nodes and apply extras
    for (auto& info : node_infos) {
        auto node = NodeRegistry::instance().create(info.type, info.id, info.name);

        // Apply extras
        for (auto& [ek, ev] : info.extras) {
            if (info.type == NodeType::NEURON && ek == "weights") {
                auto weights = parse_number_array(ev);
                auto* neuron = dynamic_cast<NeuronNode*>(node.get());
                if (neuron) {
                    neuron->set_input_count(weights.empty() ? config::NEURON_MIN_INPUT_COUNT : weights.size());
                    for (size_t i = 0; i < weights.size(); ++i) {
                        neuron->set_weight(i, weights[i]);
                    }
                }
            } else {
                // For string values, the raw value might be quoted — pass unquoted
                if (!ev.empty() && ev.front() == '"') {
                    std::string unescaped = unescape_json_string(ev);
                    node->deserialize_extra(ek, unescaped);
                } else {
                    node->deserialize_extra(ek, ev);
                }
            }
        }

        graph.add_node_existing(std::move(node));
    }

    // --- Add connections ---
    for (const auto& conn_raw : conn_raw_list) {
        auto props = extract_object_pairs(conn_raw);
        uint64_t src_node = 0, dst_node = 0;
        size_t src_port = 0, dst_port = 0;
        bool is_recurrent = false;
        for (const auto& [k, v] : props) {
            if (k == "is_recurrent") {
                is_recurrent = (v == "true");
            } else {
                double num = parse_number_value(v);
                if (k == "src_node")      src_node = static_cast<uint64_t>(num);
                else if (k == "src_port") src_port = static_cast<size_t>(num);
                else if (k == "dst_node") dst_node = static_cast<uint64_t>(num);
                else if (k == "dst_port") dst_port = static_cast<size_t>(num);
            }
        }
        graph.add_connection_existing({src_node, src_port, dst_node, dst_port, is_recurrent, Value{0.0}});
    }

    graph.set_next_id(next_id);
}

// ============================================================================
// File I/O
// ============================================================================
void save_graph_to_file(const Graph& graph, const std::string& filepath) {
    std::string json = serialize_graph(graph);
    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    out << json;
    if (!out.good()) {
        throw std::runtime_error("Failed to write to file: " + filepath);
    }
}

void load_graph_from_file(Graph& graph, const std::string& filepath) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    deserialize_graph(graph, oss.str());
}

} // namespace gpnn
