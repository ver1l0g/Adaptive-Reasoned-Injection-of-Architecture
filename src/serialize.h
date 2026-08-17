#pragma once

#include "graph.h"
#include <string>

namespace gpnn {

// Serialize a graph to a JSON string
std::string serialize_graph(const Graph& graph);

// Deserialize a JSON string into a graph (clears existing content)
void deserialize_graph(Graph& graph, const std::string& json);

// Convenience: save/load to/from a file
void save_graph_to_file(const Graph& graph, const std::string& filepath);
void load_graph_from_file(Graph& graph, const std::string& filepath);

} // namespace gpnn
