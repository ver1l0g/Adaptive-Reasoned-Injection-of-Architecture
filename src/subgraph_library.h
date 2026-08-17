#pragma once

#include "graph.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace aria {

// ============================================================================
// BehavioralFingerprint 鈥?a compact feature vector that captures a subgraph's
// I/O behavior on a standard probe bench. Both subgraph behavior and a
// problem's needed-behavior produce one of these, so comparison is a simple
// weighted feature distance 鈥?no graph re-execution at retrieval time.
//
// Design: the features mirror the shape descriptors already used in
// EvolutionEngine::compute_complexity_profile (bounded, interaction, sharp,
// sign-symmetric, Lipschitz, poly_r2, quadrant means), so the "needed
// behavior" at an insertion point (the residual profile) and the "offered
// behavior" of a library subgraph live in the same feature space.
// ============================================================================
struct BehavioralFingerprint {
    // ---- Arity ----
    size_t num_inputs  = 0;
    size_t num_outputs = 1;

    // ---- Output shape ----
    double mean        = 0.0;
    double var         = 0.0;
    double min_val     = 0.0;
    double max_val     = 0.0;
    double bound_ratio = 0.0;   // (max-min)/stddev 鈥?small => bounded

    // ---- Polynomial degree-2 fit ----
    double poly_r2     = 0.0;   // 1 - SS_res/SS_tot of degree-2 fit
    double max_linear_coef   = 0.0;
    double max_nonlin_coef   = 0.0;   // largest |square or cross-term coef|
    size_t max_coef_index    = 0;

    // ---- Interaction ----
    bool   interaction_dominant = false;   // non-linear term > linear
    size_t interact_a = 0;
    size_t interact_b = 0;
    double sobol_pairwise = 0.0;

    // ---- Shape flags ----
    bool   bounded         = false;
    bool   sharp_boundary  = false;
    bool   sign_symmetric  = false;   // output >= ~0 everywhere (|f| signature)
    double lipschitz_max   = 0.0;

    // ---- Sign-pattern (quadrant means, for >= 2 inputs) ----
    // Order: (++, +-, -+, --). Pattern distinguishes abs [+,+,-,+] vs
    // product [+,+,-,+]... actually abs is [+,+,+,+], product is [+,-,-,+].
    std::vector<double> quadrant_means;

    // ---- Compatibility: can this fingerprint match a needed-behavior? ----
    // Returns true if arities are compatible (subgraph takes >= the needed
    // inputs; extra inputs can be wired to irrelevant signals).
    bool arity_compatible(size_t needed_inputs) const {
        return num_inputs <= needed_inputs + 2;  // allow some slack
    }
};

// ============================================================================
// Distance metric between two fingerprints.
// Returns 0.0 = identical behavior, larger = more different.
// Flags dominate (mismatch = big penalty); continuous descriptors refine.
// ============================================================================
double fingerprint_distance(const BehavioralFingerprint& a,
                            const BehavioralFingerprint& b);

// ============================================================================
// Compute a fingerprint from raw input/output data.
// X: N脳k input matrix (X[i][j] = sample i, input j).
// y: N output values.
// This is the standalone behavioral summarizer 鈥?no graph, no engine.
// ============================================================================
BehavioralFingerprint compute_fingerprint(const std::vector<std::vector<double>>& X,
                                          const std::vector<double>& y);

// ============================================================================
// Compute a fingerprint for a subgraph by running it on a standard probe bench.
// The probe bench is a fixed, deterministic set of input vectors per arity,
// so fingerprints are comparable across different subgraphs and runs.
// ============================================================================
BehavioralFingerprint fingerprint_subgraph(const Graph& g,
                                           const std::vector<uint64_t>& input_node_ids,
                                           uint64_t output_node_id);

// ============================================================================
// SubgraphLibraryEntry 鈥?one stored subgraph template + its fingerprint +
// provenance (which task/commit produced it).
// ============================================================================
struct SubgraphLibraryEntry {
    BehavioralFingerprint fingerprint;
    std::string source_task;          // task name that produced this
    std::string description;          // human-readable summary
    std::string canonical_expression; // abstracted formula (variables鈫抳, numbers鈫抍)
    std::string pattern;              // recognized structural pattern (e.g. "abs_product", "sin_chain")
    // The subgraph structure itself (for future instantiation).
    // Stored as node types + connectivity (lightweight serialization).
    // For now, we store the fingerprint + metadata; instantiation comes later.
};

// ============================================================================
// SubgraphLibrary 鈥?a persistent collection of entries. Save/load to disk so
// the library accumulates across runs. Extraction is OFFLINE (post-evolution)
// so evolution speed is unaffected.
// ============================================================================
class SubgraphLibrary {
public:
    // Add an entry (called post-evolution, not during evolve()).
    // If an entry with the same canonical_expression already exists, skip
    // (dedup 鈥?prevents the library from accumulating near-identical entries).
    // Returns true if added, false if deduplicated.
    bool add(const SubgraphLibraryEntry& entry);

    // Find the top-K entries whose fingerprint best matches a needed-behavior
    // fingerprint. Returns (entry_index, distance) pairs sorted by distance.
    struct Match { size_t index; double distance; };
    std::vector<Match> find_matches(const BehavioralFingerprint& needed,
                                    size_t top_k = 3) const;

    // Save / load (simple text format).
    bool save(const std::string& filepath) const;
    bool load(const std::string& filepath);

    size_t size() const { return entries_.size(); }
    const SubgraphLibraryEntry& entry(size_t i) const { return entries_[i]; }
    const std::vector<SubgraphLibraryEntry>& entries() const { return entries_; }

private:
    std::vector<SubgraphLibraryEntry> entries_;
};

// ============================================================================
// Generate a standard probe bench for a given input arity.
// Returns a fixed set of input vectors (deterministic, seeded).
// ============================================================================
std::vector<std::vector<double>> generate_probe_bench(size_t num_inputs,
                                                       size_t target_points = 30);

// ============================================================================
// Canonicalize an expression string: replace variables (x0, x1, ...) with "v"
// and all numeric constants with "c", producing an abstract template. Two
// graphs that compute the same function (up to parameter values and input
// naming) produce the same canonical expression.
// ============================================================================
std::string canonicalize_expression(const std::string& expr);

// ============================================================================
// Recognize a structural pattern from a canonical expression. Returns a short
// tag like "abs_product", "sin_chain", "boundary_split", "linear", "unknown".
// ============================================================================
std::string recognize_pattern(const std::string& canonical_expr);

// ============================================================================
// Extract reusable sub-expression blocks from a canonical expression.
// Finds known patterns (product, sin_chain, boundary, neuron) within the
// full expression and returns one SubgraphLibraryEntry per unique match.
// Each sub-expression entry has an empty fingerprint (identified by canonical
// form + pattern only) and is deduplicated by canonical_expression.
// ============================================================================
std::vector<SubgraphLibraryEntry> extract_sub_expressions(
    const std::string& canonical_expr,
    const std::string& source_task);

} // namespace aria
