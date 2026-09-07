#include "subgraph_library.h"
#include "constants.h"
#include "node.h"
#include <algorithm>
#include <map>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <regex>
#include <sstream>

namespace aria {

// ============================================================================
// Probe bench generation 鈥?deterministic (fixed seed), covers [-1,1]^k
// ============================================================================
std::vector<std::vector<double>> generate_probe_bench(size_t k, size_t target) {
    std::vector<std::vector<double>> bench;
    std::mt19937 rng(98765);  // fixed seed for reproducibility
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    // Corners of [-1,1]^k
    size_t corners = std::min(size_t(1) << std::min(k, size_t(4)), target / 2);
    for (size_t c = 0; c < corners; ++c) {
        std::vector<double> v(k);
        for (size_t j = 0; j < k; ++j) v[j] = (c & (1ULL << j)) ? 1.0 : -1.0;
        bench.push_back(v);
    }
    // Center
    bench.push_back(std::vector<double>(k, 0.0));
    // Axis midpoints
    for (size_t j = 0; j < k && bench.size() < target; ++j) {
        for (double s : {-0.5, 0.5}) {
            std::vector<double> v(k, 0.0);
            v[j] = s;
            bench.push_back(v);
        }
    }
    // Fill remaining with seeded randoms
    while (bench.size() < target) {
        std::vector<double> v(k);
        for (size_t j = 0; j < k; ++j) v[j] = u(rng);
        bench.push_back(v);
    }
    return bench;
}

// ============================================================================
// Compute fingerprint from raw (X[N脳k], y[N]) 鈥?standalone feature extractor.
// Adapts the shape-descriptor math from EvolutionEngine::compute_complexity_
// profile but operates on plain matrices, no graph/engine coupling.
// ============================================================================
BehavioralFingerprint compute_fingerprint(const std::vector<std::vector<double>>& X,
                                          const std::vector<double>& y) {
    BehavioralFingerprint fp;
    const size_t N = y.size();
    const size_t k = X.empty() ? 0 : X[0].size();
    fp.num_inputs = k;
    if (N < 4 || k == 0) return fp;

    // --- Output stats ---
    double sum = 0;
    for (auto v : y) sum += v;
    fp.mean = sum / N;
    double var = 0;
    for (auto v : y) var += (v - fp.mean) * (v - fp.mean);
    var /= N;
    fp.var = var;
    fp.min_val = *std::min_element(y.begin(), y.end());
    fp.max_val = *std::max_element(y.begin(), y.end());
    fp.bound_ratio = (var > 1e-12) ? (fp.max_val - fp.min_val) / std::sqrt(var) : 0.0;
    fp.bounded = (fp.bound_ratio > 0.0 && fp.bound_ratio < config::PROFILE_BOUNDED_RATIO_MAX);

    // --- Sign symmetry: output >= ~0 everywhere (|f| signature) ---
    double rng = fp.max_val - fp.min_val;
    fp.sign_symmetric = (rng > 1e-9) ? (fp.min_val >= -config::ABSMUL_NONNEG_FRACTION * rng) : true;

    // --- Lipschitz estimate per axis (max adjacent slope along sorted axis) ---
    fp.lipschitz_max = 0.0;
    for (size_t j = 0; j < k; ++j) {
        std::vector<size_t> order(N);
        for (size_t i = 0; i < N; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return X[a][j] < X[b][j]; });
        for (size_t i = 1; i < N; ++i) {
            double dx = X[order[i]][j] - X[order[i-1]][j];
            if (std::abs(dx) < 1e-12) continue;
            double slope = std::abs((y[order[i]] - y[order[i-1]]) / dx);
            if (slope > fp.lipschitz_max) fp.lipschitz_max = slope;
        }
    }
    fp.sharp_boundary = (fp.lipschitz_max > 10.0);

    // --- Degree-2 polynomial fit (Gaussian elimination on normal equations) ---
    size_t n_cross = k * (k - 1) / 2;
    size_t P = 1 + 2 * k + n_cross;
    if (N >= P + 2) {
        // Build Phi[N脳P] and solve (Phi^T Phi) 尾 = Phi^T y
        auto phi_col = [&](size_t i, size_t col) -> double {
            if (col == 0) return 1.0;
            if (col <= k) return X[i][col - 1];               // linear
            if (col <= 2*k) { size_t j = col - k - 1; return X[i][j]*X[i][j]; } // square
            size_t idx = col - 2*k - 1;                        // cross term
            size_t a = 0, b = 1;
            for (size_t ia = 0; ia < k; ++ia) for (size_t ib = ia+1; ib < k; ++ib) {
                if (idx == 0) { a = ia; b = ib; }
                --idx;
            }
            return X[i][a] * X[i][b];
        };

        std::vector<double> AtA(P * P, 0.0), Aty(P, 0.0);
        for (size_t i = 0; i < N; ++i)
            for (size_t r = 0; r < P; ++r) {
                double pr = phi_col(i, r);
                Aty[r] += pr * y[i];
                for (size_t c = 0; c < P; ++c)
                    AtA[r + c*P] += pr * phi_col(i, c);
            }
        // Gaussian elimination
        std::vector<double> M = AtA, beta = Aty;
        for (size_t col = 0; col < P; ++col) {
            size_t piv = col; double mx = std::abs(M[col + col*P]);
            for (size_t r = col+1; r < P; ++r) if (std::abs(M[r + col*P]) > mx) { mx = std::abs(M[r+col*P]); piv = r; }
            if (mx < 1e-12) continue;
            if (piv != col) { for (size_t c = 0; c < P; ++c) std::swap(M[col+c*P], M[piv+c*P]); std::swap(beta[col], beta[piv]); }
            double d = M[col + col*P];
            for (size_t r = col+1; r < P; ++r) { double f = M[r+col*P]/d; for (size_t c = col; c < P; ++c) M[r+c*P] -= f*M[col+c*P]; beta[r] -= f*beta[col]; }
        }
        for (int col = (int)P - 1; col >= 0; --col) {
            double s = 0; for (size_t c = col+1; c < P; ++c) s += M[col+c*P]*beta[c];
            if (std::abs(M[col+col*P]) > 1e-12) beta[col] = (beta[col]-s)/M[col+col*P]; else beta[col] = 0;
        }
        // R虏
        double ss_res = 0, ss_tot = 0;
        for (size_t i = 0; i < N; ++i) {
            double pred = 0; for (size_t r = 0; r < P; ++r) pred += beta[r] * phi_col(i, r);
            ss_res += (y[i]-pred)*(y[i]-pred);
            ss_tot += (y[i]-fp.mean)*(y[i]-fp.mean);
        }
        fp.poly_r2 = (ss_tot > 1e-12) ? 1.0 - ss_res/ss_tot : 0.0;
        // Max linear and non-linear coefficients
        for (size_t j = 0; j < k; ++j) fp.max_linear_coef = std::max(fp.max_linear_coef, std::abs(beta[1+j]));
        for (size_t i = 1+k; i < P; ++i) {
            if (std::abs(beta[i]) > fp.max_nonlin_coef) {
                fp.max_nonlin_coef = std::abs(beta[i]);
                fp.max_coef_index = i;
            }
        }
        fp.interaction_dominant = (fp.max_linear_coef > 1e-9)
            ? (fp.max_nonlin_coef / fp.max_linear_coef >= config::PROFILE_INTERACTION_RATIO)
            : (fp.max_nonlin_coef > 1e-3);
        // Decode interaction pair
        if (fp.interaction_dominant && fp.max_coef_index >= 1+k) {
            if (fp.max_coef_index < 1+2*k) {
                size_t j = fp.max_coef_index - (1+k);
                fp.interact_a = j; fp.interact_b = j;
            } else {
                size_t idx = fp.max_coef_index - (1+2*k);
                for (size_t a = 0; a < k; ++a) for (size_t b = a+1; b < k; ++b) {
                    if (idx == 0) { fp.interact_a = a; fp.interact_b = b; }
                    --idx;
                }
            }
        }
    }

    // --- Sobol pairwise interaction (for k >= 2) ---
    if (k >= 2 && var > 1e-12) {
        const int KB = config::PROFILE_SOBOL_BINS;
        double best = -1.0;
        for (size_t a = 0; a < k; ++a) {
            for (size_t b = a+1; b < k; ++b) {
                double mn_a = X[0][a], mx_a = mn_a, mn_b = X[0][b], mx_b = mn_b;
                for (size_t i = 1; i < N; ++i) {
                    if (X[i][a]<mn_a) mn_a=X[i][a]; if (X[i][a]>mx_a) mx_a=X[i][a];
                    if (X[i][b]<mn_b) mn_b=X[i][b]; if (X[i][b]>mx_b) mx_b=X[i][b];
                }
                double ra = mx_a-mn_a, rb = mx_b-mn_b;
                if (ra < 1e-12 || rb < 1e-12) continue;
                auto bin = [KB](double v, double mn, double rg) { int i = (int)((v-mn)/rg*KB); return std::max(0, std::min(KB-1, i)); };
                std::vector<std::vector<double>> sab(KB, std::vector<double>(KB, 0.0));
                std::vector<std::vector<int>> cab(KB, std::vector<int>(KB, 0));
                std::vector<double> sa(KB, 0.0); std::vector<int> ca(KB, 0);
                std::vector<double> sb(KB, 0.0); std::vector<int> cb(KB, 0);
                for (size_t i = 0; i < N; ++i) {
                    int ia = bin(X[i][a], mn_a, ra), ib = bin(X[i][b], mn_b, rb);
                    sab[ia][ib] += y[i]; cab[ia][ib]++;
                    sa[ia] += y[i]; ca[ia]++; sb[ib] += y[i]; cb[ib]++;
                }
                double vr_ab = 0, vr_a = 0, vr_b = 0;
                for (size_t i = 0; i < N; ++i) {
                    int ia = bin(X[i][a], mn_a, ra), ib = bin(X[i][b], mn_b, rb);
                    double e_ab = cab[ia][ib]>0 ? sab[ia][ib]/cab[ia][ib] : fp.mean;
                    double e_a  = ca[ia]>0 ? sa[ia]/ca[ia] : fp.mean;
                    double e_b  = cb[ib]>0 ? sb[ib]/cb[ib] : fp.mean;
                    vr_ab += (e_ab-fp.mean)*(e_ab-fp.mean);
                    vr_a += (e_a-fp.mean)*(e_a-fp.mean);
                    vr_b += (e_b-fp.mean)*(e_b-fp.mean);
                }
                vr_ab/=N; vr_a/=N; vr_b/=N;
                double V_int = vr_ab - vr_a - vr_b;
                double strength = var > 1e-12 ? V_int/var : 0.0;
                if (strength > best) best = strength;
            }
        }
        fp.sobol_pairwise = best;
    }

    // --- Quadrant means (for k >= 2, first 2 inputs) ---
    if (k >= 2) {
        double qm[4] = {0,0,0,0}; int qn[4] = {0,0,0,0};
        for (size_t i = 0; i < N; ++i) {
            int q = (X[i][0] >= 0 ? 0 : 2) + (X[i][1] >= 0 ? 0 : 1);
            qm[q] += y[i]; qn[q]++;
        }
        for (int j = 0; j < 4; ++j) fp.quadrant_means.push_back(qn[j]>0 ? qm[j]/qn[j] : fp.mean);
    }

    return fp;
}

// ============================================================================
// Fingerprint a subgraph by running it on a probe bench
// ============================================================================
BehavioralFingerprint fingerprint_subgraph(const Graph& g,
                                           const std::vector<uint64_t>& input_ids,
                                           uint64_t output_id) {
    size_t k = input_ids.size();
    auto bench = generate_probe_bench(k);
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    // Clone the graph so we don't disturb the original's state.
    auto sg = g.clone();
    // Map original IDs to clone IDs (clone preserves IDs).
    for (const auto& probe : bench) {
        for (size_t j = 0; j < k; ++j)
            sg->set_input_value(input_ids[j], probe[j]);
        sg->execute();
        Value out = sg->get_output_value(output_id);
        X.push_back(probe);
        y.push_back(out);
    }
    return compute_fingerprint(X, y);
}

// ============================================================================
// Fingerprint distance 鈥?0 = identical behavior, larger = more different.
// Flags dominate; continuous descriptors refine.
// ============================================================================
double fingerprint_distance(const BehavioralFingerprint& a,
                            const BehavioralFingerprint& b) {
    // Arity gate
    if (!a.arity_compatible(b.num_inputs) && !b.arity_compatible(a.num_inputs))
        return 1e9;

    double d = 0.0;
    // Flag mismatches (each worth 1.0 鈥?these are the primary discriminators)
    auto flag_diff = [&](bool fa, bool fb) { return (fa != fb) ? 1.0 : 0.0; };
    d += flag_diff(a.bounded, b.bounded);
    d += flag_diff(a.sharp_boundary, b.sharp_boundary);
    d += flag_diff(a.interaction_dominant, b.interaction_dominant);
    d += flag_diff(a.sign_symmetric, b.sign_symmetric);

    // Continuous descriptors (normalized)
    auto norm_diff = [](double va, double vb, double scale) {
        return std::abs(va - vb) / (scale + 1e-12);
    };
    d += 0.3 * norm_diff(a.bound_ratio, b.bound_ratio, 5.0);
    d += 0.3 * norm_diff(a.lipschitz_max, b.lipschitz_max, 10.0);
    d += 0.2 * norm_diff(a.poly_r2, b.poly_r2, 1.0);
    d += 0.2 * norm_diff(a.sobol_pairwise, b.sobol_pairwise, 0.5);

    // Quadrant means (if both have them and same arity)
    if (a.quadrant_means.size() == 4 && b.quadrant_means.size() == 4) {
        double qd = 0;
        for (int j = 0; j < 4; ++j) qd += (a.quadrant_means[j]-b.quadrant_means[j])*(a.quadrant_means[j]-b.quadrant_means[j]);
        d += 0.3 * std::sqrt(qd);
    }

    return d;
}

// ============================================================================
// Canonicalize expression: variables鈫抳, numbers鈫抍
// ============================================================================
std::string canonicalize_expression(const std::string& expr) {
    std::string result = expr;
    // Replace variables: x0, x1, x12, ... 鈫?v
    std::regex var_re("x[0-9]+");
    result = std::regex_replace(result, var_re, "v");
    // Replace all numeric constants (floats and integers) 鈫?c
    std::regex num_re("-?[0-9]+(\\.[0-9]+)?");
    result = std::regex_replace(result, num_re, "c");
    return result;
}

// ============================================================================
// Recognize structural pattern from canonical expression
// ============================================================================
std::string recognize_pattern(const std::string& ce) {
    // |x*y|-class: contains (v*v) inside a sign-flip pattern
    if (ce.find("(v*v)") != std::string::npos &&
        (ce.find(">?c:c") != std::string::npos || ce.find(">c?") != std::string::npos))
        return "abs_product";
    // sin chain: contains sin(
    if (ce.find("sin(") != std::string::npos)
        return "sin_chain";
    // tanh chain: multiple tanh( calls
    size_t tanh_count = 0, pos = 0;
    while ((pos = ce.find("tanh(", pos)) != std::string::npos) { tanh_count++; pos += 5; }
    if (tanh_count >= 2) return "tanh_stack";
    if (tanh_count == 1) return "single_tanh";
    // boundary/step: contains ternary with >
    if (ce.find(">?c:c") != std::string::npos || ce.find(">c?") != std::string::npos)
        return "boundary_split";
    // product: contains (v*v) without abs
    if (ce.find("(v*v)") != std::string::npos) return "product";
    // linear: just weighted sum
    if (ce.find("tanh(") == std::string::npos && ce.find("sin(") == std::string::npos
        && ce.find("?") == std::string::npos)
        return "linear";
    return "unknown";
}

// ============================================================================
// SubgraphLibrary
// ============================================================================
bool SubgraphLibrary::add(const SubgraphLibraryEntry& entry) {
    // Dedup: skip if an entry with the same canonical expression exists.
    if (!entry.canonical_expression.empty()) {
        for (const auto& e : entries_) {
            if (e.canonical_expression == entry.canonical_expression) {
                return false;  // duplicate 鈥?skip
            }
        }
    }
    entries_.push_back(entry);
    return true;
}

std::vector<SubgraphLibrary::Match> SubgraphLibrary::find_matches(
    const BehavioralFingerprint& needed, size_t top_k) const {
    std::vector<Match> matches;
    for (size_t i = 0; i < entries_.size(); ++i) {
        // M7.5(a): skip empty-fingerprint (sub-expression) entries
        if (entries_[i].fingerprint.num_inputs == 0) continue;
        if (!entries_[i].fingerprint.arity_compatible(needed.num_inputs)) continue;
        matches.push_back({i, fingerprint_distance(needed, entries_[i].fingerprint)});
    }
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) { return a.distance < b.distance; });
    if (matches.size() > top_k) matches.resize(top_k);
    return matches;
}

std::vector<SubgraphLibrary::Match> SubgraphLibrary::find_matches_excluding_self(
    const BehavioralFingerprint& needed, size_t top_k,
    const std::string& current_task) const {
    // Self-echo guard: entries sourced from the CURRENT task are skipped —
    // a task matching its own earlier save re-injects what it already
    // tried (hetero3: 65 self-injects for 1 commit). Cross-task transfer
    // is the library's entire purpose.
    std::vector<Match> matches;
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].source_task == current_task) continue;
        // M7.5(a): skip empty-fingerprint entries — sub-expression blocks
        // (tanh_stack, neuron_unit, ...) are written WITHOUT fingerprints;
        // they pass arity checks vacuously (num_inputs=0) and compute
        // meaningless zero-distances that pollute match selection.
        if (entries_[i].fingerprint.num_inputs == 0) continue;
        if (!entries_[i].fingerprint.arity_compatible(needed.num_inputs)) continue;
        matches.push_back({i, fingerprint_distance(needed, entries_[i].fingerprint)});
    }
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) { return a.distance < b.distance; });
    if (matches.size() > top_k) matches.resize(top_k);
    return matches;
}

// ============================================================================
// M7.7: architecture descriptor similarity + hybrid matching
// ============================================================================
double ArchitectureDescriptor::similarity(const ArchitectureDescriptor& other) const {
    // Parse both histograms into type->count maps.
    auto parse = [](const std::string& h) {
        std::map<std::string, int> m;
        size_t p = 0;
        while (p < h.size()) {
            size_t colon = h.find(':', p);
            if (colon == std::string::npos) break;
            std::string type = h.substr(p, colon - p);
            size_t end = h.find(',', colon);
            int cnt = 0;
            try { cnt = std::stoi(h.substr(colon + 1, (end == std::string::npos ? h.size() : end) - colon - 1)); } catch (...) {}
            m[type] = cnt;
            if (end == std::string::npos) break;
            p = end + 1;
        }
        return m;
    };
    auto a = parse(node_histogram);
    auto b = parse(other.node_histogram);
    if (a.empty() && b.empty()) return 0.5;   // both unknown: neutral
    // L1 distance over the union, normalized by total node mass.
    int total = 0, diff = 0;
    for (auto& kv : a) { total += kv.second; diff += std::abs(kv.second - (b.count(kv.first) ? b.at(kv.first) : 0)); }
    for (auto& kv : b) { total += kv.second; if (!a.count(kv.first)) diff += kv.second; }
    if (total == 0) return 0.5;
    double hist_sim = 1.0 - static_cast<double>(diff) / static_cast<double>(2 * total);
    // Same creating family = architectural kin (even at different sizes).
    double family_sim = (!family.empty() && family == other.family) ? 1.0 : 0.0;
    return std::min(1.0, 0.6 * hist_sim + 0.4 * family_sim);
}

std::vector<SubgraphLibrary::Match> SubgraphLibrary::find_hybrid_matches(
    const BehavioralFingerprint& needed,
    const ArchitectureDescriptor& current_arch,
    size_t top_k,
    const std::string& current_task) const {
    std::vector<Match> matches;
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].source_task == current_task) continue;
        if (entries_[i].fingerprint.num_inputs == 0) continue;   // M7.5(a)
        if (!entries_[i].fingerprint.arity_compatible(needed.num_inputs)) continue;
        double behav = fingerprint_distance(needed, entries_[i].fingerprint);
        double arch_sim = entries_[i].arch.similarity(current_arch);
        // Hybrid score: behaviorally close AND architecturally kin wins;
        // behaviorally close but architecturally alien is demoted (the
        // tanh_stack monoculture fix — the entry must have built the
        // KIND of machine that works, not just solved a similar residual).
        double score = behav * (1.4 - 0.4 * arch_sim);
        matches.push_back({i, score});
    }
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) { return a.distance < b.distance; });
    if (matches.size() > top_k) matches.resize(top_k);
    return matches;
}

bool SubgraphLibrary::save(const std::string& filepath) const {
    // APPEND-ONLY for new entries (the concurrent lost-update fix): a
    // full-file rewrite meant two runs sharing a library erased each
    // other's entries — the last finisher's in-memory copy won. Now we
    // append only entries added since OUR load. Each entry is written as
    // one buffered unit; appends on a local filesystem are effectively
    // atomic at our entry sizes (~1-2KB). HEADERLESS format (no count —
    // the loader reads to EOF and skips a legacy integer header if
    // present, so old files remain readable).
    if (entries_.size() <= loaded_count_) return true;   // nothing new
    std::ofstream f(filepath, std::ios::app);
    if (!f) return false;
    for (size_t i = loaded_count_; i < entries_.size(); ++i) {
        const auto& e = entries_[i];
        const auto& fp = e.fingerprint;
        std::string block;
        std::ostringstream oss;
        oss << std::quoted(e.source_task) << "\t" << std::quoted(e.description)
            << "\t" << std::quoted(e.canonical_expression)
            << "\t" << std::quoted(e.pattern);
        if (!e.params.empty()) oss << "\t" << std::quoted(e.params);
        oss << "\n";
        oss << fp.num_inputs << " " << fp.num_outputs << " "
            << fp.mean << " " << fp.var << " " << fp.min_val << " " << fp.max_val << " "
            << fp.bound_ratio << " " << fp.poly_r2 << " "
            << fp.max_linear_coef << " " << fp.max_nonlin_coef << " " << fp.max_coef_index << " "
            << fp.interaction_dominant << " " << fp.interact_a << " " << fp.interact_b << " "
            << fp.sobol_pairwise << " " << fp.bounded << " " << fp.sharp_boundary << " "
            << fp.sign_symmetric << " " << fp.lipschitz_max << " "
            << fp.quadrant_means.size();
        for (auto q : fp.quadrant_means) oss << " " << q;
        oss << " " << e.arch.node_count << " " << e.arch.edge_count
            << " " << e.arch.depth << " " << e.arch.param_count
            << " " << e.arch.recurrent;
        oss << "\n";
        oss << std::quoted(e.arch.node_histogram) << "\t"
            << std::quoted(e.arch.family) << "\n";
        f << oss.str();   // one write per entry
    }
    return true;
}

// M7.7: describe a graph architecturally (shared by save-time main.cpp
// and match-time evolution.cpp).
ArchitectureDescriptor SubgraphLibrary::describe_graph(const Graph& g) {
    ArchitectureDescriptor d;
    std::map<std::string, int> hist;
    int edges = 0, params = 0, recur = 0;
    for (const auto& n : g.get_nodes()) {
        hist[node_type_to_string(n->get_type())]++;
        if (auto* nn = dynamic_cast<const NeuronNode*>(n.get())) {
            params += static_cast<int>(nn->get_num_weights()) + 1;
        } else if (n->get_type() == NodeType::OUTPUT) {
            params += 2;
        }
    }
    for (const auto& c : g.get_connections()) {
        ++edges;
        if (c.is_recurrent) ++recur;
    }
    std::string h;
    for (auto& kv : hist) {
        if (!h.empty()) h += ",";
        h += kv.first + ":" + std::to_string(kv.second);
    }
    d.node_count = static_cast<int>(g.get_nodes().size());
    d.edge_count = edges;
    // depth: longest INPUT->OUTPUT path (BFS)
    {
        std::map<uint64_t, int> lvl;
        std::vector<uint64_t> frontier;
        for (const auto& n : g.get_nodes()) {
            if (n->get_type() == NodeType::INPUT) {
                lvl[n->get_id()] = 1;
                frontier.push_back(n->get_id());
            }
        }
        int max_d = 0;
        while (!frontier.empty()) {
            std::vector<uint64_t> next;
            for (uint64_t u : frontier) {
                max_d = std::max(max_d, lvl[u]);
                for (const auto& c : g.get_connections()) {
                    if (c.src_node == u && !c.is_recurrent
                        && !lvl.count(c.dst_node)) {
                        lvl[c.dst_node] = lvl[u] + 1;
                        next.push_back(c.dst_node);
                    }
                }
            }
            frontier = std::move(next);
        }
        d.depth = max_d;
    }
    d.param_count = params;
    d.recurrent = recur;
    d.node_histogram = h;
    return d;
}

bool SubgraphLibrary::load(const std::string& filepath) {
    // LINE-BASED, TOLERANT loader (the version-skew landmine fix).
    // Reads whole lines and splits by tab: an unexpected extra field
    // (the aria25/26 params landmine) can no longer desync the stream —
    // worst case a malformed ENTRY is skipped. Reads to EOF; skips a
    // legacy integer count header if the first line is one.
    std::ifstream f(filepath);
    if (!f) return false;
    entries_.clear();
    loaded_count_ = 0;
    std::string line;
    int skipped = 0;
    // Legacy header: a first line that is ONLY an integer.
    if (std::getline(f, line)) {
        bool all_digits = !line.empty()
            && line.find_first_not_of("0123456789 \t\r\n") == std::string::npos;
        if (!all_digits) {
            // Not a header — it's an entry's first line; process below
            // by pushing it back (handled via a pending-line variable).
            f.seekg(0);
        }
        // else: header consumed, continue
    }
    auto unquote = [](std::string s) {
        if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
            return s.substr(1, s.size() - 2);
        }
        return s;
    };
    std::string pending;
    while (true) {
        if (!pending.empty()) { line = pending; pending.clear(); }
        else if (!std::getline(f, line)) break;
        if (line.find('"') == std::string::npos) continue;   // blank/garbage
        // Entry line 1: task \t desc \t canon \t pattern [\t params]
        {
            std::vector<std::string> fields;
            size_t p = 0;
            while (p <= line.size()) {
                size_t tab = line.find('\t', p);
                std::string tok = line.substr(
                    p, (tab == std::string::npos ? line.size() : tab) - p);
                fields.push_back(unquote(tok));
                if (tab == std::string::npos) break;
                p = tab + 1;
            }
            SubgraphLibraryEntry e;
            if (fields.size() >= 4) {
                e.source_task = fields[0];
                e.description = fields[1];
                e.canonical_expression = fields[2];
                e.pattern = fields[3];
                if (fields.size() >= 5) e.params = fields[4];   // tolerated
            } else {
                ++skipped;
                continue;
            }
            // Line 2: fingerprint + arch ints.
            if (!std::getline(f, line)) { ++skipped; break; }
            {
                std::istringstream fs(line);
                auto& fp = e.fingerprint;
                if (!(fs >> fp.num_inputs >> fp.num_outputs
                      >> fp.mean >> fp.var >> fp.min_val >> fp.max_val
                      >> fp.bound_ratio >> fp.poly_r2
                      >> fp.max_linear_coef >> fp.max_nonlin_coef
                      >> fp.max_coef_index
                      >> fp.interaction_dominant >> fp.interact_a >> fp.interact_b
                      >> fp.sobol_pairwise >> fp.bounded >> fp.sharp_boundary
                      >> fp.sign_symmetric >> fp.lipschitz_max)) {
                    ++skipped;
                    continue;
                }
                size_t qn = 0;
                fs >> qn;
                if (qn > 64) qn = 0;   // parse-garbage guard
                fp.quadrant_means.resize(qn);
                for (size_t j = 0; j < qn; ++j) fs >> fp.quadrant_means[j];
                // M7.7 trailing arch ints (absent on legacy = zeros).
                fs >> e.arch.node_count >> e.arch.edge_count
                   >> e.arch.depth >> e.arch.param_count >> e.arch.recurrent;
            }
            // Line 3 (optional): histogram \t family.
            if (std::getline(f, line) && !line.empty() && line[0] == '"') {
                std::vector<std::string> af;
                size_t p = 0;
                while (p <= line.size()) {
                    size_t tab = line.find('\t', p);
                    std::string tok = line.substr(
                        p, (tab == std::string::npos ? line.size() : tab) - p);
                    af.push_back(unquote(tok));
                    if (tab == std::string::npos) break;
                    p = tab + 1;
                }
                if (af.size() >= 2) {
                    e.arch.node_histogram = af[0];
                    e.arch.family = af[1];
                }
            } else if (!line.empty()) {
                pending = line;   // it was the next entry's line 1
            }
            entries_.push_back(std::move(e));
        }
    }
    loaded_count_ = entries_.size();
    if (skipped > 0) {
        // Note: logged at load site (logger not linked here).
    }
    return !entries_.empty() || skipped == 0;
}
// ============================================================================
// Extract reusable sub-expression blocks from a canonical expression
// ============================================================================
std::vector<SubgraphLibraryEntry> extract_sub_expressions(
    const std::string& canonical_expr,
    const std::string& source_task) {

    std::vector<SubgraphLibraryEntry> entries;

    struct PatternDef {
        std::string regex_str;
        std::string tag;
        std::string desc;
    };

    // Patterns searched on the CANONICAL expression (v=variable, c=constant).
    // Each match is a reusable formula block.
    std::vector<PatternDef> patterns = {
        // Product interaction: (v*v) 鈥?cross or self product
        {"\\(v\\*v\\)",                          "product",       "x*y interaction feature"},
        // Sin oscillator component: sin(tanh(...))
        {"sin\\(tanh\\([^)]*\\)\\)",             "sin_component", "sin(tanh(wx+b)) oscillator"},
        // Boundary/step: (v>c?v:...) or (...>c?v:c)
        {"\\([^)]*>c\\?[^:]*:[^)]*\\)",          "boundary",      "conditional step/boundary"},
        // Abs pattern: (...*2-c) inside a multiply (sign computation)
        {"\\*\\(c\\*[^)]*-c\\)",                 "sign_flip",     "sign(x) = 2*(x>0)-1"},
        // Single tanh neuron: tanh(c*v+c)
        {"tanh\\(c\\*[vc][^)]*\\)",              "neuron_unit",   "tanh(wx+b) neuron"},
    };

    for (const auto& pd : patterns) {
        try {
            std::regex re(pd.regex_str);
            auto begin = std::sregex_iterator(canonical_expr.begin(),
                                              canonical_expr.end(), re);
            auto end = std::sregex_iterator();
            for (auto it = begin; it != end; ++it) {
                SubgraphLibraryEntry e;
                e.canonical_expression = it->str();
                e.pattern = pd.tag;
                e.description = pd.desc;
                e.source_task = source_task;
                entries.push_back(e);
            }
        } catch (...) {
            // regex error 鈥?skip this pattern
        }
    }

    // Deduplicate within this extraction (same sub-expr may match multiple patterns)
    std::sort(entries.begin(), entries.end(),
              [](const SubgraphLibraryEntry& a, const SubgraphLibraryEntry& b) {
                  return a.canonical_expression < b.canonical_expression;
              });
    entries.erase(std::unique(entries.begin(), entries.end(),
                              [](const SubgraphLibraryEntry& a, const SubgraphLibraryEntry& b) {
                                  return a.canonical_expression == b.canonical_expression;
                              }),
                  entries.end());

    return entries;
}

} // namespace aria
