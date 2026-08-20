#include "subgraph_library.h"
#include "constants.h"
#include "node.h"
#include <algorithm>
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
                auto bin = [](double v, double mn, double rg) { int i = (int)((v-mn)/rg*KB); return std::max(0, std::min(KB-1, i)); };
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
        if (!entries_[i].fingerprint.arity_compatible(needed.num_inputs)) continue;
        matches.push_back({i, fingerprint_distance(needed, entries_[i].fingerprint)});
    }
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) { return a.distance < b.distance; });
    if (matches.size() > top_k) matches.resize(top_k);
    return matches;
}

bool SubgraphLibrary::save(const std::string& filepath) const {
    std::ofstream f(filepath);
    if (!f) return false;
    f << entries_.size() << "\n";
    for (const auto& e : entries_) {
        const auto& fp = e.fingerprint;
        f << std::quoted(e.source_task) << "\t" << std::quoted(e.description)
          << "\t" << std::quoted(e.canonical_expression)
          << "\t" << std::quoted(e.pattern) << "\n";
        f << fp.num_inputs << " " << fp.num_outputs << " "
          << fp.mean << " " << fp.var << " " << fp.min_val << " " << fp.max_val << " "
          << fp.bound_ratio << " " << fp.poly_r2 << " "
          << fp.max_linear_coef << " " << fp.max_nonlin_coef << " " << fp.max_coef_index << " "
          << fp.interaction_dominant << " " << fp.interact_a << " " << fp.interact_b << " "
          << fp.sobol_pairwise << " " << fp.bounded << " " << fp.sharp_boundary << " "
          << fp.sign_symmetric << " " << fp.lipschitz_max << " "
          << fp.quadrant_means.size();
        for (auto q : fp.quadrant_means) f << " " << q;
        f << "\n";
    }
    return true;
}

bool SubgraphLibrary::load(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f) return false;
    size_t n; f >> n;
    entries_.clear();
    for (size_t i = 0; i < n; ++i) {
        SubgraphLibraryEntry e;
        f >> std::quoted(e.source_task) >> std::quoted(e.description)
          >> std::quoted(e.canonical_expression) >> std::quoted(e.pattern);
        auto& fp = e.fingerprint;
        f >> fp.num_inputs >> fp.num_outputs
          >> fp.mean >> fp.var >> fp.min_val >> fp.max_val
          >> fp.bound_ratio >> fp.poly_r2
          >> fp.max_linear_coef >> fp.max_nonlin_coef >> fp.max_coef_index
          >> fp.interaction_dominant >> fp.interact_a >> fp.interact_b
          >> fp.sobol_pairwise >> fp.bounded >> fp.sharp_boundary
          >> fp.sign_symmetric >> fp.lipschitz_max;
        size_t qn; f >> qn;
        fp.quadrant_means.resize(qn);
        for (size_t j = 0; j < qn; ++j) f >> fp.quadrant_means[j];
        entries_.push_back(e);
    }
    return true;
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
