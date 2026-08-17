// tests/main_tests.cpp — reasoning-pipeline regression tests for GP-NN.
//
// These tests exercise the internal Phase 3-5 reasoning (blackboard, diagnose,
// complexity profile, candidate generation) that loss-only benchmarks cannot
// reach. Each test is a regression guard for a specific latent bug found
// during the diagnostic investigation:
//
//   test_blackboard_includes_inputs  — INPUT nodes used to be EXCLUDED from
//                                      blackboard_registry_, so search_black-
//                                      board() was empty on fresh graphs and
//                                      MULTIPLY_INJECTION was never emitted.
//   test_targets_are_real            — compute_targets used to look up
//                                      sample.targets with the graph node id
//                                      instead of the CSV data id, so target
//                                      defaulted to 0 for every sample and the
//                                      profiler fitted noise.
//   test_interaction_detection       — the complexity profiler used to be
//                                      fooled by its own degree-2 poly fit
//                                      (poly_r2~1.0) for y=x0*x1; it must now
//                                      flag interaction_dominant and decode
//                                      the (x0,x1) pair.
//   test_multiply_emitted_on_fresh   — on a fresh INPUT->NEURON->OUTPUT graph
//                                      for y=x0*x1, generate_candidates() must
//                                      include MULTIPLY_INJECTION (needs both
//                                      fixes above).
//
// Build (no external deps):
//   g++ -O2 -std=c++17 tests/main_tests.cpp src/node.cpp src/graph.cpp \
//       src/evolution.cpp src/logger.cpp src/serialize.cpp -o tests/gpnn_tests
//   ./tests/gpnn_tests   # exits 0 on success, 1 on any failure

#include "../../src/evolution.h"
#include "../../src/logger.h"
#include "../../src/constants.h"
#include <iostream>
#include <functional>
#include <random>
#include <string>
#include <cmath>

using namespace gpnn;

// ---- Friend accessor into EvolutionEngine's private reasoning pipeline ----
// Defined INSIDE namespace gpnn so it matches the `friend class GpnnTestAccess;`
// declaration in evolution.h (which forward-declares gpnn::GpnnTestAccess). One
// friend line grants access to every private member and nested type, so tests
// can inspect the reasoning layer without enlarging the public API.
namespace gpnn {
class GpnnTestAccess {
public:
    static const std::unordered_map<uint64_t, std::vector<Value>>& blackboard(const EvolutionEngine& e) {
        return e.blackboard_registry_;
    }
    static std::vector<EvolutionEngine::FailureDiagnosis> diagnose(EvolutionEngine& e) {
        return e.diagnose();
    }
    static EvolutionEngine::FailureType classify(const EvolutionEngine& e,
                                                 const EvolutionEngine::FailureDiagnosis& d) {
        return e.classify_failure(d);
    }
    static EvolutionEngine::ComplexityProfile profile(const EvolutionEngine& e,
                                                      const EvolutionEngine::FailureDiagnosis& d) {
        return e.compute_complexity_profile(d);
    }
    static std::vector<EvolutionEngine::BlackboardSignal> search_blackboard(EvolutionEngine& e,
                                                                             const EvolutionEngine::FailureDiagnosis& d) {
        return e.search_blackboard(d);
    }
    static std::vector<EvolutionEngine::Hypothesis> candidates(
        EvolutionEngine& e,
        const EvolutionEngine::FailureDiagnosis& d,
        const std::vector<EvolutionEngine::BlackboardSignal>& b,
        EvolutionEngine::FailureType f,
        const EvolutionEngine::ComplexityProfile& p) {
        return e.generate_candidates(d, b, f, p);
    }
    // Helpers that name private nested types/enumerators on the caller's
    // behalf (test functions aren't friends and can't name Hypothesis).
    static bool has_multiply_candidate(const std::vector<EvolutionEngine::Hypothesis>& cands) {
        for (const auto& h : cands)
            if (h.type == EvolutionEngine::Hypothesis::MULTIPLY_INJECTION) return true;
        return false;
    }
    // Build a synthetic FailureDiagnosis for pure unit tests of the profiler.
    static EvolutionEngine::FailureDiagnosis make_diag(
        uint64_t node,
        std::vector<Value> targets,
        std::vector<std::unordered_map<uint64_t, Value>> local_inputs) {
        EvolutionEngine::FailureDiagnosis d;
        d.failing_node       = node;
        d.mean_blame         = 0;
        d.is_constant_output = false;
        d.has_dead_branch    = false;
        d.targets            = std::move(targets);
        d.local_inputs       = std::move(local_inputs);
        return d;
    }
};
} // namespace gpnn

namespace {
int g_pass = 0, g_fail = 0;
void check(bool cond, const std::string& name, const std::string& detail = "") {
    if (cond) { std::cout << "[PASS] " << name << "\n"; ++g_pass; }
    else      { std::cout << "[FAIL] " << name << (detail.empty() ? "" : "  — " + detail) << "\n"; ++g_fail; }
}

Dataset make_dataset(const std::function<Value(const std::vector<Value>&)>& f,
                     int n_inputs, int n_samples, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Dataset d;
    for (int i = 0; i < n_samples; ++i) {
        Graph::SampleIODesc s;
        std::vector<Value> x(n_inputs);
        for (int j = 0; j < n_inputs; ++j) { x[j] = u(rng); s.inputs[static_cast<uint64_t>(j)] = x[j]; }
        s.targets[0] = f(x);
        d.samples.push_back(std::move(s));
    }
    return d;
}

std::vector<uint64_t> input_node_ids(const EvolutionEngine& e) {
    std::vector<uint64_t> ids;
    for (const auto& n : e.get_graph().get_nodes())
        if (n->get_type() == NodeType::INPUT) ids.push_back(n->get_id());
    return ids;
}

EvolutionEngine::Config basic_cfg(int max_epochs, unsigned seed) {
    EvolutionEngine::Config cfg;
    cfg.max_epochs = max_epochs;
    cfg.seed       = seed;
    return cfg;
}
} // namespace

// ============================================================================
// Test 1: INPUT nodes must appear in blackboard_registry_ after evaluation.
// ============================================================================
void test_blackboard_includes_inputs() {
    auto full = make_dataset([](const std::vector<Value>& x){ return x[0] * 0.5; }, 1, 60, 7);
    Dataset train, val; full.split(train, val, 0.2, 0, true);
    EvolutionEngine e(std::make_unique<Graph>(), std::move(train), std::move(val), basic_cfg(3, 101));
    e.evolve({});
    auto ids = input_node_ids(e);
    check(!ids.empty(), "blackboard: graph has >=1 INPUT node");
    const auto& bb = GpnnTestAccess::blackboard(e);
    bool all_present = !ids.empty();
    for (auto id : ids) if (!bb.count(id)) all_present = false;
    check(all_present, "blackboard includes INPUT node outputs",
          all_present ? "" : "an INPUT node id is missing from blackboard_registry_");
}

// ============================================================================
// Test 2: diagnosis targets must reflect the real dataset targets (non-constant
// variance), regressing the target=0 lookup bug.
// ============================================================================
void test_targets_are_real() {
    // y = x0*x1 — a single NEURON cannot fit this, so the model stays at a
    // plateau and diagnose() surfaces the bottleneck via the structural-
    // inability path. compute_targets must then read the real per-sample target.
    auto full = make_dataset([](const std::vector<Value>& x){ return x[0] * x[1]; }, 2, 80, 11);
    Dataset train, val; full.split(train, val, 0.2, 0, true);
    EvolutionEngine e(std::make_unique<Graph>(), std::move(train), std::move(val), basic_cfg(2, 202));
    e.evolve({});
    auto diags = GpnnTestAccess::diagnose(e);
    check(!diags.empty(), "targets: diagnose returns >=1 failure after plateau");
    if (diags.empty()) return;
    const auto& t = diags[0].targets;
    check(!t.empty(), "targets: diagnosis has a target vector");
    if (t.empty()) return;
    double mean = 0; for (auto v : t) mean += v; mean /= static_cast<double>(t.size());
    double var = 0;  for (auto v : t) var += (v - mean) * (v - mean); var /= static_cast<double>(t.size());
    // Real x0*x1 over [-1,1]^2 has Var ~ 1/9 ~ 0.11. The target=0 bug gave var ~ 0.
    check(var > 1e-4, "targets have real variance (not the target=0 bug)",
          "var=" + std::to_string(var));
}

// ============================================================================
// Test 3: complexity profiler must flag y=x0*x1 as interaction-dominant and
// decode the (x0,x1) pair — pure unit test of compute_complexity_profile.
// ============================================================================
void test_interaction_detection() {
    auto full = make_dataset([](const std::vector<Value>& x){ return x[0] * x[1]; }, 2, 60, 31);
    Dataset train, val; full.split(train, val, 0.2, 0, true);
    EvolutionEngine e(std::make_unique<Graph>(), std::move(train), std::move(val), basic_cfg(1, 303));
    e.evolve({});
    auto ids = input_node_ids(e);
    check(ids.size() >= 2, "interaction: graph has >=2 INPUTs");
    if (ids.size() < 2) return;

    // Synthesize a clean (x0,x1) -> x0*x1 mini-dataset for the profiler.
    std::mt19937 rng(31337);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int N = 200;
    std::vector<Value> targets; targets.reserve(N);
    std::vector<std::unordered_map<uint64_t, Value>> li; li.reserve(N);
    for (int i = 0; i < N; ++i) {
        double a = u(rng), b = u(rng);
        targets.push_back(a * b);
        std::unordered_map<uint64_t, Value> m;
        m[ids[0]] = a; m[ids[1]] = b;
        li.push_back(std::move(m));
    }
    auto diag = GpnnTestAccess::make_diag(ids[0], targets, li);
    auto prof = GpnnTestAccess::profile(e, diag);
    check(prof.interaction_dominant, "interaction: profiler flags interaction_dominant for y=x0*x1",
          "poly_r2=" + std::to_string(prof.poly_r2) +
          " max_coef_idx=" + std::to_string(prof.max_coef_index));
    check(prof.interact_a == 0 && prof.interact_b == 1,
          "interaction: decoded pair is (x0,x1)",
          "got (" + std::to_string(prof.interact_a) + "," + std::to_string(prof.interact_b) + ")");
}

// ============================================================================
// Test 4: on a fresh graph for y=x0*x1, generate_candidates() must include
// MULTIPLY_INJECTION (requires blackboard non-empty + interaction detection).
// ============================================================================
void test_multiply_emitted_on_fresh() {
    auto full = make_dataset([](const std::vector<Value>& x){ return x[0] * x[1]; }, 2, 80, 41);
    Dataset train, val; full.split(train, val, 0.2, 0, true);
    // 2 outer epochs: enough to populate the blackboard and plateau, but
    // before the first structural commit (~epoch 4) so the graph is still
    // the fresh INPUT->NEURON->OUTPUT shape.
    EvolutionEngine e(std::make_unique<Graph>(), std::move(train), std::move(val), basic_cfg(2, 404));
    e.evolve({});
    auto diags = GpnnTestAccess::diagnose(e);
    check(!diags.empty(), "multiply-emitted: diagnose returns >=1 failure");
    if (diags.empty()) return;
    auto& d = diags[0];

    auto signals = GpnnTestAccess::search_blackboard(e, d);
    check(!signals.empty(), "multiply-emitted: blackboard non-empty on fresh graph",
          "size=" + std::to_string(signals.size()));

    auto ftype = GpnnTestAccess::classify(e, d);
    auto prof  = GpnnTestAccess::profile(e, d);
    auto cands = GpnnTestAccess::candidates(e, d, signals, ftype, prof);
    bool has_multiply = GpnnTestAccess::has_multiply_candidate(cands);
    check(has_multiply, "multiply-emitted: MULTIPLY_INJECTION present in candidates",
          "n_candidates=" + std::to_string(cands.size()));
}

// ============================================================================
// Test 5: recurrent BPTT — a self-recurrent NEURON must train on a sequence
// whose target depends on the previous timestep (running sum). Confirms the
// train() backward pass routes gradient through delay_buffer (teacher-forced
// k=1 BPTT). If this fails, recurrent weights get no gradient and d6 is
// unreachable regardless of any structural hypothesis.
// ============================================================================
void test_recurrent_bptt_trains() {
    auto g = std::make_unique<Graph>();
    uint64_t in_id  = g->add_node(NodeType::INPUT, "in");
    uint64_t n_id   = g->add_node(NodeType::NEURON, "n");
    uint64_t out_id = g->add_node(NodeType::OUTPUT, "out");

    Node* n = g->get_node(n_id);
    if (n && n->get_type() == NodeType::NEURON) {
        auto* nn = static_cast<NeuronNode*>(n);
        nn->set_input_count(2);
        nn->set_weight(0, 0.1);   // external input
        nn->set_weight(1, 0.1);   // self-recurrent input
        nn->set_bias(0.0);
    }
    Node* on = g->get_node(out_id);
    if (on && on->get_type() == NodeType::OUTPUT) {
        static_cast<OutputNode*>(on)->set_scale(1.0);
        static_cast<OutputNode*>(on)->set_bias(0.0);
    }
    g->add_connection(in_id, 0, n_id, 0);    // external input  -> port 0
    g->add_connection(n_id,  0, n_id, 1);    // self-recurrent  -> port 1 (cycle)
    g->add_connection(n_id,  0, out_id, 0);  // neuron -> output

    // Sequence: running sum of small inputs. out_t ~= out_{t-1} + x_t needs the
    // self-loop to carry the previous output. Order matters -> recurrence.
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> u(-0.1, 0.1);
    Dataset seq;
    double acc = 0.0;
    for (int t = 0; t < 40; ++t) {
        double x = u(rng);
        acc += x;
        Graph::SampleIODesc s;
        s.inputs[0]  = x;
        s.targets[0] = acc;
        seq.samples.push_back(std::move(s));
    }
    std::unordered_map<uint64_t, uint64_t> in_map{{0, in_id}}, out_map{{0, out_id}};

    auto loss = [&]() -> double {
        g->reset_recurrent_state();
        double total = 0.0;
        for (const auto& s : seq.samples) {
            g->set_input_value(in_id, s.inputs.at(0));
            g->execute();
            Value pred = g->get_output_value(out_id);
            double d = pred - s.targets.at(0);
            total += d * d;
        }
        return total / static_cast<double>(seq.samples.size());
    };

    double before = loss();
    Graph::TrainConfig tc;
    tc.epochs         = 400;
    tc.learning_rate  = 0.05;
    tc.gradient_clip  = 1e3;
    tc.momentum       = 0.0;
    tc.weight_decay   = 0.0;
    tc.loss_type      = Graph::LossType::MSE;
    tc.input_data_to_graph  = in_map;
    tc.output_data_to_graph = out_map;
    g->train(seq.samples, tc);
    double after = loss();

    check(after < before * 0.5, "recurrent BPTT: training reduces sequence loss",
          "before=" + std::to_string(before) + " after=" + std::to_string(after));
}

int main() {
    Logger::init(false, "");  // silence engine logs during tests
    std::cout << "=== GP-NN reasoning-pipeline tests ===\n";
    test_blackboard_includes_inputs();
    test_targets_are_real();
    test_interaction_detection();
    test_multiply_emitted_on_fresh();
    test_recurrent_bptt_trains();
    std::cout << "----------------------------------------\n";
    std::cout << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
