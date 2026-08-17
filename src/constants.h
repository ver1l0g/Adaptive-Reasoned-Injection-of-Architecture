// ============================================================================
// constants.h 鈥?Centralized configuration constants for the GP NN system
// ============================================================================
#pragma once

#include "node.h"

#include <cstddef>
#include <limits>

namespace aria {

// ============================================================================
// Numeric type
// ============================================================================
using Value = double;

namespace config {

// ============================================================================
// Evolution defaults
// ============================================================================
constexpr int   DEFAULT_POPULATION_LIMIT       = 50;
constexpr int   DEFAULT_EVAL_BUDGET            = 20;
constexpr int   DEFAULT_MAX_GENERATIONS        = 50;
constexpr int   DEFAULT_MUTATION_DEPTH         = 1;   // 1 = classic single-step mutation
constexpr int   DEFAULT_MUTATION_MAX_CHILDREN  = 3;   // 1..N random branching factor
constexpr int   DEFAULT_INITIAL_ID             = 1;
constexpr int   DEFAULT_GRAPH_NODE_ID_START    = 1;

// ============================================================================
// Random distribution ranges
// ============================================================================
constexpr double RANDOM_VALUE_MIN   = -1.0;   // for new CONSTANT nodes
constexpr double RANDOM_VALUE_MAX   =  1.0;
constexpr double RANDOM_SMALL_MIN   = -0.5;   // for value perturbation
constexpr double RANDOM_SMALL_MAX   =  0.5;

// ============================================================================
// Elitism 鈥?carry best individuals unchanged to next generation
// ============================================================================
constexpr int    ELITE_COUNT                  = 5;

// ============================================================================
// Restart 鈥?population re-seed on plateau
// ============================================================================
constexpr int    MAX_RESTARTS                 = 5;     // max restarts before giving up

// ============================================================================
// Selection
// ============================================================================
constexpr int    DEFAULT_TOURNAMENT_SIZE      = 3;     // tournament selection pressure

// ============================================================================
// Loss-guided breeding (Pattern #1 鈥?XGBoost LossGuide-inspired PQ expansion)
// ============================================================================
// Instead of uniform tournament selection, a priority queue ranks parents by
// fitness improvement potential: fitness + LOSS_GUIDE_DELTA_WEIGHT * (fitness - parent_fitness).
// When enabled, single-step mode allocates more offspring slots to higher-priority parents.
constexpr bool   LOSS_GUIDED_BREEDING_ENABLED = true;  // enable loss-guided PQ parent selection
constexpr double LOSS_GUIDE_DELTA_WEIGHT      = 0.3;   // weight on parent鈫抍hild fitness delta
constexpr double LOSS_GUIDE_DECAY            = 0.85;   // repeated parent selection priority decay

// ============================================================================
// Crossover 鈥?recombination of subgraphs across parents
// ============================================================================
constexpr double CROSSOVER_RATE             = 0.3;    // probability of crossover vs pure mutation
constexpr double CROSSOVER_UNIFIED_WEIGHT   = 0.5;    // blame weight in unified signal for crossover output selection
constexpr int    CROSSOVER_MAX_ATTEMPTS     = 5;      // retries if crossover produces oversize graph

// ============================================================================
// Adaptive tournament 鈥?tournament size adjusts to population diversity
// ============================================================================
constexpr int    ADAPTIVE_TOURNAMENT_MIN     = 2;     // min tournament size (high diversity)
constexpr int    ADAPTIVE_TOURNAMENT_MAX     = 7;     // max tournament size (low diversity)
constexpr int    ADAPTIVE_TOURNAMENT_INTERVAL = 5;    // gens between recalculating diversity

// ============================================================================
// Diversity 鈥?fitness sharing + crowding to maintain structural variety
// ============================================================================
constexpr double DIVERSITY_SHARING_SIGMA    = 0.15;   // sharing radius for fitness penalty
constexpr double DIVERSITY_SHARING_ALPHA    = 1.0;    // shape parameter (1=linear)
constexpr int    DIVERSITY_CHECK_INTERVAL   = 5;      // gens between computing structural hashes
constexpr double DIVERSITY_STRUCTURAL_THRESHOLD = 0.6; // similarity above which individuals are considered duplicates

// ============================================================================
// Cross-output block reuse 鈥?use blocks from other outputs when none exists
// ============================================================================
constexpr bool   CROSS_OUTPUT_BLOCK_ENABLED  = true;  // enable cross-output block lookup
constexpr double CROSS_OUTPUT_BLOCK_BLAME_THRESHOLD = -1.0; // min |blame| for cross-output replacement

// ============================================================================
// Blame-based selection 鈥?prioritise harmful nodes/connections for mutation
// ============================================================================
constexpr double BLAME_SELECTION_FLOOR_WEIGHT = 0.1;   // minimum weight for non-harmful entities

// Dual-gradient architecture (XGBoost-inspired): separate structure gradient
// from value gradient.  When enabled, structural mutations (add/remove/rewire)
// use structure_grad_map_, value mutations use value_grad_map_.
constexpr bool DUAL_GRADIENT_ENABLED = true;
constexpr int    OUTPUT_ERROR_MAX_SAMPLES     = 10;    // max samples for output-error-weighted selection

// ============================================================================
// Gradient-guided mutation
// ============================================================================
constexpr double GRADIENT_LEARNING_RATE       = 0.1;   // step size for gradient descent on values
constexpr double GRADIENT_NOISE_STDEV         = 0.05;  // exploration noise (std dev) mixed with gradient step
constexpr double LOGIC_STE_TEMPERATURE        = 1.0;   // softness for Straight-Through Estimator on logic/comparison nodes

// ============================================================================
// Error attribution 鈥?perturbation-based blame analysis
// ============================================================================
constexpr double ATTRIBUTION_DEFAULT_EPSILON       = 0.1;   // default perturbation epsilon
constexpr int    ATTRIBUTION_MAX_SAMPLES           = 15;    // max samples for attribution in mutate() (was 10)
constexpr int    ATTRIBUTION_MAX_CANDIDATES        = 15;    // max candidate nodes for blame analysis
constexpr double CONSTANT_OUTPUT_BLAME_BOOST       = -10.0; // blame floor for constant-output nodes

// ============================================================================
// Constant output detection 鈥?degenerate-output identification
// ============================================================================
constexpr double CONSTANT_OUTPUT_SAME_SIGN_THRESHOLD = 0.5;   // sigmoid midpoint threshold
constexpr double CONSTANT_OUTPUT_VARIANCE_THRESHOLD  = 0.001; // variance below which output is constant

// ============================================================================
// Training 鈥?SGD on weights/biases/constants
// ============================================================================
constexpr int    DEFAULT_TRAIN_EPOCHS               = 10;
constexpr double DEFAULT_TRAIN_LEARNING_RATE        = 0.01;  // default for Adam (10x smaller than SGD's 0.1)
constexpr double DEFAULT_TRAIN_GRADIENT_CLIP        = 5.0;

// SGD divergence detection: if a single SGD phase makes the loss worse by
// more than this factor, the graph is restored to its pre-training state.
// Prevents the runaway divergence that produces Class B blowups (e.g. t23
// reaching loss 1.16 from a starting loss of ~0.01 on an unlucky seed).
constexpr double SGD_DIVERGENCE_FACTOR              = 3.0;
constexpr double EVOLVE_DIVERGENCE_FACTOR           = 5.0;   // global safety net threshold
constexpr int    EVOLVE_DIVERGENCE_PATIENCE          = 3;     // consecutive epochs above threshold before restore
constexpr double DEFAULT_TRAIN_MOMENTUM             = 0.0;   // 0 = pure SGD; 0.9 = momentum SGD
constexpr double DEFAULT_TRAIN_WEIGHT_DECAY         = 0.0;   // 0 = no L2 reg; typical 5e-4

// ============================================================================
// Search improvements 鈥?node-type replacement search
// ============================================================================
constexpr int    SEARCH_IMPROVEMENT_DEFAULT_TOP_K   = 8;
constexpr int    SEARCH_IMPROVEMENT_NUM_SUBSETS     = 3;
constexpr int    SEARCH_IMPROVEMENT_NUM_INSERT_TRIALS = 5;
constexpr int    SEARCH_IMPROVEMENT_MAX_SAMPLES     = 50;
constexpr int    SEARCH_IMPROVEMENT_TRIAL_SUBSET    = 10;   // samples for quick trial pre-screen in search_improvements

// ============================================================================
// Progressive evaluation gating 鈥?two-stage fitness evaluation
// ============================================================================
constexpr int    FITNESS_QUICK_EVAL_SUBSET          = 10;   // sample count for quick-fitness pre-screen
constexpr int    PROGRESSIVE_EVAL_QUICK_MULTIPLIER = 4;    // quick-eval 4x more individuals than eval_budget, then full-eval top eval_budget

// ============================================================================
// Importance-based protection
// ============================================================================
constexpr double IMPORTANCE_PROTECTION        = 1.0;   // divisor in weight = 1/(1+|imp|*protection)

// ============================================================================
// Fitness sentinels & thresholds
// ============================================================================
constexpr double FITNESS_UNEVALUATED          = -1.0;
constexpr double FITNESS_SELECTION_THRESHOLD  =  0.0;   // minimum for roulette eligibility
constexpr double FITNESS_ROOT_PARENT          =  0.0;   // default parent_fitness for roots

// ============================================================================
// Mutation limits
// ============================================================================
constexpr int    MUTATION_ADD_CONNECTION_MAX_ATTEMPTS = 10;
constexpr int    MUTATION_NUM_OPERATORS               =  6;
constexpr size_t MUTATION_REMOVE_NODE_MAX_REWIRE      =  4;  // upstream/downstream buffer
constexpr int    MUTATION_CHANGE_TYPE_MAX_REROLL       = 20;  // max retries to pick a different type
constexpr double ADD_NODE_SHIFTED_PORT_PROBABILITY    = 0.5; // probability of using a random (non-zero) input port when adding a node
constexpr int    MUTATION_SEQUENCE_MIN                =  1;  // min operators per mutate() call
constexpr int    MUTATION_SEQUENCE_MAX                =  3;  // max operators per mutate() call

// ============================================================================
// Adaptive mutation 鈥?boosted mutation when evolution plateaus
// ============================================================================
// Minimum fitness improvement required to reset the stagnation counter.
// Micro-improvements from value-only tuning (~1e-5 to 1e-4) should NOT count
// as progress 鈥?only structural-level improvements (~5e-4+) reset stagnation.
constexpr double FITNESS_IMPROVEMENT_THRESHOLD        = 0.0005;
constexpr int    ADAPTIVE_MUTATION_THRESHOLD          = 30;  // gens without improvement before boost
constexpr int    MUTATION_SEQUENCE_BOOST_MIN          =  3;  // min operators per mutate() when boosted
constexpr int    MUTATION_SEQUENCE_BOOST_MAX          = 10;  // max operators per mutate() when boosted
constexpr double BLOCK_POSTFIT_VALUE_LIMIT            = 1e6; // revert block insert if any output exceeds this
constexpr double MUTATION_BOOST_PERTURB_SCALE         = 3.0; // scale factor for value perturbations when boosted
constexpr int    MUTATION_SCALE_BASE_NODES            = 30;  // graph size at which mutation count scaling = 1.0脳; larger graphs get sqrt(N/30)脳 more mutations

// ============================================================================
// SGD deep fine-tuning 鈥?longer SGD runs on the elite
// ============================================================================
constexpr int    SGD_FINE_TUNE_INTERVAL               = 50;  // gens between deep fine-tuning sessions
// SGD Top-K fine-tuning 鈥?instead of tuning only the #1 individual, tune top K
constexpr int    SGD_TOP_K_COUNT                     = 3;   // number of top individuals to fine-tune
constexpr int    SGD_TOP_K_EPOCHS                    = 200; // epochs per individual in top-K mode

// ============================================================================
// Restart 鈥?save elite, reinitialize from it with increased population
// ============================================================================
constexpr int    RESTART_STAGNATION_THRESHOLD         = 80;  // gens without improvement before restart
constexpr double RESTART_POPULATION_MULTIPLIER        = 1.15; // conservative growth 鈥?was 1.5 causing runaway 300鈫?50鈫?75
constexpr int    RESTART_POPULATION_MAX               = 2000;// cap to prevent multiplicative population bloat
// Restart mutation range 鈥?higher than regular boost to force genuine topological
// changes during re-seeding.  Regular boost uses [3, 10]; restart uses [5, 15].
constexpr int    RESTART_MUTATION_SEQUENCE_MIN        =  5;
constexpr int    RESTART_MUTATION_SEQUENCE_MAX        = 15;
// Fraction of restart population seeded from diverse sources (non-elite):
// - divert_non_elite / 2 from top historical bests in the population
// - divert_non_elite / 2 from the original MLP template (if factory available)
constexpr double RESTART_DIVERT_NON_ELITE_FRACTION    = 0.40;
// Extra fixed slots added to population limit on each restart (in addition to multiplier growth)
constexpr int    RESTART_POPULATION_INCREMENT         = 10;

// ============================================================================
// Dead-output recovery 鈥?detect and fix outputs that never fire
// ============================================================================
// After this many consecutive stagnant generations, check for dead outputs
// and inject fresh subgraphs to revive them.
constexpr int    DEAD_OUTPUT_STAGNATION_DETECT        = 5;    // consecutive stagnant gens before dead-output check
constexpr int    DEAD_OUTPUT_REBIRTH_NODES            = 3;    // nodes injected per dead output (CONSTANT + ACTIVATION + bridge)
// Extra mutation count per dead output: num_mutations * (1 + DEAD_OUTPUT_ESCALATION_FACTOR * dead_count)
constexpr double DEAD_OUTPUT_ESCALATION_FACTOR        = 0.3;

// ============================================================================
// Architecture perturbation escalator 鈥?force structural change on plateau
// ============================================================================
// When stagnation exceeds this threshold, temporarily restrict mutations to
// structural-only operators (add/remove node, add/remove connection) to force
// the graph topology to escape local optima.
constexpr int    ARCH_PERTURB_THRESHOLD               = 20;   // stagnant gens before forcing structural ops
constexpr int    ARCH_PERTURB_DURATION                = 5;    // gens to keep structural-only mode active
constexpr int    ARCH_PERTURB_MAX_DURATION            = 15;   // max gens for structural-only mode (prevents infinite loop)

// ============================================================================
// Building Block Library 鈥?subgraph extraction and reuse
// ============================================================================
// When |raw_output 鈭?target| < threshold for ALL samples of an output, the
// subgraph feeding that output is extracted and stored as a reusable block.
constexpr double BLOCK_EXTRACTION_ACCURACY_THRESHOLD = 0.3;
// Compose a full graph from best blocks every N generations.
constexpr int    BLOCK_LIBRARY_COMPOSE_INTERVAL      = 10;
// Maximum number of blocks stored per output node.
constexpr int    BLOCK_LIBRARY_MAX_SIZE              = 100;
// Minimum fitness improvement over existing block to replace it.
constexpr double BLOCK_LIBRARY_MIN_IMPROVEMENT       = 1e-6;
// When the block library is empty, relax the accuracy threshold and accept
// lower-quality blocks to bootstrap the library.
constexpr double BLOCK_EXTRACTION_RELAXED_MULTIPLIER  = 2.0;   // multiplier on accuracy threshold when relaxed
constexpr double BLOCK_EXTRACTION_RELAXED_QUALITY     = 0.5;   // quality score weight when relaxed

// ============================================================================
// Memory pressure 鈥?graceful shutdown when system RAM runs low
// ============================================================================
// When free physical RAM drops below this fraction of total RAM, the evolution
// loop calls the MemoryCallback (if set) to save the best graph and exits.
// With 64GB RAM and a 0.20 threshold, the system has ~12.8GB of headroom
// for serialization and I/O before the OS kills the process.
constexpr double MEMORY_CRITICAL_THRESHOLD = 0.20;

// ============================================================================
// Parallel execution
// ============================================================================
constexpr int  EVOLUTION_NUM_THREADS = 20;  // use all 20 cores
constexpr bool EVOLUTION_PARALLEL    = true;

// ============================================================================
// Node type probability weights for mutation operators
// ============================================================================
// Higher weight = more likely to be chosen.
// Used by random_insertable_type() (mutate_add_node) and random_node_type() (mutate_change_type).

constexpr size_t NODE_INSERT_WEIGHT_COUNT = 21;
constexpr NodeType NODE_INSERT_WEIGHT_TYPES[NODE_INSERT_WEIGHT_COUNT] = {
    NodeType::NEGATE,         NodeType::RELU,         NodeType::SIGMOID,   NodeType::TANH,
    NodeType::ADD,            NodeType::SUBTRACT,     NodeType::MULTIPLY,  NodeType::DIVIDE,
    NodeType::NEURON,         NodeType::IF,
    NodeType::EQUAL,          NodeType::NOT_EQUAL,    NodeType::GREATER,   NodeType::LESS,
    NodeType::GREATER_EQUAL,  NodeType::LESS_EQUAL,
    NodeType::AND,            NodeType::OR,            NodeType::NOT,       NodeType::XOR,
    NodeType::ABSENT,
};
constexpr double NODE_INSERT_WEIGHTS[NODE_INSERT_WEIGHT_COUNT] = {
    1.0, 3.0, 3.0, 3.0,   // NEGATE, RELU, SIGMOID, TANH (neural boosted for classification)
    1.0, 1.0, 1.0, 1.0,   // ADD, SUBTRACT, MULTIPLY, DIVIDE
    3.0, 1.0,              // NEURON (boosted), IF
    1.0, 1.0, 1.0, 1.0,   // logic comparisons: EQUAL, NOT_EQUAL, GREATER, LESS
    1.0, 1.0,              // GREATER_EQUAL, LESS_EQUAL
    1.0, 1.0, 1.0, 1.0,   // AND, OR, NOT, XOR
    0.5,                   // ABSENT
};

constexpr size_t NODE_REPLACE_WEIGHT_COUNT = 24;
constexpr NodeType NODE_REPLACE_WEIGHT_TYPES[NODE_REPLACE_WEIGHT_COUNT] = {
    NodeType::CONSTANT,      NodeType::SINK,
    NodeType::ADD,           NodeType::SUBTRACT,     NodeType::MULTIPLY,  NodeType::DIVIDE,
    NodeType::NEGATE,        NodeType::NEURON,       NodeType::RELU,
    NodeType::SIGMOID,       NodeType::TANH,         NodeType::IF,        NodeType::IFELSE,
    NodeType::EQUAL,         NodeType::NOT_EQUAL,    NodeType::GREATER,   NodeType::LESS,
    NodeType::GREATER_EQUAL, NodeType::LESS_EQUAL,
    NodeType::AND,           NodeType::OR,            NodeType::NOT,       NodeType::XOR,
    NodeType::ABSENT,
};
constexpr double NODE_REPLACE_WEIGHTS[NODE_REPLACE_WEIGHT_COUNT] = {
    1.0, 0.3,              // CONSTANT, SINK
    0.4, 0.4, 0.4, 0.4,   // ADD, SUBTRACT, MULTIPLY, DIVIDE (reduced: not useful for raw pixels)
    0.4, 5.0, 5.0,         // NEGATE, NEURON (strongly boosted), RELU (strongly boosted)
    5.0, 5.0, 0.3, 0.3,   // SIGMOID, TANH (strongly boosted), IF, IFELSE (reduced: control-flow harms gradient flow)
    1.0, 1.0, 1.0, 1.0,   // EQUAL, NOT_EQUAL, GREATER, LESS (boosted: enable logic node utilization)
    1.0, 1.0,              // GREATER_EQUAL, LESS_EQUAL
    1.0, 1.0, 1.0, 1.0,   // AND, OR, NOT, XOR (boosted: enable logic node utilization)
    1.0,                   // ABSENT
};

// ============================================================================
// Node / weight initialization
// ============================================================================
constexpr Value  NEURON_DEFAULT_BIAS          = 0.0;
constexpr Value  NEURON_DEFAULT_WEIGHT        = 0.0;
constexpr double WEIGHT_INIT_XAVIER_GAIN      = 0.1;    // reduced from 1.0 to prevent tanh saturation with unscaled inputs
constexpr int    NEURON_MIN_INPUT_COUNT       = 1;
constexpr int    NEURON_MAX_INPUT_PORTS      = 64;     // max input ports for NEURON node type

// ============================================================================
// Graph execution
// ============================================================================
constexpr Value  GRADIENT_SEED               = 1.0;   // dOut/dOut = 1
constexpr Value  GRADIENT_ZERO_THRESHOLD     = 1e-9;
constexpr Value  ZERO_FILL_VALUE             = 0.0;    // for unresolved/cycle inputs

// ============================================================================
// Serialization
// ============================================================================
constexpr int    SERIALIZATION_FORMAT_VERSION = 1;
constexpr size_t DESERIALIZE_MAX_PAIRS        = 10000;  // max key-value pairs in deserialized maps
constexpr size_t DESERIALIZE_MAX_ELEMENTS     = 10000;  // max elements in deserialized arrays

// ============================================================================
// DOT visualization
// ============================================================================
constexpr const char* DOT_HEADER             = "digraph lineage {\n";
constexpr const char* DOT_RANKDIR            = "  rankdir=TB;\n";
constexpr const char* DOT_NODE_STYLE         = "  node [shape=box, style=rounded];\n";
constexpr const char* DOT_ROOT_LABEL         = "  root [label=\"root\", shape=circle];\n";
constexpr const char* DOT_COLOR_UNEVAL       = "gray";
constexpr const char* DOT_COLOR_POSITIVE     = "green";
constexpr const char* DOT_COLOR_ZERO         = "orange";
constexpr int         DOT_FONT_SIZE          = 10;

// ============================================================================
// Fitness evaluation
// ============================================================================
constexpr int    FITNESS_EVAL_SAMPLES        = 30;
constexpr double SIZE_PENALTY_COEFFICIENT    = 0.005;  // relaxed: allow exploration growth (was 0.02, too tight for 134-node baseline)
constexpr int    DEFAULT_MAX_NODE_COUNT      = 200;   // hard cap 鈥?mutation refuses to add beyond this
constexpr double INVALID_EVAL_ERROR          = 1e6;
constexpr Value  INVALID_FITNESS_FALLBACK    = 0.0;

// ============================================================================
// Numerical guards & thresholds
// ============================================================================
constexpr double SAFE_DIV_EPSILON                = 1e-9;   // minimum denominator to avoid div-by-zero
constexpr double GAUSS_PIVOT_TOLERANCE           = 1e-12;  // zero threshold for Gaussian elimination pivot
constexpr double IMPORTANCE_MIN_FLOOR            = 1e-12;  // below this, importance-based selection treats weight as zero
constexpr double IMPORTANCE_WEIGHT_FLOOR         = 1e-12;  // minimum clamped importance weight
constexpr double IMPORTANCE_FALLBACK_WEIGHT      = 1e-6;   // default importance weight when no record exists
constexpr double NEURON_FIT_ATANH_CLIP           = 0.999;  // clip atanh input to avoid infinity at 卤1

// ============================================================================
// Search improvement penalties 鈥?complexity penalties for node-type selection
// ============================================================================
constexpr double SEARCH_REPLACE_SINK_PENALTY    = 1e6;    // SINK node in replace-type: prohibitively expensive
constexpr double SEARCH_REPLACE_ABSENT_PENALTY  = 1.5;    // ABSENT node in replace-type: prefers deletion
constexpr double SEARCH_REPLACE_IFELSE_PENALTY  = 1.2;    // IFELSE node: mild complexity penalty
constexpr double SEARCH_REPLACE_NEURON_PENALTY  = 1.05;   // NEURON node: slight complexity penalty
constexpr double SEARCH_INSERT_ABSENT_PENALTY   = 1.5;    // ABSENT node in insert-type search
constexpr double SEARCH_INSERT_NEURON_PENALTY   = 1.05;   // NEURON node in insert-type search

// ============================================================================
// Blame & weight multipliers 鈥?for gradient-guided mutation selection
// ============================================================================
constexpr double DEPTH_WEIGHT_MIN_FLOOR          = 0.001;  // minimum depth weighting for deep nodes
constexpr double BLAME_WEIGHT_MULTIPLIER         = 10.0;   // boost for error blame in selection weight
constexpr double OUTPUT_ERROR_WEIGHT_MULTIPLIER  = 15.0;   // boost for per-output error contribution
constexpr double OUTPUT_ERROR_BLEND_MULTIPLIER   = 20.0;   // boost when blending error with output selection
constexpr double IMPORTANCE_SCALING_FACTOR       = 10.0;   // scale factor for importance in target weight
constexpr double IMPORTANCE_SELECTION_MULTIPLIER = 5.0;    // per-connection importance weight multiplier
constexpr double DST_WEIGHTS_MIN_FLOOR           = 0.01;   // minimum clamped destination weight
constexpr double VALUES_GUIDED_MIN_FACTOR        = 0.05;   // minimum multiplier for guided value mutation
constexpr double VALUES_INITIAL_WEIGHT           = 0.01;   // initial weight for value-mutation target sampling

// ============================================================================
// Retry / attempt limits
// ============================================================================
constexpr int    TRY_SRC_FOR_DST_MAX_ATTEMPTS    = 5;      // max retries to find input source for output dest
constexpr int    MOVE_CONNECTION_MAX_ATTEMPTS    = 10;     // retries to find distinct src/dst for connection move
constexpr int    SEARCH_REWIRE_MAX_ATTEMPTS      = 50;     // retries to find a new source for rewire

// ============================================================================
// Mutation probability thresholds
// ============================================================================
constexpr double MOVE_CONN_RANDOM_OUTPUT_PROB     = 0.3;    // probability of random output when moving connections
constexpr double INSERT_NEURON_SECOND_INPUT_PROB   = 0.5;    // chance of connecting second input when inserting NEURON
constexpr double OUTPUT_DIVERSIFY_ACTIVATION_PROB  = 0.8;    // probability of random activation for output diversity
constexpr double ANALYZE_ERRORS_THRESHOLD          = 0.3;    // minimum |error| for analyze_errors() notification

// ============================================================================
// UCB operator dispatch 鈥?multi-armed bandit selection of mutation operators
// ============================================================================
// Operators are selected via weighted sampling where the weight for operator i is:
//   w_i = rate + UCB_WEIGHT_FLOOR + exploration + fitness_bonus + recency + guidance_bonus
constexpr double UCB_NEUTRAL_PRIOR              = 0.5;    // success rate for unexplored operators
constexpr double UCB_FITNESS_DELTA_SCALE        = 5.0;    // scale factor: avg_fitness_delta 鈫?bonus
constexpr double UCB_FITNESS_DELTA_CAP          = 2.0;    // maximum fitness-delta bonus
constexpr double UCB_MAX_EXPLORATION_CAP        = 2.0;    // cap on combined exploration multiplier
constexpr double UCB_EXPLORATION_PREFACTOR      = 2.0;    // prefactor for sqrt term in exploration
constexpr double UCB_MAX_RECENCY_CAP            = 0.3;    // cap on recency bonus
constexpr double UCB_RECENCY_LOG_FACTOR         = 0.05;   // log multiplier for recency
constexpr double UCB_GUIDANCE_BONUS_SCALE       = 1.5;    // scale factor for operator guidance bonus
constexpr double UCB_WEIGHT_FLOOR               = 0.01;   // minimum selection weight per operator

// ============================================================================
// Hashing
// ============================================================================
constexpr uint32_t STRUCTURAL_HASH_GOLDEN_RATIO     = 0x9e3779b9;         // golden ratio for 32-bit hash combine
constexpr uint64_t THREAD_RNG_SEED_MULTIPLIER       = 0x9E3779B97F4A7C15ULL; // golden-ratio-like 64-bit seed

// ============================================================================
// Credit operators 鈥?dynamic structural vs value credit allocation
// ============================================================================
// At progress p (0鈫?), structural credit weight = CREDIT_STRUCT_MAX_WEIGHT * (1-p),
// value credit weight = CREDIT_VALUE_MIN_WEIGHT + p * (1 - CREDIT_VALUE_MIN_WEIGHT).
constexpr double CREDIT_STRUCT_MAX_WEIGHT           = 2.0;   // maximum structural credit weight (at progress=0)
constexpr double CREDIT_VALUE_MIN_WEIGHT            = 0.5;   // minimum value credit weight (at progress=0)

// ============================================================================
// Runner utilities
// ============================================================================
constexpr const char* OUTPUT_DIR_PREFIX             = "results/";       // prefix for output directories
constexpr double BINARY_CLASSIFICATION_THRESHOLD     = 0.5;    // sigmoid threshold for binary classification
constexpr double DEFAULT_ACCURACY_TOLERANCE          = 0.01;   // default tolerance for accuracy comparison
constexpr int    PROGRESS_BAR_DEFAULT_WIDTH          = 20;     // default width for ASCII progress bar

// ============================================================================
// EvolutionEngine::Config defaults
// ============================================================================
// NOTE: these intentionally differ from the legacy DEFAULT_TRAIN_* constants
// above (which were tuned for the population-based mutation system). The
// 7-phase EvolutionEngine uses higher SGD momentum (0.9) and more epochs.
constexpr int    DEFAULT_EVOLUTION_MAX_EPOCHS         = 500;   // outer evolve() loop iterations
constexpr int    DEFAULT_SGD_EPOCHS_PER_PHASE         = 50;    // inner SGD epochs per Phase 2 / per shadow validation
constexpr int    DEFAULT_SGD_BATCH_SIZE               = 32;    // mini-batch size (0 = full-batch)
constexpr double DEFAULT_SGD_MOMENTUM                 = 0.9;   // momentum SGD (Phase 2 + shadow validation)
constexpr double DEFAULT_SGD_WEIGHT_DECAY             = 1e-5;  // L2 reg (small, non-zero)
constexpr int    DEFAULT_PLATEAU_PATIENCE             = 3;     // SGD plateau generations before triggering Phase 3
constexpr double DEFAULT_PLATEAU_MIN_IMPROVEMENT      = 1e-3;  // min loss reduction to count as improvement

// High-dimensional scaling: for tasks with many inputs (e.g. 64-pixel MNIST),
// the base model needs more SGD epochs to converge before structural search
// interrupts. Scale plateau_patience with input count so the engine doesn't
// interrupt SGD before the linear map is learned.
constexpr int    HIGH_DIM_INPUT_THRESHOLD             = 20;    // above this, scale patience
constexpr double HIGH_DIM_PATIENCE_PER_INPUT          = 0.2;   // extra patience per input above threshold
constexpr int    DEFAULT_FORCE_STRUCTURAL_EVERY       = 20;    // epochs without structural change before forcing Phase 3
constexpr int    DEFAULT_MAX_FAILURES_TO_FIX          = 15;    // #hypothesis candidates attempted per failure (multi-output needs more)
constexpr int    DEFAULT_BLACKBOARD_MAX_CANDIDATES    = 20;    // top-N correlated signals kept by Phase 4
constexpr double DEFAULT_VALIDATION_THRESHOLD         = 0.0;   // Phase 7 commit requires shadow_val < base_val - threshold
constexpr bool   DEFAULT_COMPILE_ENABLED              = true;  // Phase 6 graph compilation on/off
constexpr int    DEFAULT_COMPILE_INTERVAL             = 10;    // generations between Phase 6 compiles

// Determinism seed (0 => random_device, nonzero => reproducible). Fixed default
// so out-of-the-box runs are reproducible; override via --seed.
constexpr unsigned int DEFAULT_SEED                    = 12345;

// Outer-loop early stopping: if best_overall_loss_ has not improved for this
// many consecutive epochs AND no structural commit happened in that window,
// stop evolve(). 0 disables. Trims wasted epochs on easy/noisy tasks
// (d9 noise, converged t-tasks) and bounds thrash on hard ones. The counter
// also resets on any structural commit (see evolve), so tasks whose progress
// comes in bursts after long structural plateaus are not cut off mid-search.
constexpr int    DEFAULT_EARLY_STOP_PATIENCE          = 100;

// Structural cooldown: when this many consecutive structural cycles fail to
// commit anything, skip structural search for DEFAULT_STRUCTURAL_COOLDOWN
// epochs (let SGD settle on the current architecture before trying again).
// Directly targets the 700+ failed-commit thrash seen on hard diagnostic tasks.
constexpr int    DEFAULT_STRUCTURAL_FAILURE_THRESHOLD = 5;
constexpr int    DEFAULT_STRUCTURAL_COOLDOWN          = 10;

// ============================================================================
// Dataset split / loss sentinels
// ============================================================================
constexpr double DEFAULT_VALIDATION_FRACTION          = 0.2;   // train/val split fraction held out for Phase 7
// Loss sentinel: initialize best_phase2_loss_ / best_overall_loss_ so any
// real loss is smaller. NOTE: 1e9 is intentionally not infinity 鈥?kept for
// behavior compatibility. Switch to std::numeric_limits<double>::max() only
// if losses can legitimately exceed 1e9 (currently they cannot).
constexpr double LOSS_SENTINEL_INF                    = 1e9;

// ============================================================================
// Numerical guards (backward pass, BCE log, plateau detection)
// ============================================================================
constexpr double BCE_LOG_CLAMP_EPSILON                = 1e-9;  // clamp sigmoid output before log() to avoid 卤inf
constexpr double BACKWARD_DIV_EPSILON                 = 1e-10; // DivideNode backward denom floor (smaller than SAFE_DIV_EPSILON)
constexpr double DIVIDE_FORWARD_EPSILON               = 1e-6;  // DivideNode forward |denominator| floor (sign-preserving clamp)
constexpr double PLATEAU_RELATIVE_DENOM_FLOOR         = 1e-8;  // floor for |loss_before| in relative-improvement ratio
constexpr double DEAD_BRANCH_SIGNAL_THRESHOLD         = 1e-6;  // |output| below this 鈫?IFELSE branch never fires
constexpr int    ATTRIBUTION_SEED_RANGE               = 100000;// modulus for rng_() % N seed generation in diagnose()

// ============================================================================
// classify_failure thresholds (Phase 3 failure-type detection)
// ============================================================================
constexpr double BOOLEAN_GAP_RATIO_THRESHOLD          = 0.25;  // sorted-target gap ratio above which to test cluster split
constexpr double BOOLEAN_BETWEEN_WITHIN_RATIO         = 2.5;   // F-ratio (between/within variance) for BOOLEAN_BOUNDARY classification
constexpr int    LINEAR_FIT_MAX_FEATURES              = 10;    // cap upstream-feature count for least-squares linear fit
constexpr double LINEAR_FIT_R2_THRESHOLD              = 0.75;  // R虏 above which failure is classified LINEAR_OFFSET
constexpr double CONTEXT_WIRE_CORR_TRIGGER            = 0.5;   // min Pearson r to consider CONTEXT_WIRE in form_hypothesis()

// ============================================================================
// ComplexityProfile diagnostic (Phase 3 鈥?bottleneck signature analysis)
// ============================================================================
// Step 1 of the compound-hypothesis roadmap: compute a low-cost "fingerprint"
// of the residual at the bottleneck node so hypothesis selection can be
// driven by data instead of pure correlation. Used to detect signatures like
// "bounded + oscillating + high pairwise interaction" 鈫?sin(x路y), which
// requires a MULTIPLY鈫扤EURON_TANH compound template rather than a single
// hypothesis.
constexpr int    PROFILE_SOBOL_BINS                   = 5;     // discretization bins per axis for Sobol pairwise index
constexpr double PROFILE_PAIRWISE_INTERACTION_MIN     = 0.30;  // V_int / Var(r) above which "high pairwise interaction" flag is set
constexpr double PROFILE_BOUNDED_RATIO_MAX            = 4.0;   // (max-min)/stddev below this 鈬?residual is "bounded" (e.g., sin)
constexpr double PROFILE_SHARP_BOUNDARY_LIPSCHITZ     = 5.0;   // axis Lipschitz above this 鈬?"sharp boundary" (IFELSE signature)

// Compound-hypothesis detection (Phase 5 鈥?COMPOUND_MULTIPLY_NEURON trigger).
// Fires when the polynomial degree-2 fit shows a dominant cross-term (a_ij)
// AND the residual is bounded (suggesting a nonlinear activation on top of
// the product). This is the signature of sin(x路y), sigmoid(x路y), tanh(x路y).
constexpr double PROFILE_COMPOUND_CROSS_TERM_RATIO    = 0.20;  // |a_ij| / max(|linear coef|) above this 鈬?cross-term is structurally meaningful
constexpr double SCORE_COMPOUND_MULTIPLY_NEURON       = 0.96;  // ranks above all single-hypothesis candidates when signature matches
constexpr int    SHADOW_COMPOUND_SGD_MULTIPLIER       = 3;     // extra SGD budget for compound shadows
constexpr double SHADOW_COMPOUND_LR_MULTIPLIER        = 0.2;   // reduce LR for compound shadows so zero-init chains grow without disrupting existing weights
constexpr double SHADOW_COMPOUND_VAL_TOLERANCE        = 0.02;  // compound accepted if val_loss <= baseline_val * (1 + tolerance)
constexpr int    COMPOUND_COMMIT_GRACE_EPOCHS         = 50;    // after committing a compound, delay next plateau trigger by this many epochs (chain starts at identity, needs SGD time to grow)

// Compound TANH series 鈥?K parallel NEURON鈫扵ANH chains summed via ADD.
// Triggered when residual is BOUNDED, smooth (low Lipschitz), and still
// high-variance (linear fit R虏 is misleading because the residual oscillates
// around zero 鈥?signature of needing more curve capacity, e.g., sin(x)).
constexpr double SCORE_COMPOUND_TANH_SERIES          = 0.94;  // below COMPOUND_MULTIPLY_NEURON (0.96), above LINEAR_OFFSET fallbacks
constexpr double COMPOUND_TANH_VAR_MIN               = 0.05;  // min residual variance to trigger (lower = residual already small)
constexpr double COMPOUND_TANH_LIPSCHITZ_MAX         = 2.0;   // residual must be smooth (low 1st-derivative magnitude)
constexpr int    COMPOUND_TANH_SERIES_K              = 3;
constexpr int    COMPOUND_TANH_SERIES_K_MAX          = 12;    // adaptive-K ceiling

// SIN_INJECTION input limit: above this many inputs, the sign-change detection
// in SIN_INJECTION's emission gate is unreliable (high-dimensional BCE residuals
// oscillate from class competition, not from genuine signal oscillation). Disable
// SIN to prevent noise-fitting on tasks like MNIST.
constexpr int    SIN_INJECTION_MAX_INPUTS            = 16;

// COMPOUND_MULTIPLY_ABS sign-symmetry gate: emit only when the target's
// minimum is within this fraction of its range below zero (i.e. essentially
// non-negative, like |x*y|). 0.05 = allow up to 5% negative excursion.
constexpr double ABSMUL_NONNEG_FRACTION              = 0.05;

// RECURRENT_SELF_WIRE: only attempted in sequence mode (--no-shuffle), since
// memory-based targets need temporal order to carry state. Low score so it's
// tried but doesn't dominate; validation rejects it where recurrence doesn't
// help.
constexpr double SCORE_RECURRENT_SELF_WIRE          = 0.10;

// DEEP_INSERTION: serial NEURON鈫扵ANH residual chain after failing node.
// Adds depth (hierarchical feature extraction) rather than width.
// Ranked between NEURON_TANH_INJECTION and compound hypotheses.
constexpr double SCORE_DEEP_INSERTION              = 0.50;

// RECURRENT_XOR: recurrent XOR for running parity/accumulation tasks.
// Only emitted in sequence mode with binary inputs.
constexpr double SCORE_RECURRENT_XOR               = 0.97;

// MULTI_LAYER_STACK: injects a K-width hidden layer + combining neuron.
// Creates a 2-layer MLP: INPUT 鈫?K NEURONs 鈫?NEURON_combine 鈫?OUTPUT.
// Emitted for high-complexity problems (spirals, checkerboard) where
// single-layer + width-expansion can't capture the decision boundary.
constexpr double SCORE_MULTI_LAYER_STACK           = 0.60;
constexpr int    MULTI_LAYER_STACK_K               = 4;     // hidden width
constexpr int    MULTI_LAYER_STACK_K_MAX           = 8;     // ceiling for adaptive K
constexpr int    MULTI_LAYER_STACK_MAX_INPUTS      = 128;   // don't stack on huge input dims (too many new weights)
constexpr double MULTI_LAYER_STACK_BOOST_LIPSCHITZ = 100.0; // lipschitz above this 鈫?boost score to 0.95
constexpr int    MULTI_LAYER_STACK_SGD_MULTIPLIER  = 6;     // extra SGD budget (stacks need more training)

// PATCH_POOLING: local block averages for image-like inputs. Creates
// patch_size脳patch_size average-pool LINEAR nodes (uniform 1/k weights 鈥?// SGD refines them into learned filters) and feeds the pooled features
// into the failing node. Gives the graph a coarse convolutional prior:
// CIFAR 32脳32 gray 鈫?64 pooled 4脳4 blocks + raw pixels.
// Emitted only when input count is a perfect square (image-like layout).
constexpr double SCORE_PATCH_POOLING              = 0.90;
constexpr int    PATCH_POOL_PATCH_SIZE            = 4;     // px per side of each pooled block
constexpr int    PATCH_POOL_MIN_SIDE              = 16;    // min image side (16虏 = 256 inputs)

// PARITY_TREE: linear-fold XOR tree over all binary inputs 鈫?OUTPUT.
// Solves k-bit parity (XOR-5D) in ONE structural change. Greedy
// incremental commits can't build intermediate XORs (each step gives zero
// loss improvement), so the tree must be atomic. Exact function 鈥?no
// training needed; SGD only refines the OUTPUT scale/bias for confidence.
constexpr double SCORE_PARITY_TREE                = 0.98;
constexpr int    PARITY_TREE_MIN_INPUTS           = 3;     // 2-input parity = BOOLEAN_COMPOSE territory
constexpr int    PARITY_TREE_MAX_INPUTS           = 16;    // tree cost is O(k); cap for safety

// DIVIDE_INJECTION: quotient feature a/b for ratio targets (Feynman m/V,
// (u+v)/(1+uv), q虏a虏/c鲁 鈥?the operator MULTIPLY cannot express). Sources
// mirror MULTIPLY_INJECTION (poly fit / Sobol pair / blackboard top-2);
// both orderings (a/b, b/a) emitted with per-candidate denominator safety
// check: min |denominator| over training data must clear a floor relative
// to its std (z-scored inputs live near 0 鈥?raw ratios explode).
constexpr double SCORE_DIVIDE_BOOST                   = 0.94;  // just under MULTIPLY plateau fallback (0.95)
constexpr double SCORE_DIVIDE_BASE                    = 0.06;  // unboosted baseline
constexpr double DIVIDE_DENOMINATOR_MIN_REL           = 0.25;  // min |den| 鈮?this 脳 std(den values)
constexpr double DIVIDE_DENOMINATOR_ABS_FLOOR         = 1e-3;  // or this absolute floor (whichever is larger)

// COMPOUND_SIN_PRODUCT: MULTIPLY(a,b) 鈫?NEURON(freq-init) 鈫?SIN 鈫?ADD.
// For sin-of-product targets (Korns F4 x鈧乻in(x鈧€x鈧?, F8; Feynman I.29.16
// A路sin(kx)). Single-op hypotheses can't express it: SIN_INJECTION takes
// one input; MULTIPLY_INJECTION yields an unbounded product with no
// oscillation. The composition with a FREQUENCY-INITED inner neuron
// (same estimation as SIN_INJECTION freq-init) closes the gap.
constexpr double SCORE_COMPOUND_SIN_PRODUCT           = 0.96;  // above MULTIPLY plateau fallback; below PARITY_TREE

// COMPOUND_DIVIDE_PRODUCT: DIVIDE(MULTIPLY(a,b), c) 鈫?NEURON(zero-init) 鈫?ADD.
// For q虏a虏/c鲁-class targets (Feynman I.32.8, r^3 denominators) where the
// numerator is itself a product 鈥?DIVIDE_INJECTION (raw a/b) and
// MULTIPLY_INJECTION can't express the quotient-of-products. Denominator
// safety-gated like DIVIDE_INJECTION. Zero-gain output neuron gives
// identity start (raw quotients can be large).
constexpr double SCORE_COMPOUND_DIVIDE_PRODUCT       = 0.955; // outranks speculative stack (0.95) in its narrow gate (poly_r2<0.95 + safe denom): the quotient-of-products IS the right first guess there

// Subgraph library: when the needed-behavior fingerprint matches a stored
// entry within this distance, boost the suggested hypothesis type's score.
constexpr double LIBRARY_MATCH_THRESHOLD            = 2.0;
constexpr double LIBRARY_BOOST                      = 0.20;
constexpr double LIBRARY_INJECT_THRESHOLD           = 1.0;   // tighter: inject candidates the gates blocked

// Compound MULTIPLY3 鈥?three-way product for sin(x路y路z)-class functions.
// Triggered when 3+ inputs, bounded residual, but NO single cross-term
// dominates (all pairwise interactions are moderate 鈥?the true feature is
// the 3-way product x路y路z). Score is slightly below COMPOUND_MULTIPLY_NEURON
// since it's a fallback when pairwise analysis fails to find a dominant pair.
constexpr double SCORE_COMPOUND_MULTIPLY3_NEURON     = 0.93;
constexpr double COMPOUND3_POLY_R2_MAX               = 0.85;  // degree-2 fit is poor 鈫?missing higher-order term
constexpr double COMPOUND3_VAR_MIN                   = 0.05;  // residual still has meaningful variance     // number of parallel NEURON鈫扵ANH chains to inject

// ============================================================================
// Hypothesis candidate scoring (Phase 5 鈥?generate_candidates ranking)
// ============================================================================
// Higher score = tried first. The proven NEURON_TANH_INJECTION fallback sits
// at 0.1; new/experimental types (MULTIPLY_INJECTION, BOOLEAN_COMPOSE) sit
// below at 0.05 unless boosted by failure-type signals.
constexpr double SCORE_CW_LINEAR_BOOST                = 0.85;  // CONTEXT_WIRE boost for LINEAR_OFFSET
constexpr double SCORE_CW_BOOLEAN_BOOST               = 0.5;   // CONTEXT_WIRE boost for BOOLEAN_BOUNDARY
constexpr double SCORE_IFELSE_BASE                    = 0.3;   // IFELSE_BOUNDARY_SPLIT score floor
constexpr double SCORE_IFELSE_GAP_WEIGHT              = 0.6;   // IFELSE score: base + weight * gap_ratio
constexpr double SCORE_IFELSE_BOOLEAN_BOOST           = 0.9;   // IFELSE boost for BOOLEAN_BOUNDARY
constexpr double SCORE_IFELSE_LINEAR_BOOST            = 0.55;  // IFELSE boost for LINEAR_OFFSET
constexpr double SCORE_NTI_BASE                       = 0.1;   // NEURON_TANH_INJECTION baseline (proven fallback)
constexpr double SCORE_NTI_NONLINEAR_BOOST            = 0.7;   // NEURON_TANH_INJECTION boost for NON_LINEAR_CURVE
constexpr double SCORE_MULTIPLY_BASE                  = 0.05;  // MULTIPLY_INJECTION baseline (experimental)
constexpr double SCORE_MULTIPLY_NONLINEAR_BOOST       = 0.45;  // MULTIPLY boost for NON_LINEAR_CURVE / LINEAR_OFFSET
constexpr double SCORE_MULTIPLY_INTERACTION_BOOST     = 0.55;  // MULTIPLY boost when two top blackboard signals correlate
constexpr double SCORE_MULTIPLY_PLATEAU_FALLBACK      = 0.95;  // MULTIPLY boost on plateau w/ 鈮? INPUT signals, non-binary (above CONTEXT_WIRE's correlation-based score)
constexpr double SCORE_BOOLEAN_COMPOSE                = 0.65;  // BOOLEAN_COMPOSE score for binary-classification problems
constexpr double MULTIPLY_CORR_TRIGGER                = 0.4;   // min |r| of both top signals to trigger interaction boost
constexpr double BINARY_LABEL_MIN_RANGE               = 0.1;   // min label range to consider binary classification
constexpr double BINARY_LABEL_GAP_RATIO               = 0.4;   // min sorted-label gap ratio for binary split
constexpr double BINARY_LABEL_MAX_WITHIN_VAR          = 0.05;  // max within-cluster variance for binary classification

// ============================================================================
// Shadow validation (Phase 7)
// ============================================================================
constexpr double SHADOW_TRAIN_REGRESSION_FACTOR       = 1.5;   // shadow_train_loss may regress up to baseline * factor (catastrophic-damage guard)
constexpr double SHADOW_TRAIN_MAX_REGRESSION_ABS      = 0.01;  // absolute drift tolerance (used when baseline is small)
constexpr double SHADOW_TRAIN_MAX_REGRESSION_REL      = 0.10;  // relative drift tolerance (used when baseline is large 鈥?10% allows XOR-style cold-start regressions)
constexpr double COMMIT_MIN_IMPROVEMENT               = 1e-9;  // tie-guard epsilon (strict-better)
constexpr double COMMIT_MIN_IMPROVEMENT_FRACTION      = 0.01;  // required improvement 鈮?1% of baseline, capped by validation_threshold
constexpr double COMMIT_RANK_BONUS_FRACTION           = 0.01;  // rank-aware tie-breaking: 1% of current_loss per rank step

// ============================================================================
// Target nudge (compute_targets, importance-zero fallback)
// ============================================================================
constexpr double TARGET_NUDGE_FACTOR                  = 0.1;   // lr * error * factor when no importance signal available

// Target propagation clamp. The ideal failing-node target is computed as
// current_out - error/importance (the value that would zero the output error
// under a local-linear approximation). When importance is near zero this can
// blow up, so the correction is clamped to [-TARGET_PROP_CLAMP, +TARGET_PROP_CLAMP].
constexpr double TARGET_PROP_CLAMP                    = 20.0;

// Profiler interaction-dominance: the dominant non-bias polynomial term
// (a square x_i^2 or cross x_i*x_j) must exceed the largest |linear coef| by
// this factor before we declare the residual "interaction-dominated". Used to
// pick MULTIPLY sources from the polynomial structure (robust where Sobol is
// noisy on low-variance residuals).
constexpr double PROFILE_INTERACTION_RATIO            = 1.0;

// ============================================================================
// Node-execution defaults
// ============================================================================
constexpr double RELU_LEAKY_SLOPE                     = 0.01;  // LeakyReLU slope on x<0 (ReLUNode::backward)
constexpr double OUTPUT_DEFAULT_SCALE                 = 1.0;   // OutputNode trainable scale_ initial value
constexpr double OUTPUT_DEFAULT_BIAS                  = 0.0;   // OutputNode trainable bias_ initial value

// ============================================================================
// Logger configuration
// ============================================================================
constexpr const char* LOG_TIMESTAMP_FORMAT            = "%Y-%m-%d %H:%M:%S";
constexpr const char* LOG_LEVEL_INFO                  = "[INFO]";
constexpr const char* LOG_LEVEL_PHASE                 = "[PHASE]";
constexpr const char* LOG_LEVEL_WARN                  = "[WARN]";
constexpr const char* LOG_LEVEL_VERBOSE               = "[VERBOSE]";
constexpr const char* LOG_LEVEL_DECISION              = "[DECISION]";
constexpr const char* LOG_BANNER_SEPARATOR            = "============================================================\n";

// ============================================================================
// CLI / runner defaults
// ============================================================================
constexpr int    DEFAULT_CLI_MAX_EPOCHS               = 500;   // --max-epochs default (was 500; --help text previously misstated 100)
constexpr int    DEFAULT_OUTPUT_COLS                  = 1;     // --output-cols default
constexpr double DEFAULT_SWEEP_MIN                    = 0.0;   // --sweep min x
constexpr double DEFAULT_SWEEP_MAX                    = 10.0;  // --sweep max x
constexpr double DEFAULT_SWEEP_STEP                   = 0.25;  // --sweep step size
constexpr double SWEEP_RANGE_EPSILON                  = 1e-9;  // FP epsilon for sweep endpoint comparison

// ============================================================================
// Special values
// ============================================================================
constexpr size_t INDEX_INVALID               = static_cast<size_t>(-1);
constexpr double SENTINEL_NEG_INF            = -std::numeric_limits<double>::infinity();

} // namespace config
} // namespace aria