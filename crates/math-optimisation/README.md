<!-- markdownlint-disable-file MD013 -->

# Math Audio Differential Evolution

This crate provides a pure Rust implementation of Differential Evolution (DE) global optimization algorithm with advanced features including L-SHADE.

## Features

- **Pure Rust Implementation**: No external dependencies for core optimization
- **Multiple DE Strategies**: Various mutation and crossover strategies
- **L-SHADE**: Linear population size reduction for faster convergence
- **External Archive**: Maintains diversity by storing discarded solutions
- **Current-to-pbest/1**: SHADE mutation strategy for balanced exploration
- **Constraint Handling**: Linear and nonlinear constraint support
- **Adaptive Parameters**: Self-adjusting F and CR parameters (SHADE-style)
- **Evaluation Recording**: Track optimization progress and convergence
- **Visualization Tools**: Plot test functions and optimization traces

## Optimization Strategies

### Mutation Strategies

- `DE/rand/1`: `x_trial = x_r1 + F * (x_r2 - x_r3)`
- `DE/best/1`: `x_trial = x_best + F * (x_r1 - x_r2)`
- `DE/current-to-best/1`: Combines current and best vectors
- `DE/rand/2`: Uses five random vectors for mutation
- `DE/current-to-pbest/1`: Blends current with top-p% individual (SHADE)

### L-SHADE Strategy

L-SHADE combines three key improvements:
1. **Linear Population Reduction**: Starts with large population (18×dim), reduces to minimum (4)
2. **External Archive**: Stores discarded solutions for diversity
3. **Current-to-pbest/1**: Selects pbest from top p% of population

```rust
use math_audio_optimisation::{
    differential_evolution, DEConfigBuilder, Strategy, LShadeConfig
};

let bounds = vec![(-5.0, 5.0); 10];  // 10D problem

let lshade_config = LShadeConfig {
    np_init: 18,      // Initial NP = 18 × dim = 180
    np_final: 4,      // Final NP = 4
    p: 0.11,          // Select from top 11%
    arc_rate: 2.1,    // Archive size = 2.1 × current NP
    ..Default::default()
};

let config = DEConfigBuilder::new()
    .maxiter(500)
    .strategy(Strategy::LShadeBin)
    .lshade(lshade_config)
    .seed(42)
    .build()
    .expect("invalid config");

let result = differential_evolution(
    &|x| x.iter().map(|&xi| xi * xi).sum(),
    &bounds,
    config,
).expect("optimization should succeed");
```

### Crossover Strategies

- **Binomial**: Random parameter-wise crossover
- **Exponential**: Sequential parameter crossover

## Usage

```rust
use math_audio_optimisation::{differential_evolution, DEConfigBuilder, Strategy, Mutation};
use ndarray::Array1;

// Example objective function (Rosenbrock)
let objective = |x: &Array1<f64>| {
    let a = 1.0;
    let b = 100.0;
    (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
};

// Define bounds for 2D problem
let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

let config = DEConfigBuilder::new()
    .strategy(Strategy::Rand1Bin)
    .maxiter(1000)
    .popsize(50)
    .mutation(Mutation::Factor(0.8))
    .recombination(0.9)
    .seed(42)
    .build()
    .expect("invalid config");

let result = differential_evolution(&objective, &bounds, config)
    .expect("optimization should succeed");
println!("Best solution: {:?}", result.x);
println!("Best fitness: {}", result.fun);
```

## Constraint Support

### Linear Constraints

```rust
use math_audio_optimisation::{LinearConstraintHelper, DEConfig};
use ndarray::{Array1, Array2};

// Linear constraint: x1 + x2 <= 1.0
let constraint = LinearConstraintHelper {
    a: Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap(),
    lb: Array1::from_vec(vec![f64::NEG_INFINITY]),
    ub: Array1::from_vec(vec![1.0]),
};

// Apply to configuration with penalty weight
let mut config = DEConfig::default();
constraint.apply_to(&mut config, 1000.0); // penalty weight
```

### Nonlinear Constraints

```rust
let nonlinear_constraint = |x: &[f64]| -> f64 {
    x[0].powi(2) + x[1].powi(2) - 1.0 // circle constraint
};
```

## Constrained global optimisation

For constrained non-convex problems (`g(x) <= 0`), from cheapest to most
sample-efficient:

- `cobyla` multistart: cheap local baseline, no surrogate.
- `cma_es` with `CmaEsConstraint` + `CmaEsConfig::adapt_penalty_weight`:
  robust native-constraint baseline with IPOP restarts (`max_restarts`) and an
  optional diagonal (`CmaCovariance::Diagonal`) mode for high dimensions.
- `cobra` (`CobraConfig`): SACOBRA-style RBF-surrogate optimiser, one surrogate
  per constraint plus objective, distance-requirement cycle, self-adjusting
  output transform, stagnation kicks, COBYLA polish. Best default when
  evaluations are expensive.
- `constrained_bayesian_optimization` (`ConstrainedBayesOptConfig`):
  single-trust-region constrained BO with constrained EI. Most sample-efficient
  on low-dimensional multimodal problems; GP cost caps it at a few hundred
  evaluations.
- `RbfSurrogate` (`surrogate`): shared RBF models (cubic/Gaussian/multiquadric)
  with holdout model selection for custom surrogate loops.

Head-to-head on `benchmark-constrained` (G06, G08, G24, G04, G09,
rosenbrock-disk; budget rungs 10-200 evals/dim, 3 seeds, success = feasible
within per-problem tolerance of best known):

| solver | problems solved | rung successes |
|---|---|---|
| cobra | 5/6 | 30/90 |
| cobyla-ms | 3/6 | 24/90 |
| cmaes+nat | 2/6 | 12/90 |
| cmaes+pen | 2/6 | 10/90 |
| scbo (<=250 evals) | 2/6 | 12/63 |
| bo+pen | 0/6 | 1/60 |
| isres | 0/6 | 2/90 |
| lshade+pen | 0/6 | 1/90 |

COBRA wins on smooth constrained problems (including 5-D G04 and 7-D G09);
SCBO wins on multimodal low-dimensional ones (G08, rosenbrock-disk) at tiny
budgets. Run with `cargo run --release -p math-optimisation --bin
benchmark-constrained`.

## Visualization

The crate includes a `plot_functions` binary for visualizing test functions and optimization traces:

```bash
# Plot test functions as contour plots
cargo run --bin plot_functions -- --functions rosenbrock,sphere

# Show optimization traces from CSV files
cargo run --bin plot_functions -- --csv-dir traces/ --show-traces
```

## Levenberg-Marquardt

The crate also includes a Levenberg-Marquardt local optimizer for nonlinear least-squares refinement:

```rust
use math_audio_optimisation::levenberg_marquardt::{levenberg_marquardt, LMConfigBuilder};
use ndarray::array;

let residual = |x: &ndarray::Array1<f64>| x.clone();
let bounds = vec![(-10.0, 10.0); 2];
let config = LMConfigBuilder::new()
    .x0(array![3.0, -2.0])
    .maxiter(100)
    .build();

let result = levenberg_marquardt(&residual, &bounds, config).unwrap();
```

## Integration

This crate is part of the Math Audio ecosystem:

- Used by `autoeq` for filter parameter optimization
- Integrates with `math-test-functions` for validation
- Works with `math-iir-fir` for audio filter optimization

## Examples

The crate includes several example programs demonstrating different DE capabilities:

- `optde_basic`: Simple unconstrained optimization
- `optde_adaptive_demo`: Adaptive mutation strategy
- `optde_linear_constraints`: Linear constraint handling
- `optde_nonlinear_constraints`: Nonlinear constraint optimization
- `optde_parallel`: Parallel population evaluation

## Performance Tips

1. **Use L-SHADE for high-dimensional problems** (>10 dimensions)
2. **Start with default LShadeConfig** and tune p if needed
3. **Lower p values (0.05-0.1)** favor exploitation
4. **Higher p values (0.15-0.25)** favor exploration

## [References](./REFERENCES.md)

## License

GPL-3.0-or-later
