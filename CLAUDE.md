# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Math-Audio is a Rust workspace containing numerical computing libraries for acoustic simulation and audio processing. The primary focus is BEM (Boundary Element Method) and FEM (Finite Element Method) solvers for the Helmholtz equation, used in room acoustics simulations.

## Build and Test Commands

```bash
# Install dependencies (run once)
just install-macos    # macOS with Homebrew
just install-linux    # Linux (Ubuntu/Debian)

# Build
just build            # Build release (alias for prod)
just dev              # Build debug

# Test
just test             # cargo check + cargo test --lib
just ntest            # Run with nextest (parallel, no-fail-fast)

# Run single test
cargo test -p math-fem test_name
cargo test -p math-bem test_name -- --nocapture

# QA suites (integration tests)
just qa               # Run both FEM and BEM QA
just qa-fem           # FEM QA suite only
just qa-bem           # BEM QA suite only

# Format and lint
just fmt              # Format all code
cargo clippy --workspace  # Lint (strict clippy enabled in workspace)
```

## Running Simulators

```bash
# FEM room simulator
cargo run --release --bin roomsim-fem --features "cli native" -- --config room.json

# BEM room simulator
cargo run --release --bin roomsim --features="cli native parallel memory-optimized out-of-core" -- --config config.json
```

## Architecture

### Workspace Crates

| Crate | Purpose |
|-------|---------|
| `math-bem` | Boundary Element Method with FMM acceleration |
| `math-fem` | Finite Element Method with multigrid solvers |
| `math-solvers` | GMRES, CG, MINRES, preconditioners (ILU, AMG, Schwarz) |
| `math-xem-common` | Shared types for BEM/FEM (room geometry, sources, configs) |
| `math-wave` | Analytical solutions for wave/Helmholtz equations |
| `math-iir-fir` | IIR/FIR filters, biquads, PEQ, EqualizerAPO export |
| `math-differential-evolution` | DE optimization algorithm |
| `math-test-functions` | Test functions for optimization |
| `math-convex-hull` | 3D convex hull (Quickhull) |

### Key Dependency Flow

```
math-bem → math-xem-common → math-solvers → ndarray
math-fem → math-xem-common → math-solvers
         → math-wave (validation)
```

### BEM Architecture (math-bem/src/)

- `core/assembly/`: Traditional BEM (`tbem.rs`) and FMM (`slfmm.rs`) assembly
- `core/solver/`: GMRES, ILU preconditioner, batched BLAS for FMM
- `core/integration/`: Gaussian quadrature for singular/near-singular integrals
- `room_acoustics/`: Room simulation config, mesh generation, high-level solvers

### FEM Architecture (math-fem/src/)

- `assembly/`: Stiffness/mass matrix, `HelmholtzAssembler` for efficient frequency sweeps
- `basis/`: Lagrange polynomials (P1, P2, P3)
- `boundary/`: Dirichlet, Neumann, Robin (impedance) conditions
- `mesh/`: Box and L-shaped mesh generators
- `multigrid/`: V-cycle and W-cycle geometric multigrid
- `solver/`: Iterative solve with warm-starting

### Solvers Architecture (math-solvers/src/)

- `direct/`: LU decomposition
- `iterative/`: GMRES, CG, MINRES implementations
- `preconditioners/`: ILU, AMG, Schwarz domain decomposition
- `sparse/`: Sparse matrix support (CSR format)
- `traits.rs`: Solver and preconditioner interfaces

## Feature Flags

Important features to know when building:

- `native`: Enables BLAS/LAPACK and rayon parallelism
- `parallel`: Explicit rayon parallelization
- `cli`: CLI dependencies (clap, etc.)
- `memory-optimized`: Memory efficiency for large problems (BEM)
- `out-of-core`: Streaming for very large BEM problems
- `wasm`: WebAssembly support (requires nightly for parallel)

## Solver Selection (BEM)

| Problem Size | Solver Method |
|--------------|---------------|
| N < 1000 | `"direct"` (LU) |
| N < 5000 | `"gmres+ilu"` |
| N < 20000 | `"fmm+gmres+ilu"` |
| N > 20000 | `"fmm+batched"` |

## Mathematical Context

- **Helmholtz equation**: ∇²p + k²p = 0 (acoustic pressure)
- **Burton-Miller formulation**: Used in BEM to avoid spurious resonances
- **FMM**: Single-Level Fast Multipole Method reduces O(N²) to O(N log N)
- **Robin boundary conditions**: Used for wall absorption/impedance in acoustics

## Configuration Files

Both simulators use JSON configs. See:
- `math-fem/docs/room_config_schema.json` - FEM config schema
- `math-fem/docs/room_simulator.md` - FEM room simulator documentation
- `math-bem/configs/` - Example BEM configurations

## Rust Edition and Toolchain

- Edition: 2024
- Toolchain: 1.92.0 (pinned in `rust-toolchain.toml`)
- Strict clippy lints enabled at workspace level
