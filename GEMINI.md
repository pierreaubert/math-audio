# Math-Audio

## Project Overview

`math-audio` is a Rust workspace containing a comprehensive suite of numerical computing libraries tailored for acoustic simulation and audio processing. The core components are BEM (Boundary Element Method) and FEM (Finite Element Method) solvers for the Helmholtz equation, designed for room acoustics simulations. It also includes tools for optimization (Differential Evolution), convex hull computation, and filter design (IIR/FIR).

## Building and Running

The project uses `just` as a command runner.

### Prerequisites
*   Rust (managed via `rustup`)
*   `just` (`cargo install just`)
*   System dependencies (BLAS/LAPACK, etc.) - see `Justfile` for platform-specific install commands.

### Key Commands

| Command | Description |
| :--- | :--- |
| `just build` (or `just prod`) | Build the workspace in release mode. |
| `just dev` | Build the workspace in debug mode. |
| `just test` | Run unit tests (`cargo test --lib`). |
| `just ntest` | Run tests using `nextest` (parallel, fast). |
| `just fmt` | Format all code using `rustfmt`. |
| `just qa` | Run integration quality assurance suites for both FEM and BEM. |
| `just qa-fem` | Run FEM QA suite only. |
| `just qa-bem` | Run BEM QA suite only. |

### Running Simulators

**FEM Room Simulator:**
```bash
cargo run --release --bin roomsim-fem --features "cli native" -- --config room.json
```

**BEM Room Simulator:**
```bash
cargo run --release --bin roomsim --features="cli native parallel memory-optimized out-of-core" -- --config config.json
```

## Architecture

### Workspace Crates

| Crate | Purpose |
| :--- | :--- |
| `math-bem` | Boundary Element Method solver with FMM acceleration. |
| `math-fem` | Finite Element Method solver with multigrid support. |
| `math-solvers` | Iterative solvers (GMRES, CG, MINRES) and preconditioners (ILU, AMG). |
| `math-xem-common` | Shared data structures for BEM/FEM (geometry, sources, configs). |
| `math-wave` | Analytical solutions for the wave and Helmholtz equations. |
| `math-iir-fir` | IIR and FIR filter implementations, compatible with EqualizerAPO. |
| `math-differential-evolution`| Differential Evolution optimization algorithm. |
| `math-test-functions` | Test functions for validating optimization algorithms. |
| `math-convex-hull` | 3D convex hull implementation (Quickhull). |

### Dependency Flow
*   **Core Math:** `math-solvers` depends on `ndarray`.
*   **Simulation Core:** `math-xem-common` depends on `math-solvers`.
*   **Solvers:**
    *   `math-bem` → `math-xem-common`
    *   `math-fem` → `math-xem-common`

## Development Conventions

*   **Rust Edition:** 2024
*   **Toolchain:** Pinned in `rust-toolchain.toml` (currently 1.92.0).
*   **Linting:** Strict `clippy` lints are enabled at the workspace level in `Cargo.toml`.
*   **Testing:**
    *   Unit tests are located within `src/` or `tests/` in each crate.
    *   Integration tests ("QA suites") are run via `just qa`.
*   **Feature Flags:**
    *   `native`: Enables BLAS/LAPACK and `rayon` parallelism (Recommended for performance).
    *   `parallel`: Enables explicit `rayon` parallelization.
    *   `wasm`: WebAssembly support.

## Mathematical Context

*   **Helmholtz Equation:** $\nabla^2 p + k^2 p = 0$. Solved for acoustic pressure frequency response.
*   **Burton-Miller:** Formulation used in BEM to mitigate spurious resonances.
*   **FMM:** Single-Level Fast Multipole Method used to accelerate BEM matrix-vector products ($O(N \log N)$).
*   **Multigrid:** Geometric multigrid (V-cycle/W-cycle) used to accelerate FEM convergence.
