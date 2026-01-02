<!-- markdownlint-disable-file MD013 -->

# Math-Audio: a toolkit for audio application

## Install

### Cargo

Install [rustup](https://rustup.rs/) first.

If you already have cargo / rustup, you can jump to:

```shell
cargo install just
just
```

Select the correct install just command for your platform:
```shell
just install-...
```

You can build or test with a simple:
```shell
just build
just test
just qa
```

and you are set up.

## Toolkit

### math-testfunctions

A [set of functions](math-test-functions/README.md) for testing non linear optimisation algorithms used in the next crate.

### math-differential-evolution

A implementation of [differential evolution algorithm](math-differential-evolution/README.md) (forked from Scipy) with an interface to NLopt and MetaHeuristics two libraries that also provide various optimisation algorithms. DE support linear and non-linear constraints and implement other features like JADE or adaptative behaviour.

Status: good for speaker equalisation. Not tested enough for other use cases.

### math-iir-fir

An IIR and FIR filter implementation in rust. Does what you expect. Compatible with Equalizer APO. It can generate various output formats.

Status: stable and working well.

### math-solvers

A set of classical solvers with preconditionners that use LAPACK, BLAS and rayon for parallelisation. Support sparse matrices.
Also can work in WASM which is convenient for web demos.

Status: correct and relatively fast but not optimised to death. WASM needs rust nightly to run in parallel.

### math-wave

A set of functions to compute known analytical solutions of the wave equation.

Status: correct.

### math-xem-common

Implement BEM and FEM for the Helmotz and wave equations. Support multigrid for both systems. XEM holds the common code.

Status: unknown, results match analytical results on simple mesh. Needs more testing especially for the advance features.

### math-convexhull3d

This crate computes a convex hull in 3d.

Status: good quality aka no known bug.

### math-bem

This crate implements a BEM (Boundary Element Method) solver, see [BEM README](math-bem/README.md)

Status: ok-ish

### math-fem

This crate implements a FEM (Finite Element Method) solver, see [FEM README](math-fem/README.md)

Status: ok-ish

