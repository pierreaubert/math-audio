# math-autodiff

Frequency-domain differentiable audio DSP.

`math-autodiff` provides LTI audio modules whose complex frequency response is
differentiable with respect to their parameters. Optimization is performed in
the frequency domain using analytical gradients, so no time-domain automatic
differentiation framework is required.

## Modules

| Module | Description |
|--------|-------------|
| `fft` | Real-to-complex FFT / inverse-FFT differentiable wrapper. |
| `gain` | Frequency-independent real gain matrices and magnitude output. |
| `delay` | MIMO (`Delay`) and diagonal per-channel (`ParallelDelay`) frequency-domain delays. |
| `matrix` | Learnable real matrices: dense (`Dense`) and orthogonal (`Orthogonal`) parameterizations. |
| `recursion` | Closed-loop composition (`Recursion`) for feedback systems such as FDNs. |
| `iir::biquad` | Differentiable RBJ biquad filter. |
| `iir::sos_filter` | Generic learnable cascade of second-order sections. |
| `iir::svf` | State Variable Filter parameterized by `fc`/`R`/`gain`. |
| `iir::geq` | Graphic EQ with fixed ISO frequencies and learnable per-band gains. |
| `iir::peq` | Parametric EQ with learnable frequency, Q, and gain per section. |
| `system` | `Series`, `Parallel`, and `Shell` compositions. |
| `signals` | Signal generators: impulse, step, noise, sine sweep, wav file loader. |
| `loss` | MSE loss and backward helper. |
| `optim` | Simple SGD optimizer. |

All modules implement `DiffModule<f64>` and expose `forward`, `backward`,
`parameters`, `parameters_mut`, `gradients`, and `zero_grad`.

## Example: FDN magnitude-response matching

The `fdn_match` example learns an FDN core that matches a target magnitude
response. The model is a `Shell` composed as `Fft -> FDN -> Magnitude`:

```bash
cargo run --release -p math-autodiff --example fdn_match
```

The example builds a 4-channel FDN from `ParallelDelay` delay lines, an
orthogonal feedback matrix, and a scalar feedback gain wrapped in `Recursion`.
It optimizes the delay values and feedback matrix to minimize the MSE between
the predicted and target magnitude responses.

A second example, `biquad_match`, fits a single biquad section:

```bash
cargo run --release -p math-autodiff --example biquad_match
```

A third example, `peq_match`, fits a parametric EQ cascade to a target
magnitude response:

```bash
cargo run --release -p math-autodiff --example peq_match
```

A fourth example, `svf_match`, fits a State Variable Filter to a target
magnitude response:

```bash
cargo run --release -p math-autodiff --example svf_match
```

A fifth example, `geq_match`, fits a graphic EQ to a target magnitude response:

```bash
cargo run --release -p math-autodiff --example geq_match
```

A sixth example, `fdn_direct_match`, matches an FDN core plus a separate
direct path using the `Parallel` combiner:

```bash
cargo run --release -p math-autodiff --example fdn_direct_match
```

## Benchmarks

```bash
cargo bench -p math-autodiff --bench biquad_bench
```

## Tests

```bash
cargo test -p math-autodiff --release
```
