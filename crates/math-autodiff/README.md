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
| `system` | `Series` and `Shell` compositions. |
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

## Benchmarks

```bash
cargo bench -p math-autodiff --bench biquad_bench
```

## Tests

```bash
cargo test -p math-autodiff --release
```
