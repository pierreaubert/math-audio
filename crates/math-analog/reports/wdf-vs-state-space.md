# Component evaluation spike decision record

Status: preparatory spike only; no component library is committed.

The first comparison target is a single diode clipper. A bounded state-space
implementation is the current baseline because it has a small fixed state,
deterministic convergence, and a straightforward finite fallback. A WDF
implementation remains the accuracy comparison candidate when a published
diode reference or measured unit is supplied.

The shared `solve_bounded_nonlinear` utility now provides the common capped
Newton/fixed-point contract: at most 64 iterations, finite residual checks,
bound projection, and a click-safe non-converged result. The comparison must
record CPU/sample, convergence failures, and error against the same reference
matrix before adopting a WDF or component abstraction.

No diode, triode, or tone-stack model is enabled without an external schematic
or measured reference and a pre-registered tolerance.
