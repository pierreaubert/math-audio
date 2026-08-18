# Math-analog realtime performance report

Run the bounded callback timing and allocation checks with:

```text
rtk cargo test -p math-analog --test performance --offline -- --nocapture
rtk cargo test -p math-analog --test realtime --offline
```

Captured release timing output on 2026-08-18:

```text
math-analog SIMD criterion: channels=6 scalar_ns=559625 simd_ns=134959 available=true
math-analog SIMD criterion: channels=12 scalar_ns=1372292 simd_ns=269916 available=true
math-analog worst callback: 1689542 ns, model=2, channels=12
```

The timing fixture covers model IDs 0–5 at 1, 2, 6, and 12 channels with
2,048-frame callbacks at 48 kHz. That callback period is 42,666,667 ns. The
local provisional realtime guard is 25% of the callback period, or
10,666,667 ns; the current release worst case is 3.96% of the period and
therefore passes this synthetic fixture gate. Debug timing is report-only; the
release test is the hard bound. The allocation fixture warms the same matrix
and asserts zero allocations and reallocations during eight steady-state
callbacks.

The SIMD criterion compares the runtime-dispatched recurrence with an opaque
test-only scalar baseline so the release compiler cannot auto-vectorize both
sides. In this capture the SIMD path is 4.1× faster at six channels and 5.1×
faster at twelve channels, with exact order-5 output equality.

This is a reproducible engineering guard for the declared benchmark machine,
not a universal CPU budget or hardware-model evidence. The value is
machine- and scheduler-sensitive; release acceptance still requires rerunning
the fixture on the selected target machine and confirming the 25% budget.
