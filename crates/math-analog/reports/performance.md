# Math-analog realtime performance report

Run the bounded callback timing and allocation checks with:

```text
rtk cargo test -p math-analog --test performance --offline -- --nocapture
rtk cargo test -p math-analog --test realtime --offline
```

Captured release timing output on 2026-09-27:

```text
math-analog SIMD criterion: channels=6 scalar_ns=3324250 simd_ns=230537 available=true
math-analog SIMD criterion: channels=12 scalar_ns=4239995 simd_ns=445644 available=true
math-analog worst callback: 3497678 ns, model=2, channels=12
```

The timing fixture covers model IDs 0–8 at 1, 2, 6, and 12 channels with
2,048-frame callbacks at 48 kHz. That callback period is 42,666,667 ns. The
local provisional realtime guard is 25% of the callback period, or
10,666,667 ns; the current release worst case is 8.20% of the period and
therefore passes this synthetic fixture gate. Debug timing is report-only; the
release test is the hard bound. The allocation fixture warms the same matrix
and asserts zero allocations and reallocations during eight steady-state
callbacks. The worst case remains model 2 (Hammerstein) at 12 channels on a
silent block: the Newton-solved component models converge in one evaluation
on silence.

The SIMD criterion compares the runtime-dispatched recurrence with an opaque
test-only scalar baseline so the release compiler cannot auto-vectorize both
sides. In this capture the SIMD path is 14.4× faster at six channels and 9.5×
faster at twelve channels, with exact order-5 output equality.

This is a reproducible engineering guard for the declared benchmark machine,
not a universal CPU budget or hardware-model evidence. The value is
machine- and scheduler-sensitive; release acceptance still requires rerunning
the fixture on the selected target machine and confirming the 25% budget.
