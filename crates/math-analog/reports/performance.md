# Math-analog realtime performance report

Run the bounded callback timing and allocation checks with:

```text
rtk cargo test -p math-analog --test performance --offline -- --nocapture
rtk cargo test -p math-analog --test realtime --offline
```

Captured timing output on 2026-08-13 (current rerun):

```text
math-analog worst callback: 5861875 ns, model=2, channels=12
```

The timing fixture covers model IDs 0–4 at 1, 2, 6, and 12 channels with
2,048-frame callbacks at 48 kHz. That callback period is 42,666,667 ns. The
local provisional realtime guard is 25% of the callback period, or
10,666,667 ns; the current worst case is 13.74% of the period and therefore
passes this synthetic fixture gate. The allocation fixture warms the same
matrix and asserts zero allocations and reallocations during eight
steady-state callbacks.

This is a reproducible engineering guard for the declared benchmark machine,
not a universal CPU budget or hardware-model evidence. The value is
machine- and scheduler-sensitive; release acceptance still requires rerunning
the fixture on the selected target machine and confirming the 25% budget.
