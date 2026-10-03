# Shared DSP forks

`rubato` is the single SOTF Rubato 5.0.0 fork. The DAW and capture workspaces
refer to this path directly; it is excluded from the math-audio workspace so
the math crates do not acquire a resampling dependency. Read
[`rubato/SOTF_FORK.md`](rubato/SOTF_FORK.md) before updating its upstream base
or changing its realtime behavior.
