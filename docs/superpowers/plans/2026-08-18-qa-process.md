# Workspace QA Process Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `just qa` run the full QA process for the whole workspace and `just qa-{crate}` run it per crate, mirroring the CI gates.

**Architecture:** Pure Justfile. One private parameterized ladder recipe `_qa crate threshold` (fmt-check → clippy → unit/integration/doc tests → bench compile check → per-crate coverage gate), nine thin public wrappers adding crate-specific extras (fuzzers, examples, bench `--quick` smokes), and a `qa` aggregator that runs all nine plus the two workspace-level CI gates.

**Tech Stack:** Just, cargo, cargo-llvm-cov 0.8.7 (already installed), criterion 0.8 (`--quick` supported), simd-fuzzer binary in math-dsp.

**Spec:** `docs/superpowers/specs/2026-08-18-qa-process-design.md`

## Global Constraints

- The Justfile uses **tabs** for recipe-line indentation, never spaces.
- Do not modify existing recipes (`test`, `ntest`, `fmt`, `lint`, `prod`, `bench`, `examples-*`, `install-*`, `publish-*`). Only replace the QA section (currently Justfile lines 223–230) and append new recipes after it.
- Per-crate coverage thresholds are a ratchet: set each to the measured value from Task 1 rounded DOWN to a whole percent; they may be raised later, never silently lowered.
- The workspace coverage gate is exactly `cargo llvm-cov --lib --summary-only --release --fail-under-lines 90` (mirrors `.github/workflows/ci.yml`).
- The workspace clippy gate is exactly `cargo clippy --all --features plotly -- -D warnings` (same command as the existing `lint` recipe).
- Fuzzers and property tests run at full strength (simd-fuzzer default = 10000 iterations/function; proptest defaults). Benches are compile-checked in the ladder and smoke-run with `--quick` only where the spec lists them.
- After editing the Justfile, verify syntax with `just --list` before running anything heavy.

---

### Task 1: Coverage baselines + `_qa` ladder + geometry crates

**Files:**
- Modify: `Justfile` (replace QA section, lines 223–230)
- Create: `target/qa-coverage-baselines.md` (scratch, gitignored, not committed)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: private recipe `_qa crate threshold`; Just variables `qa_cov_analog`, `qa_cov_autodiff`, `qa_cov_convex_hull`, `qa_cov_delaunay`, `qa_cov_dsp`, `qa_cov_iir_fir`, `qa_cov_optimisation`, `qa_cov_rir`, `qa_cov_test_functions`; public recipes `qa-convex-hull`, `qa-delaunay`. All later tasks call `_qa` with one of these variables.

- [ ] **Step 1: Measure per-crate coverage baselines**

Run (takes a while — instrumentation rebuild per crate):

```bash
mkdir -p target
{
  for c in math-analog math-autodiff math-convex-hull math-delaunay math-dsp math-iir-fir math-optimisation math-rir math-test-functions; do
    echo "== $c =="
    cargo llvm-cov -p "$c" --lib --summary-only --release 2>&1 | grep -E '^TOTAL|Total' | tail -1
  done
  echo "== workspace =="
  cargo llvm-cov --lib --summary-only --release 2>&1 | grep -E '^TOTAL|Total' | tail -1
} | tee target/qa-coverage-baselines.md
```

Extract the `lines` percentage from each TOTAL row. Per-crate threshold = that percentage rounded DOWN to a whole number (e.g. 87.4% → `87`). If the workspace TOTAL is below 90%, STOP and report to the user before continuing — the spec says report, not weaken the gate.

- [ ] **Step 2: Replace the QA section in the Justfile**

Replace lines 223–230 (the `# QA` section with `qa: qa-math` and `qa-math:`) with:

````just
# ----------------------------------------------------------------------
# QA
# ----------------------------------------------------------------------

alias qa-math := qa

# Per-crate coverage ratchet thresholds (lines %, measured 2026-08-18).
# Raise these when coverage improves; never lower them.
qa_cov_analog := "<measured>"
qa_cov_autodiff := "<measured>"
qa_cov_convex_hull := "<measured>"
qa_cov_delaunay := "<measured>"
qa_cov_dsp := "<measured>"
qa_cov_iir_fir := "<measured>"
qa_cov_optimisation := "<measured>"
qa_cov_rir := "<measured>"
qa_cov_test_functions := "<measured>"

[private]
_qa crate threshold:
	echo "==================== QA: {{crate}} ===================="
	cargo fmt -p {{crate}} -- --check
	cargo clippy -p {{crate}} --all-targets -- -D warnings
	cargo test -p {{crate}} --lib --release
	cargo test -p {{crate}} --tests --release
	cargo test -p {{crate}} --doc
	cargo bench -p {{crate}} --no-run
	cargo llvm-cov -p {{crate}} --lib --summary-only --release --fail-under-lines {{threshold}}

qa-convex-hull: (_qa "math-convex-hull" qa_cov_convex_hull)

qa-delaunay: (_qa "math-delaunay" qa_cov_delaunay)
````

(Indentation in the `_qa` body must be tabs. Replace each `<measured>` with the threshold from Step 1.)

- [ ] **Step 3: Verify syntax**

Run: `just --list`
Expected: lists `qa-convex-hull`, `qa-delaunay`, alias `qa-math`; no syntax errors.

- [ ] **Step 4: Run the two new recipes**

Run: `just qa-convex-hull && just qa-delaunay`
Expected: both pass all ladder steps (fmt, clippy, tests, bench compile, coverage gate).

- [ ] **Step 5: Commit**

```bash
git add Justfile
git commit -m "qa: add _qa ladder recipe and geometry crate QA (convex-hull, delaunay)"
```

---

### Task 2: qa-rir and qa-test-functions

**Files:**
- Modify: `Justfile` (append after the recipes added in Task 1)

**Interfaces:**
- Consumes: `_qa crate threshold`, `qa_cov_rir`, `qa_cov_test_functions` from Task 1; existing recipe `examples-testfunctions`.
- Produces: public recipes `qa-rir`, `qa-test-functions`.

- [ ] **Step 1: Append the recipes**

````just
qa-rir: (_qa "math-rir" qa_cov_rir)
	cargo bench -p math-rir --bench iso3382 -- --quick

qa-test-functions: (_qa "math-test-functions" qa_cov_test_functions) examples-testfunctions
	cargo run --release -p math-test-functions --example test_additional_functions
	cargo run --release -p math-test-functions --example test_gramacy_lee
	cargo run --release -p math-test-functions --example find_hartman_4d_min
	cargo build --release --bin plot-functions -p math-test-functions --features plotly
	cargo bench -p math-test-functions --bench eval -- --quick
````

(`examples-testfunctions` covers `test_hartman_4d` and `test_new_sfu_functions`; the other three examples are run explicitly. `plot-functions` requires the `plotly` feature and is build-checked only — running it would need chromedriver.)

- [ ] **Step 2: Verify syntax**

Run: `just --list`
Expected: no errors.

- [ ] **Step 3: Run the recipes**

Run: `just qa-rir && just qa-test-functions`
Expected: pass. First run is slow (release builds + property tests at full strength).

- [ ] **Step 4: Commit**

```bash
git add Justfile
git commit -m "qa: add qa-rir and qa-test-functions recipes"
```

---

### Task 3: qa-iir-fir

**Files:**
- Modify: `Justfile` (append)

**Interfaces:**
- Consumes: `_qa`, `qa_cov_iir_fir`, existing recipe `examples-iir`.
- Produces: public recipe `qa-iir-fir`.

- [ ] **Step 1: Append the recipe**

````just
qa-iir-fir: (_qa "math-iir-fir" qa_cov_iir_fir) examples-iir
	cargo build --release --bin filter_bench -p math-iir-fir
	cargo bench -p math-iir-fir --bench biquad_bench -- --quick
	cargo bench -p math-iir-fir --bench response_bench -- --quick
	cargo bench -p math-iir-fir --bench fir_design_bench -- --quick
````

- [ ] **Step 2: Verify syntax**

Run: `just --list`
Expected: no errors.

- [ ] **Step 3: Run the recipe**

Run: `just qa-iir-fir`
Expected: pass. `examples-iir` runs the 5 known-good examples (format_demo, format_rme_room_demo, readme_example, fir_example, peq_loudness_compensation).

- [ ] **Step 4: Commit**

```bash
git add Justfile
git commit -m "qa: add qa-iir-fir recipe"
```

---

### Task 4: qa-optimisation

**Files:**
- Modify: `Justfile` (append)

**Interfaces:**
- Consumes: `_qa`, `qa_cov_optimisation`, existing recipe `examples-optimisation`.
- Produces: public recipe `qa-optimisation`.

- [ ] **Step 1: Append the recipe**

````just
qa-optimisation: (_qa "math-optimisation" qa_cov_optimisation) examples-optimisation
	cargo build --release --bin run-de -p math-optimisation
	cargo build --release --bin benchmark-convergence -p math-optimisation
	cargo build --release --bin plot-de -p math-optimisation --features plotly
	cargo bench -p math-optimisation --bench de_bench -- --quick
	cargo bench -p math-optimisation --bench cmaes_bench -- --quick
````

(`plot-de` requires the `plotly` feature — build check only, matching how `prod` builds it.)

- [ ] **Step 2: Verify syntax**

Run: `just --list`
Expected: no errors.

- [ ] **Step 3: Run the recipe**

Run: `just qa-optimisation`
Expected: pass. Property tests and DE/CMA-ES examples are the slow parts.

- [ ] **Step 4: Commit**

```bash
git add Justfile
git commit -m "qa: add qa-optimisation recipe"
```

---

### Task 5: qa-autodiff

**Files:**
- Modify: `Justfile` (append)

**Interfaces:**
- Consumes: `_qa`, `qa_cov_autodiff`, existing recipe `examples-autodiff`.
- Produces: public recipe `qa-autodiff`.

- [ ] **Step 1: Append the recipe**

````just
qa-autodiff: (_qa "math-autodiff" qa_cov_autodiff) examples-autodiff
	cargo bench -p math-autodiff --bench biquad_bench -- --quick
````

(`examples-autodiff` runs the 6 `*_match` examples: biquad_match, fdn_direct_match, fdn_match, geq_match, peq_match, svf_match.)

- [ ] **Step 2: Verify syntax**

Run: `just --list`
Expected: no errors.

- [ ] **Step 3: Run the recipe**

Run: `just qa-autodiff`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add Justfile
git commit -m "qa: add qa-autodiff recipe"
```

---

### Task 6: qa-dsp

**Files:**
- Modify: `Justfile` (append)

**Interfaces:**
- Consumes: `_qa`, `qa_cov_dsp`.
- Produces: public recipe `qa-dsp`.

- [ ] **Step 1: Append the recipe**

````just
qa-dsp: (_qa "math-dsp" qa_cov_dsp)
	cargo run --release --bin simd-fuzzer -p math-dsp
	cargo build --release --bin wav2csv -p math-dsp
	if [ "$(uname -m)" = "x86_64" ]; then RUSTFLAGS="-C target-feature=+avx2" cargo check -p math-dsp --all-targets && RUSTFLAGS="-C target-feature=+avx2" cargo test -p math-dsp; else echo "Skipping AVX2 pass (not x86_64)"; fi
	cargo bench -p math-dsp --bench welch_spectrum -- --quick
	cargo bench -p math-dsp --bench audio_features -- --quick
````

(The simd-fuzzer runs its default 10000 iterations per function = full strength, random seed printed for reproduction. The AVX2 pass mirrors the CI `avx2-tests` job and is skipped on Apple Silicon.)

- [ ] **Step 2: Verify syntax**

Run: `just --list`
Expected: no errors.

- [ ] **Step 3: Run the recipe**

Run: `just qa-dsp`
Expected: pass; on non-x86_64 the AVX2 step prints the skip note. simd-fuzzer prints per-function failure counts, all zero.

- [ ] **Step 4: Commit**

```bash
git add Justfile
git commit -m "qa: add qa-dsp recipe (simd-fuzzer, AVX2 gate, wav2csv)"
```

---

### Task 7: qa-analog

**Files:**
- Modify: `Justfile` (append)

**Interfaces:**
- Consumes: `_qa`, `qa_cov_analog`.
- Produces: public recipe `qa-analog`.

- [ ] **Step 1: Append the recipe**

````just
qa-analog: (_qa "math-analog" qa_cov_analog)
	cargo run --release -p math-analog --example analysis_report
	cargo run --release -p math-analog --example alias_reference_report
	cargo run --release -p math-analog --example console_preamp_report
	cargo run --release -p math-analog --example model_matrix_report
	cargo run --release -p math-analog --example fitting_report --features fitting
	cargo bench -p math-analog --bench harmonics -- --quick
````

(The report examples print deterministic evidence reports to stdout — same generators used for the checked-in `crates/math-analog/reports/*.md` artifacts. They are smoke-run, not diffed against the curated files. `fitting_report` needs the `fitting` feature, which pulls in math-optimisation.)

- [ ] **Step 2: Verify syntax**

Run: `just --list`
Expected: no errors.

- [ ] **Step 3: Run the recipe**

Run: `just qa-analog`
Expected: pass; each example prints a report and exits 0.

- [ ] **Step 4: Commit**

```bash
git add Justfile
git commit -m "qa: add qa-analog recipe (report examples, harmonics bench smoke)"
```

---

### Task 8: qa aggregator

**Files:**
- Modify: `Justfile` (append the `qa` recipe at the end of the QA section)

**Interfaces:**
- Consumes: all nine `qa-*` recipes from Tasks 1–7.
- Produces: public recipe `qa`; the existing alias `qa-math := qa` (Task 1) now points at it.

- [ ] **Step 1: Append the recipe**

````just
qa: qa-analog qa-autodiff qa-convex-hull qa-delaunay qa-dsp qa-iir-fir qa-optimisation qa-rir qa-test-functions
	cargo clippy --all --features plotly -- -D warnings
	cargo llvm-cov --lib --summary-only --release --fail-under-lines 90
````

(Just runs dependencies in listed order and stops at the first failing crate — fail-fast, matching CI semantics.)

- [ ] **Step 2: Verify syntax and alias**

Run: `just --list && just --summary | tr ' ' '\n' | grep -c '^qa'`
Expected: no errors; `qa`, `qa-math`, and all nine `qa-*` recipes are present.

- [ ] **Step 3: Smoke the aggregator wiring without a full run**

Run: `just --dry-run qa 2>/dev/null | head -5 || just -n qa | head -5`
Expected: shows the dependency order without executing.

- [ ] **Step 4: Commit**

```bash
git add Justfile
git commit -m "qa: add qa aggregator with workspace clippy and 90% coverage gates"
```

---

### Task 9: CI wiring + AGENTS.md + full verification run

**Files:**
- Modify: `.github/workflows/ci.yml:94-95`
- Modify: `AGENTS.md` ("Build and Test Commands" section, QA block)

**Interfaces:**
- Consumes: `qa` recipe from Task 8.
- Produces: CI runs the same QA as local; documented commands.

- [ ] **Step 1: Update the CI qa job**

In `.github/workflows/ci.yml`, change:

```yaml
      - name: Run RoomEQ CI QA
        run: just qa-math
```

to:

```yaml
      - name: Run workspace QA
        run: just qa
```

Also rename the job key/name for clarity — `qa-math:` → `qa:` and `name: Math Audio QA` → `name: Workspace QA`.

Note for the implementer: `just qa` now includes coverage instrumentation runs; if the CI job later times out, that is a follow-up tuning issue, not part of this plan.

- [ ] **Step 2: Validate the workflow file**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` (or `ruby -ryaml -e 'YAML.load_file(".github/workflows/ci.yml")'`)
Expected: parses without error.

- [ ] **Step 3: Update AGENTS.md**

In root `AGENTS.md`, replace:

```markdown
# QA suites
just qa               # Run math QA suite
just qa-math          # math-dsp fuzzer
```

with:

```markdown
# QA suites
just qa               # Full workspace QA: per-crate ladder (fmt, clippy, tests,
                      # coverage ratchet) + crate extras + workspace CI gates
just qa-{crate}       # QA for one crate, e.g. just qa-dsp, just qa-analog,
                      # qa-autodiff, qa-convex-hull, qa-delaunay, qa-iir-fir,
                      # qa-optimisation, qa-rir, qa-test-functions
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml AGENTS.md
git commit -m "qa: CI runs just qa; document qa and qa-{crate} in AGENTS.md"
```

- [ ] **Step 5: Full verification run**

Run (long — instrumentation rebuilds and full-strength property tests; use a background task with a generous timeout, e.g. 2 hours):

```bash
just qa
```

Expected: every crate passes the ladder and its extras; the final workspace clippy and 90% coverage gates pass. If a per-crate step fails on a pre-existing issue (not caused by this plan's changes), report it to the user rather than weakening a gate.

---

## Self-Review Notes

- Spec coverage: ladder (Task 1), all nine per-crate recipes with their extras (Tasks 1–7), aggregator + workspace gates + `qa-math` alias (Tasks 1, 8), CI update and AGENTS.md docs (Task 9), coverage ratchet policy (Task 1), AVX2 arch gate (Task 6), fail-fast behavior (Task 8 note). All spec sections map to a task.
- Placeholders: `<measured>` in Task 1 Step 2 is filled from Step 1's recorded table — the only data-dependent value, by design of the ratchet policy.
- Type consistency: recipe/variable names (`_qa`, `qa_cov_*`, `qa-*`) are identical across tasks; dependency-call syntax `(recipe arg1 arg2)` is used consistently.
