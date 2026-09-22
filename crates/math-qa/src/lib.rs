//! Wolfram Engine cross-validation harness for math-audio.
//!
//! Each QA case pairs a closed-form `.wls` oracle (`wolfram/`) with a Rust
//! comparison test (`tests/`). References resolve live when the
//! `WOLFRAMSCRIPT` environment variable points at an activated engine, and
//! otherwise fall back to the checked-in goldens (`wolfram/goldens/`).
//! Regenerate goldens with `just qa-goldens` (needs the engine).
//!
//! Conventions (mirroring `sonium-qa`):
//! - Every `.wls` script prints exactly one compact RawJSON payload as its
//!   last stdout line; the harness parses that line.
//! - Comparisons use [`rel_error`] (sonium's zero-reference handling) and
//!   case-local tolerances recorded in `validation-manifest.toml`.
//! - Passing comparison tests print one `QA_RESULT:` JSON line.

use std::env;
use std::path::PathBuf;
use std::process::Command;

use serde::Serialize;

/// Relative error `|a − b| / |b|` with deterministic zero handling: 0 when
/// both are zero (or both non-finite-equal), ∞ when only the reference is
/// zero, NaN when either side is NaN.
pub fn rel_error(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return 0.0;
    }
    if b == 0.0 {
        return f64::INFINITY;
    }
    ((a - b) / b).abs()
}

/// Relative error of two complex values: `|a − b| / |b|`.
///
/// This is the norm of the difference over the reference magnitude — not
/// `rel_error` of the two norms, which collapses to ~1 whenever the
/// values agree. Zero handling mirrors [`rel_error`]: 0 when `a == b`,
/// ∞ when only the reference is zero, NaN when either side is NaN.
pub fn complex_rel_error(a: num_complex::Complex64, b: num_complex::Complex64) -> f64 {
    let num = (a - b).norm();
    let denom = b.norm();
    if num.is_nan() || denom.is_nan() {
        return f64::NAN;
    }
    if num == 0.0 {
        return 0.0;
    }
    if denom == 0.0 {
        return f64::INFINITY;
    }
    num / denom
}

/// Assert `rel_error(actual, expected) <= tol` with a diagnostic message.
pub fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = rel_error(actual, expected);
    assert!(
        err <= tol,
        "{what}: actual={actual:.12e} expected={expected:.12e} rel_err={err:.3e} tol={tol:.1e}"
    );
}

/// Assert `|actual − expected| <= tol` (for near-zero quantities).
pub fn assert_close_abs(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= tol,
        "{what}: actual={actual:.12e} expected={expected:.12e} abs_err={err:.3e} tol={tol:.1e}"
    );
}

/// Directory holding the checked-in golden JSON files.
///
/// Overridable with `MATH_QA_GOLDEN_DIR` (used by regeneration checks);
/// defaults to `wolfram/goldens` next to this crate's manifest.
pub fn golden_dir() -> PathBuf {
    if let Some(dir) = env::var_os("MATH_QA_GOLDEN_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/goldens")
}

/// Absolute path of a `.wls` oracle script.
pub fn wolfram_script(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram")
        .join(name)
}

/// Reference payload for a case: live engine output when `WOLFRAMSCRIPT`
/// is set, otherwise the checked-in golden. Returns `None` (loudly) when
/// neither is available so tests skip instead of failing blind.
pub fn reference(case: &str, script: &str) -> Option<serde_json::Value> {
    match env::var_os("WOLFRAMSCRIPT") {
        Some(executable) => run_wolfram_script_with(&executable, script),
        None => reference_with(case, &golden_dir()),
    }
}

/// [`reference`] against an explicit golden directory (no env lookup), so
/// the golden/skip branches are unit-testable without touching process env.
pub fn reference_with(case: &str, golden_dir: &std::path::Path) -> Option<serde_json::Value> {
    let path = golden_dir.join(format!("{case}.json"));
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(
            serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("golden {} is not JSON: {error}", path.display())),
        ),
        Err(_) => {
            println!(
                "SKIPPED {case}: no golden at {} and WOLFRAMSCRIPT unset (run `just qa-goldens`)",
                path.display()
            );
            None
        }
    }
}

/// Run a `.wls` script through the live engine; `None` when the engine
/// binary is missing. Follows the `sonium-qa` invocation pattern,
/// including the macOS `WolframKernel` fallback. Takes the engine binary
/// explicitly (no env lookup) so the launch/parse branches are
/// unit-testable with a fixture script; [`reference`] wires in
/// `WOLFRAMSCRIPT`.
pub fn run_wolfram_script_with(
    executable: &std::ffi::OsStr,
    script: &str,
) -> Option<serde_json::Value> {
    let path = wolfram_script(script);
    let mut command = Command::new(executable);
    command.arg("-file").arg(&path);
    #[cfg(target_os = "macos")]
    if env::var_os("WolframKernel").is_none() {
        let kernel = PathBuf::from(
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        );
        if kernel.is_file() {
            command.env("WolframKernel", kernel);
        }
    }
    let output = match command.output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("WOLFRAMSCRIPT binary not found; skipping live reference");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram script {script} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram script {script} output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

/// Structured pass record printed by every comparison test.
#[derive(Debug, Clone, Serialize)]
pub struct QaResult {
    /// Manifest case id (e.g. `math-qa.biquad-response.v1`).
    pub case: String,
    /// True when the comparison passed.
    pub pass: bool,
    /// Worst relative error observed.
    pub max_rel_error: f64,
    /// Case tolerance from the manifest.
    pub tolerance: f64,
    /// `live-engine` or `checked-in-golden`.
    pub provenance: String,
}

/// Emit one `QA_RESULT:` JSON line.
pub fn emit_result(result: &QaResult) {
    println!(
        "QA_RESULT: {}",
        serde_json::to_string(result).unwrap_or_else(|_| "{}".to_string())
    );
}

/// Whether the reference came from the live engine.
pub fn provenance() -> String {
    provenance_with(env::var_os("WOLFRAMSCRIPT").is_some())
}

/// [`provenance`] over an explicit live flag (no env lookup).
pub fn provenance_with(live: bool) -> String {
    if live {
        "live-engine".to_string()
    } else {
        "checked-in-golden".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rel_error_handles_zeros() {
        assert_eq!(rel_error(0.0, 0.0), 0.0);
        assert_eq!(rel_error(1.0, 1.0), 0.0);
        assert_eq!(rel_error(1.0, 0.0), f64::INFINITY);
        assert!(rel_error(f64::NAN, 1.0).is_nan());
        assert!((rel_error(1.1, 1.0) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn complex_rel_error_is_difference_over_reference() {
        use num_complex::Complex64;
        // Identical values read 0, even at zero (where rel_error of the
        // norms would collapse to 1 once they agree).
        assert_eq!(
            complex_rel_error(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
            0.0
        );
        assert_eq!(
            complex_rel_error(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)),
            0.0
        );
        // |(1+i) − 1| / |1| = |i| = 1.
        assert!(
            (complex_rel_error(Complex64::new(1.0, 1.0), Complex64::new(1.0, 0.0)) - 1.0).abs()
                < 1e-12
        );
        // Zero reference with nonzero actual is infinite; NaN propagates.
        assert_eq!(
            complex_rel_error(Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)),
            f64::INFINITY
        );
        assert!(
            complex_rel_error(Complex64::new(f64::NAN, 0.0), Complex64::new(1.0, 0.0)).is_nan()
        );
    }

    /// Restore an env variable to a previous value.
    ///
    /// # Safety
    ///
    /// Call only when no other thread in this process touches process env
    /// concurrently. Each variable restored here has exactly one test
    /// touching it, and integration tests run in separate processes, so no
    /// concurrent reader exists.
    unsafe fn restore_var(key: &str, previous: Option<std::ffi::OsString>) {
        unsafe {
            match previous {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
        }
    }

    /// Default and `MATH_QA_GOLDEN_DIR`-override resolution in one test:
    /// a single test touching this variable cannot race itself, and no
    /// other test in this binary reads it (integration tests run in
    /// separate processes), which keeps Edition-2024 `unsafe` env mutation
    /// sound. Routing every restore through [`restore_golden_dir`] with
    /// both `Some` and `None` covers both arms deterministically.
    #[test]
    fn golden_dir_default_and_override() {
        let ambient = env::var_os("MATH_QA_GOLDEN_DIR");
        unsafe {
            env::remove_var("MATH_QA_GOLDEN_DIR");
        }
        assert_eq!(
            golden_dir(),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/goldens")
        );
        unsafe {
            env::set_var("MATH_QA_GOLDEN_DIR", "/tmp/math-qa-override");
        }
        assert_eq!(golden_dir(), PathBuf::from("/tmp/math-qa-override"));
        unsafe {
            restore_var("MATH_QA_GOLDEN_DIR", Some("/tmp/math-qa-seed".into()));
            assert_eq!(golden_dir(), PathBuf::from("/tmp/math-qa-seed"));
            restore_var("MATH_QA_GOLDEN_DIR", None);
            assert_eq!(
                golden_dir(),
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/goldens")
            );
            restore_var("MATH_QA_GOLDEN_DIR", ambient);
        }
    }

    #[test]
    #[should_panic(expected = "rel_err")]
    fn assert_close_panics_with_diagnostic() {
        assert_close(1.0, 2.0, 1e-9, "mismatch");
    }

    #[test]
    #[should_panic(expected = "abs_err")]
    fn assert_close_abs_panics_with_diagnostic() {
        assert_close_abs(1.0, 2.0, 1e-9, "mismatch");
    }

    /// Live-engine dispatch through [`reference`] with a fixture engine:
    /// the only test touching `WOLFRAMSCRIPT` (and `WolframKernel`), so the
    /// set/restore sequence cannot race another reader — the pure
    /// `provenance_with` test reads no env. `WolframKernel` is cleared so
    /// the macOS fallback block executes deterministically.
    #[cfg(unix)]
    #[test]
    fn reference_takes_live_path_with_fixture_engine() {
        let dir = std::env::temp_dir().join("math-qa-reference-live");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let engine = fixture_engine(&dir);
        let ambient_script = env::var_os("WOLFRAMSCRIPT");
        let ambient_kernel = env::var_os("WolframKernel");
        unsafe {
            env::set_var("WOLFRAMSCRIPT", &engine);
            env::remove_var("WolframKernel");
        }
        let value = reference("any-case", "any.wls").expect("live reference parses");
        assert_eq!(value["ok"], true);
        assert_eq!(provenance(), "live-engine");
        unsafe {
            // Both `restore_var` arms are already covered by the golden-dir
            // test, so these ambient restores add no new uncovered lines.
            restore_var("WOLFRAMSCRIPT", ambient_script);
            restore_var("WolframKernel", ambient_kernel);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wolfram_script_points_at_wolfram_dir() {
        assert_eq!(
            wolfram_script("biquad_spots.wls"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/biquad_spots.wls")
        );
    }

    #[test]
    fn reference_with_reads_golden_and_skips_when_missing() {
        let dir = std::env::temp_dir().join("math-qa-reference-with");
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(dir.join("present.json"), r#"{"answer": 42.0}"#).expect("fixture");
        let value = reference_with("present", &dir).expect("golden loads");
        assert_eq!(value["answer"], 42.0);
        assert!(reference_with("absent-case-xyz", &dir).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[should_panic(expected = "is not JSON")]
    fn reference_with_rejects_corrupt_golden() {
        let dir = std::env::temp_dir().join("math-qa-reference-corrupt");
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(dir.join("broken.json"), "not json{").expect("fixture");
        let _ = reference_with("broken", &dir);
    }

    /// Fixture "engine": prints noise lines, then one JSON line, mimicking a
    /// `.wls` oracle run through `wolframscript -file`.
    fn fixture_engine(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("fake-engine.sh");
        std::fs::write(
            &path,
            "#!/bin/sh\necho 'banner noise'\necho '{\"ok\": true, \"v\": 1.5}'\n",
        )
        .expect("fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod fixture");
        }
        path
    }

    #[cfg(unix)]
    #[test]
    fn runner_parses_last_json_line_and_handles_missing_binary() {
        let dir = std::env::temp_dir().join("math-qa-runner");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let engine = fixture_engine(&dir);
        let value = run_wolfram_script_with(engine.as_os_str(), "any.wls").expect("parses");
        assert_eq!(value["ok"], true);
        assert_eq!(value["v"], 1.5);
        // Nonexistent binary -> loud skip, not a panic.
        assert!(
            run_wolfram_script_with(
                std::ffi::OsStr::new("/tmp/math-qa-no-such-engine-xyz"),
                "any.wls"
            )
            .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    #[should_panic(expected = "not JSON")]
    fn runner_panics_on_non_json_output() {
        let dir = std::env::temp_dir().join("math-qa-runner-garbage");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("garbage-engine.sh");
        std::fs::write(&path, "#!/bin/sh\necho 'no json here'\n").expect("fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod fixture");
        }
        let _ = run_wolfram_script_with(path.as_os_str(), "any.wls");
    }

    #[cfg(unix)]
    #[test]
    #[should_panic(expected = "failed to launch")]
    fn runner_panics_on_unlaunchable_binary() {
        // A directory is not executable: surfaces the non-NotFound spawn
        // error arm without touching process env.
        let dir = std::env::temp_dir();
        let _ = run_wolfram_script_with(dir.as_os_str(), "any.wls");
    }

    #[cfg(unix)]
    #[test]
    #[should_panic(expected = "failed")]
    fn runner_panics_on_failing_engine() {
        let dir = std::env::temp_dir().join("math-qa-runner-fail");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("fail-engine.sh");
        std::fs::write(&path, "#!/bin/sh\necho boom >&2\nexit 3\n").expect("fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod fixture");
        }
        let _ = run_wolfram_script_with(path.as_os_str(), "any.wls");
    }

    #[test]
    fn provenance_labels_both_modes() {
        // Pure: reads no env, so it cannot race the live-path test's
        // `WOLFRAMSCRIPT` mutation. The env-backed `provenance()` arms are
        // covered by the live-path test and the golden-mode comparisons.
        assert_eq!(provenance_with(true), "live-engine");
        assert_eq!(provenance_with(false), "checked-in-golden");
    }
}
