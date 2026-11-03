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
use sha2::{Digest, Sha256};

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
/// is set, otherwise the checked-in golden.
///
/// # Panics
/// Panics when the selected reference is missing, unreadable, or invalid.
/// The optional return type is retained for source compatibility; success
/// always returns `Some`. Required comparisons never silently skip.
pub fn reference(case: &str, script: &str) -> Option<serde_json::Value> {
    match env::var_os("WOLFRAMSCRIPT") {
        Some(executable) => run_wolfram_script_with(&executable, script),
        None => reference_with(case, &golden_dir()),
    }
}

/// [`reference`] against an explicit golden directory (no env lookup), so
/// reference branches are unit-testable without touching process env.
///
/// # Panics
/// Panics if the golden is missing, unreadable, or invalid JSON.
pub fn reference_with(case: &str, golden_dir: &std::path::Path) -> Option<serde_json::Value> {
    let path = golden_dir.join(format!("{case}.json"));
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(
            serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("golden {} is not JSON: {error}", path.display())),
        ),
        Err(error) => panic!(
            "required golden {} is unavailable: {error} (run `just qa-goldens`)",
            path.display()
        ),
    }
}

/// Run a `.wls` script through the required live engine.
///
/// # Panics
/// Panics if the executable is unavailable, fails, or emits invalid JSON.
///
/// Follows the `sonium-qa` invocation pattern,
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

/// Hash the paths and values of every numeric tolerance in a reference.
///
/// # Errors
/// Returns an error when an oracle tolerance is not a finite number.
pub fn allowance_digest(reference: &serde_json::Value) -> Result<String, String> {
    fn visit(value: &serde_json::Value, path: &str, digest: &mut Sha256) -> Result<(), String> {
        match value {
            serde_json::Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by_key(|(key, _)| *key);
                for (key, value) in entries {
                    let path = format!("{path}/{key}");
                    if key.contains("tolerance") {
                        let number = value
                            .as_f64()
                            .filter(|n| n.is_finite() && *n >= 0.0)
                            .ok_or_else(|| format!("invalid allowance: {path}"))?;
                        digest.update(path.as_bytes());
                        digest.update([0]);
                        digest.update(number.to_le_bytes());
                    } else {
                        visit(value, &path, digest)?;
                    }
                }
            }
            serde_json::Value::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    visit(value, &format!("{path}/{index}"), digest)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    let mut digest = Sha256::new();
    visit(reference, "", &mut digest)?;
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// Collect raw errors against the manifest's named allowances.
pub struct NormalizedErrors {
    allowances: std::collections::BTreeMap<String, f64>,
    errors: std::collections::BTreeMap<String, f64>,
}

fn normalized_allowances(
    case: &toml::Value,
) -> Result<std::collections::BTreeMap<String, f64>, String> {
    let table = case
        .get("allowances")
        .and_then(toml::Value::as_table)
        .ok_or("missing normalized allowances")?;
    if table.is_empty() {
        return Err("empty normalized allowances".into());
    }
    table
        .iter()
        .map(|(key, value)| {
            let value = value
                .as_float()
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or("invalid normalized allowance")?;
            Ok((key.clone(), value))
        })
        .collect()
}

impl NormalizedErrors {
    /// Create an accumulator using the checked-in manifest.
    pub fn new(case_id: &str) -> Self {
        let manifest: toml::Value =
            toml::from_str(include_str!("../validation-manifest.toml")).expect("valid manifest");
        let case = manifest["case"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["id"].as_str() == Some(case_id))
            .expect("manifest case");
        Self {
            allowances: normalized_allowances(case).expect("normalized allowances"),
            errors: Default::default(),
        }
    }

    /// Record one raw nonnegative error under its canonical allowance ID.
    pub fn record(&mut self, id: &str, error: f64) {
        assert!(self.allowances.contains_key(id), "unknown allowance {id}");
        assert!(error.is_finite() && error >= 0.0, "invalid error for {id}");
        self.errors
            .entry(id.to_owned())
            .and_modify(|v| *v = v.max(error))
            .or_insert(error);
    }

    /// Emit a summary containing raw maxima and their derived ratio.
    pub fn emit(self, case: &str, reference: &serde_json::Value) {
        assert_eq!(
            self.errors.len(),
            self.allowances.len(),
            "missing normalized components"
        );
        let error = self
            .errors
            .iter()
            .map(|(id, raw)| raw / self.allowances[id])
            .fold(0.0_f64, f64::max);
        let result = serde_json::json!({"case": case, "pass": true, "metric": "max_error_over_entry_tolerance", "unit": "ratio", "error": error, "tolerance": 1.0, "allowance_sha256": allowance_digest(reference).expect("numeric oracle allowances"), "provenance": provenance(), "components": self.errors});
        println!("QA_RESULT: {result}");
    }
}

/// Structured pass record printed by every comparison test.
#[derive(Debug, Clone, Serialize)]
pub struct QaResult {
    /// Manifest case id (e.g. `math-qa.biquad-response.v1`).
    pub case: String,
    /// True when the comparison passed.
    pub pass: bool,
    /// Name of the recorded comparison metric.
    pub metric: String,
    /// Unit of the recorded error and tolerance.
    pub unit: String,
    /// Worst error observed for the named metric.
    pub error: f64,
    /// Case tolerance from the manifest.
    pub tolerance: f64,
    /// Digest of the reference's per-entry allowances.
    pub allowance_sha256: String,
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

/// Verify that every manifest case emitted exactly one successful comparison.
///
/// # Errors
/// Rejects malformed records, missing/duplicate/unknown cases, unsuccessful
/// comparisons, invalid numeric diagnostics, and unknown reference provenance.
pub fn verify_result_records(log: &str, manifest: &str) -> Result<usize, String> {
    use std::collections::{BTreeMap, BTreeSet};
    let manifest: toml::Value = toml::from_str(manifest).map_err(|e| format!("manifest: {e}"))?;
    let cases = manifest
        .get("case")
        .and_then(toml::Value::as_array)
        .ok_or("manifest has no cases")?;
    let mut expected = BTreeMap::new();
    for case in cases {
        let id = case
            .get("id")
            .and_then(toml::Value::as_str)
            .ok_or("manifest case has no id")?;
        let metric = case
            .get("metric")
            .and_then(toml::Value::as_str)
            .ok_or("manifest case has no metric")?;
        let unit = case
            .get("unit")
            .and_then(toml::Value::as_str)
            .ok_or("manifest case has no unit")?;
        let limit = case
            .get("limit")
            .and_then(toml::Value::as_float)
            .filter(|v| v.is_finite() && *v >= 0.0)
            .ok_or("manifest case has invalid limit")?;
        let allowance = case
            .get("allowance_sha256")
            .and_then(toml::Value::as_str)
            .filter(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
            .ok_or("manifest case has invalid allowance digest")?;
        if metric.is_empty() || unit.is_empty() {
            return Err("manifest case has empty metric or unit".into());
        }
        if expected
            .insert(
                id,
                (
                    metric,
                    unit,
                    limit,
                    allowance,
                    if metric == "max_error_over_entry_tolerance" {
                        Some(normalized_allowances(case)?)
                    } else {
                        None
                    },
                ),
            )
            .is_some()
        {
            return Err(format!("duplicate manifest case: {id}"));
        }
    }
    if expected.is_empty() {
        return Err("manifest has no cases".into());
    }
    let mut seen = BTreeSet::new();
    for line in log.lines() {
        let Some((_, payload)) = line.split_once("QA_RESULT: ") else {
            continue;
        };
        let record: serde_json::Value =
            serde_json::from_str(payload).map_err(|e| format!("QA_RESULT: {e}"))?;
        let id = record["case"].as_str().ok_or("comparison has no case id")?;
        let (metric, unit, limit, allowance, components) = expected
            .get(id)
            .ok_or_else(|| format!("unknown comparison: {id}"))?;
        if record["metric"].as_str() != Some(*metric) || record["unit"].as_str() != Some(*unit) {
            return Err(format!("{id}: wrong or missing metric/unit"));
        }
        if record["allowance_sha256"].as_str() != Some(*allowance) {
            return Err(format!("{id}: reference allowances differ from manifest"));
        }
        if !seen.insert(id.to_owned()) {
            return Err(format!("duplicate comparison: {id}"));
        }
        if record["pass"].as_bool() != Some(true) {
            return Err(format!("failed comparison: {id}"));
        }
        let mut numeric = [0.0_f64; 2];
        for (slot, key) in numeric.iter_mut().zip(["error", "tolerance"]) {
            let value = record[key]
                .as_f64()
                .ok_or_else(|| format!("{id}: invalid {key}"))?;
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{id}: invalid {key}"));
            }
            *slot = value;
        }
        if let Some(allowances) = components {
            let raw = record["components"]
                .as_object()
                .ok_or_else(|| format!("{id}: missing components"))?;
            if raw.len() != allowances.len() {
                return Err(format!("{id}: wrong component set"));
            }
            let mut ratio = 0.0_f64;
            for (key, allowance) in allowances {
                let value = raw
                    .get(key)
                    .and_then(serde_json::Value::as_f64)
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .ok_or_else(|| format!("{id}: invalid component {key}"))?;
                ratio = ratio.max(value / allowance);
            }
            if ratio != numeric[0] {
                return Err(format!("{id}: normalized error does not match components"));
            }
        }
        if numeric[1] != *limit {
            return Err(format!("{id}: tolerance differs from manifest"));
        }
        // A self-reported pass is not evidence: the recorded error must
        // actually meet the recorded tolerance.
        if numeric[0] > numeric[1] {
            return Err(format!(
                "{id}: error {} exceeds tolerance {}",
                numeric[0], numeric[1]
            ));
        }
        if !matches!(
            record["provenance"].as_str(),
            Some("live-engine" | "checked-in-golden")
        ) {
            return Err(format!("{id}: invalid provenance"));
        }
    }
    let missing: Vec<_> = expected.keys().filter(|id| !seen.contains(**id)).collect();
    if !missing.is_empty() {
        return Err(format!("missing comparisons: {missing:?}"));
    }
    Ok(seen.len())
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
    fn reference_with_reads_golden_and_rejects_missing() {
        let dir = std::env::temp_dir().join("math-qa-reference-with");
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(dir.join("present.json"), r#"{"answer": 42.0}"#).expect("fixture");
        let value = reference_with("present", &dir).expect("golden loads");
        assert_eq!(value["answer"], 42.0);
        assert!(std::panic::catch_unwind(|| reference_with("absent-case-xyz", &dir)).is_err());
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
        // An explicitly selected missing engine must fail the comparison.
        assert!(
            std::panic::catch_unwind(|| run_wolfram_script_with(
                std::ffi::OsStr::new("/tmp/math-qa-no-such-engine-xyz"),
                "any.wls"
            ))
            .is_err()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    #[should_panic(expected = "not JSON")]
    fn runner_panics_on_non_json_output() {
        // `/bin/echo` ignores the `-file <script>` arguments and prints
        // them back, which is non-JSON stdout with a success status:
        // exactly the parse-failure branch under test. A freshly written
        // script fixture is not used because write-then-exec races on
        // overlay filesystems (transient ETXTBSY on CI runners).
        let _ = run_wolfram_script_with(std::ffi::OsStr::new("/bin/echo"), "any.wls");
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
