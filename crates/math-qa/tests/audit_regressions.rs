//! Reference availability must affect the validation result.

// Rust guideline compliant 2026-02-21
#[test]
fn audit_missing_explicit_engine_is_a_failure() {
    let missing = std::env::temp_dir().join(format!("missing-wolfram-{}", std::process::id()));
    assert!(!missing.exists());
    let result = std::panic::catch_unwind(|| {
        math_qa::run_wolfram_script_with(missing.as_os_str(), "schroeder_t60.wls")
    });
    assert!(
        result.is_err(),
        "an explicitly selected engine cannot silently skip"
    );
}

#[test]
fn audit_missing_golden_is_a_failure() {
    let missing = std::env::temp_dir().join(format!("missing-golden-{}", std::process::id()));
    assert!(!missing.exists());
    let result = std::panic::catch_unwind(|| math_qa::reference_with("absent-case", &missing));
    assert!(
        result.is_err(),
        "a required comparison needs an actual reference"
    );
}

#[test]
fn comparison_record_gate_rejects_missing_duplicate_unknown_and_failed_cases() {
    let manifest = "[[case]]\nid = 'one'\nmetric = 'peak_error'\nunit = 'dB'\nlimit = 0.01\nallowance_sha256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'\n";
    let valid = "QA_RESULT: {\"case\":\"one\",\"pass\":true,\"metric\":\"peak_error\",\"unit\":\"dB\",\"error\":0.0,\"tolerance\":0.01,\"allowance_sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\",\"provenance\":\"checked-in-golden\"}\n";
    assert_eq!(math_qa::verify_result_records(valid, manifest).unwrap(), 1);
    assert!(math_qa::verify_result_records(valid, &manifest.replace("0.01", "0.02")).is_err());
    for invalid in [
        String::new(),
        valid.repeat(2),
        valid.replace("one", "unknown"),
        valid.replace("true", "false"),
        valid.replace("checked-in-golden", "skipped"),
        valid.replace("0.01", "null"),
        valid.replace("0.01", "0.02"),
        valid.replace("peak_error", "wrong"),
        valid.replace("dB", "ratio"),
        valid.replace("\"metric\":\"peak_error\",", ""),
        valid.replace("0.0,", "null,"),
        "QA_RESULT: invalid\n".into(),
    ] {
        assert!(math_qa::verify_result_records(&invalid, manifest).is_err());
    }
}

#[test]
fn comparison_record_gate_rejects_error_above_tolerance() {
    let manifest = "[[case]]\nid = 'one'\nmetric = 'peak_error'\nunit = 'dB'\nlimit = 0.01\nallowance_sha256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'\n";
    // Self-reported pass with an error far above its own tolerance must not
    // be certified by the record gate.
    let exceeding = "QA_RESULT: {\"case\":\"one\",\"pass\":true,\"metric\":\"peak_error\",\"unit\":\"dB\",\"error\":999.0,\"tolerance\":0.01,\"allowance_sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\",\"provenance\":\"checked-in-golden\"}\n";
    assert!(math_qa::verify_result_records(exceeding, manifest).is_err());
    let at_tolerance = "QA_RESULT: {\"case\":\"one\",\"pass\":true,\"metric\":\"peak_error\",\"unit\":\"dB\",\"error\":0.01,\"tolerance\":0.01,\"allowance_sha256\":\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\",\"provenance\":\"live-engine\"}\n";
    assert_eq!(
        math_qa::verify_result_records(at_tolerance, manifest).unwrap(),
        1
    );
}

#[test]
fn comparison_record_gate_returns_errors_for_malformed_manifests() {
    for manifest in [
        "",
        "case = 3",
        "case = []",
        "[[case]]\nwrong = 'one'",
        "[[case]]\nid = 'one'\n[[case]]\nid = 'one'",
        "this is not toml",
    ] {
        assert!(math_qa::verify_result_records("", manifest).is_err());
    }
}

#[test]
fn reference_allowance_changes_are_rejected() {
    let reference = serde_json::json!({"entries": [{"tolerance": 0.01}, {"tolerance": 0.02}]});
    let digest = math_qa::allowance_digest(&reference).unwrap();
    let manifest = format!(
        "[[case]]\nid = 'one'\nmetric = 'error'\nunit = 'ratio'\nlimit = 1.0\nallowance_sha256 = '{digest}'\n"
    );
    let record = |value: &serde_json::Value| {
        format!(
            "QA_RESULT: {}",
            serde_json::json!({
                "case": "one", "pass": true, "metric": "error", "unit": "ratio",
                "error": 0.5, "tolerance": 1.0, "allowance_sha256": math_qa::allowance_digest(value).unwrap(),
                "provenance": "checked-in-golden"
            })
        )
    };
    assert!(math_qa::verify_result_records(&record(&reference), &manifest).is_ok());
    let changed = serde_json::json!({"entries": [{"tolerance": 0.02}, {"tolerance": 0.02}]});
    assert!(math_qa::verify_result_records(&record(&changed), &manifest).is_err());
    let missing = serde_json::json!({"entries": [{}, {"tolerance": 0.02}]});
    assert!(math_qa::verify_result_records(&record(&missing), &manifest).is_err());
}
