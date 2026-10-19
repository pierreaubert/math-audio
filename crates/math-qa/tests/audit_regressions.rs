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
    let manifest = "[[case]]\nid = 'one'\n";
    let valid = "QA_RESULT: {\"case\":\"one\",\"pass\":true,\"max_rel_error\":0.0,\"tolerance\":0.01,\"provenance\":\"checked-in-golden\"}\n";
    assert_eq!(math_qa::verify_result_records(valid, manifest).unwrap(), 1);
    for invalid in [
        String::new(),
        valid.repeat(2),
        valid.replace("one", "unknown"),
        valid.replace("true", "false"),
        valid.replace("checked-in-golden", "skipped"),
        valid.replace("0.01", "null"),
        "QA_RESULT: invalid\n".into(),
    ] {
        assert!(math_qa::verify_result_records(&invalid, manifest).is_err());
    }
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
