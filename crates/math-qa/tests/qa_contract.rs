//! Contract: every manifest case has a `.wls` oracle, a checked-in
//! golden, and a comparison test target. Fails loudly while the engine
//! step (`just qa-goldens`) is still pending.

use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn manifest_cases_are_complete() {
    let text = std::fs::read_to_string(manifest_dir().join("validation-manifest.toml")).unwrap();
    let manifest: toml::Value = toml::from_str(&text).unwrap();
    let cases = manifest["case"].as_array().unwrap();
    assert!(!cases.is_empty(), "manifest lists no cases");

    let mut missing_goldens = Vec::new();
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let script = manifest_dir()
            .join("wolfram")
            .join(case["script"].as_str().unwrap());
        assert!(
            script.is_file(),
            "{id}: missing oracle {}",
            script.display()
        );
        let test = manifest_dir()
            .join("tests")
            .join(format!("{}.rs", case["test"].as_str().unwrap()));
        assert!(test.is_file(), "{id}: missing test {}", test.display());
        let golden = manifest_dir()
            .join("wolfram/goldens")
            .join(case["golden"].as_str().unwrap());
        if !golden.is_file() {
            missing_goldens.push(id.to_string());
        }
        // Tolerance must be a non-empty recorded claim.
        assert!(
            !case["tolerance"].as_str().unwrap().is_empty(),
            "{id}: missing tolerance"
        );
    }
    assert!(
        missing_goldens.is_empty(),
        "cases without engine-blessed goldens (run `just qa-goldens`): {missing_goldens:?}"
    );
}
