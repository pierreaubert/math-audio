//! Require executed comparison records for every validation-manifest case.

// Rust guideline compliant 2026-02-21
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: verify-qa-results <test-log>")?;
    let log = std::fs::read_to_string(path)?;
    let count =
        math_qa::verify_result_records(&log, include_str!("../../validation-manifest.toml"))?;
    println!("Verified {count} executed comparisons against the validation manifest");
    Ok(())
}
