// Rust guideline compliant 2026-02-21
fn main() {
    // Build scripts run on the host, so use Cargo's target OS for link directives.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS")
        .expect("Cargo must set CARGO_CFG_TARGET_OS for build scripts");
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
