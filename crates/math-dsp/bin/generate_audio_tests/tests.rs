use super::consts::AMP_STD;
use super::consts::ID_BASE_FREQ;
use super::consts::ID_MAX_FREQ;
use super::consts::ID_STEP_FREQ;
use super::consts::SMPTE_POWER_RATIO;
use super::consts::generate_one;
use super::consts::id_frequency;
use super::gen_::gen_tone_checked;
use super::misc::build_info_chunk;
use super::signal_kind::SignalKind;
use super::write::write_wav;
use std::fs;

use serde_json::Value;

#[test]
fn id_frequency_uses_shared_step_formula() {
    assert_eq!(id_frequency(0), ID_BASE_FREQ);
    assert_eq!(id_frequency(1), ID_BASE_FREQ + ID_STEP_FREQ);
    assert_eq!(id_frequency(32), ID_MAX_FREQ);
}

#[test]
fn checked_tone_rejects_nyquist_frequency() {
    let err = gen_tone_checked(24_000.0, AMP_STD, 48_000, 0.01).unwrap_err();
    assert!(err.contains("Nyquist violation"), "unexpected error: {err}");
}

#[test]
fn smpte_metadata_uses_power_ratio() {
    let dir = tempfile::tempdir().unwrap();
    let wav = generate_one(dir.path(), SignalKind::ImdSmpte, 1, 48_000, 16, 0.01, 0.0).unwrap();
    let sidecar = fs::read_to_string(wav.with_extension("wav.json")).unwrap();
    let sidecar: Value = serde_json::from_str(&sidecar).unwrap();

    assert_eq!(sidecar["signal"]["type"], "imd_smpte");
    assert_eq!(sidecar["signal"]["power_ratio"], SMPTE_POWER_RATIO);
    assert!(sidecar["signal"].get("ratio").is_none());
}

#[test]
fn write_wav_rejects_unsupported_bit_depth_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.wav");
    let err = write_wav(&path, &[0.0], 48_000, 1, 12, &[]).unwrap_err();
    assert!(
        err.contains("Unsupported bit depth"),
        "unexpected error: {err}"
    );
    assert!(!path.exists());
}

#[test]
fn info_chunk_is_word_aligned() {
    let chunk = build_info_chunk(&[(b"IART", "SotF"), (b"IPRD", "Odd")]);
    assert_eq!(chunk.len() % 2, 0);
    assert!(chunk.starts_with(b"LIST"));
    assert_eq!(&chunk[8..12], b"INFO");
}

#[test]
fn stipa_sidecar_records_bands_and_target_rms() {
    let dir = tempfile::tempdir().unwrap();
    let wav = generate_one(dir.path(), SignalKind::Stipa, 1, 48_000, 16, 0.05, 0.0).unwrap();
    assert!(wav.exists());
    let sidecar = fs::read_to_string(wav.with_extension("wav.json")).unwrap();
    let sidecar: Value = serde_json::from_str(&sidecar).unwrap();

    assert_eq!(sidecar["signal"]["type"], "stipa");
    assert_eq!(sidecar["signal"]["bands"].as_array().unwrap().len(), 7);
    assert_eq!(
        sidecar["signal"]["modulations"].as_array().unwrap().len(),
        7
    );
    let target_rms = sidecar["signal"]["target_rms"].as_f64().unwrap();
    assert!((target_rms - 0.07).abs() < 1e-6);
}

#[test]
fn full_sti_sidecar_records_segment_grid() {
    let dir = tempfile::tempdir().unwrap();
    let wav = generate_one(dir.path(), SignalKind::FullSti, 1, 48_000, 16, 0.05, 0.01).unwrap();
    assert!(wav.exists());
    let sidecar = fs::read_to_string(wav.with_extension("wav.json")).unwrap();
    let sidecar: Value = serde_json::from_str(&sidecar).unwrap();

    assert_eq!(sidecar["signal"]["type"], "full_sti");
    assert_eq!(sidecar["signal"]["segments"], 98);
    assert_eq!(sidecar["signal"]["order"], "fm-major");
    assert_eq!(
        sidecar["signal"]["modulations"].as_array().unwrap().len(),
        14
    );
    // Top-level duration covers the whole 98-segment file.
    let expected = 98.0 * 0.05 + 97.0 * 0.01;
    let duration = sidecar["duration"].as_f64().unwrap();
    assert!((duration - expected).abs() < 1e-4);
}

#[test]
fn sti_signals_skip_sample_rates_below_minimum() {
    let dir = tempfile::tempdir().unwrap();
    for signal in [SignalKind::Stipa, SignalKind::FullSti] {
        let err = generate_one(dir.path(), signal, 1, 16_000, 16, 0.05, 0.0).unwrap_err();
        assert!(err.contains("skipped"), "unexpected error: {err}");
    }
}

#[test]
fn full_sti_is_opt_in_only() {
    let all = SignalKind::all();
    assert!(all.contains(&SignalKind::Stipa));
    assert!(!all.contains(&SignalKind::FullSti));
}
