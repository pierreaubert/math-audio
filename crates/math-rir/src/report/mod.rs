//! Report primitives for the RoomEQ report (see `req-math-audio-report.md`).
//!
//! - M1: [`reflection_table`] — band-limited early-reflection picking.
//! - M3: [`t60_batch`] — batched octave-band T60 over 63 Hz–16 kHz.

pub mod reflection_table;
pub mod t60_batch;

pub use reflection_table::{
    DIRECT_EXCLUSION_MS, DIRECT_SEARCH_MS, ENVELOPE_PROMINENCE_DB, ENVELOPE_SMOOTH_MS,
    EarlyReflection, PROMINENCE_VALLEY_MS, REFLECTION_BAND_HI_HZ, REFLECTION_BAND_LO_HZ,
    REFLECTION_FILTER_ORDER, REPORT_SOUND_SPEED_M_S, ReflectionTable, ReflectionTableConfig,
    early_reflection_table, early_reflection_table_with_workspace, reflection_bandpass,
};
pub use t60_batch::{
    DEFAULT_MIN_R2, MIN_T60_BANDWIDTH_PRODUCT, OCTAVE_FRACTIONAL_BW, OctaveT60, T60_FILTER_ORDER,
    T60_OCTAVE_CENTERS_HZ, T60BatchConfig, T60FitRange, analyze_t60_octaves,
};
