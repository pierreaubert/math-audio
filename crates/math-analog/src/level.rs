/// Proposed analog calibration: 0 VU corresponds to -18 dBFS.
pub const DEFAULT_REFERENCE_LEVEL_DBFS: f32 = -18.0;

/// Convert decibels to a linear gain.
#[inline]
pub fn db_to_gain(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert a positive linear gain to decibels.
#[inline]
pub fn gain_to_db(gain: f32) -> Option<f32> {
    if gain.is_finite() && gain > 0.0 {
        Some(20.0 * gain.log10())
    } else {
        None
    }
}

/// Convert a VU reading to dBFS using the crate's explicit calibration.
#[inline]
pub fn vu_to_dbfs(vu: f32) -> f32 {
    DEFAULT_REFERENCE_LEVEL_DBFS + vu
}

/// Convert dBFS to VU using the crate's explicit calibration.
#[inline]
pub fn dbfs_to_vu(dbfs: f32) -> f32 {
    dbfs - DEFAULT_REFERENCE_LEVEL_DBFS
}

/// Return the linear input gain for a requested drive in dB.
#[inline]
pub fn calibrated_input_gain(drive_db: f32) -> f32 {
    db_to_gain(drive_db)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_conversions_are_explicit() {
        assert_eq!(DEFAULT_REFERENCE_LEVEL_DBFS, -18.0);
        assert!((db_to_gain(0.0) - 1.0).abs() < 1e-6);
        assert!((vu_to_dbfs(0.0) + 18.0).abs() < 1e-6);
        assert!((dbfs_to_vu(-18.0)).abs() < 1e-6);
        assert!((gain_to_db(1.0).unwrap()).abs() < 1e-6);
        assert_eq!(gain_to_db(0.0), None);
    }
}
