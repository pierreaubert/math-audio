#[cfg(not(test))]
pub(super) const MAX_GATING_BLOCKS: usize = 36_000;

pub(super) const TRUE_PEAK_FIR_REFERENCE_SAMPLE_RATE: u32 = 48_000;

/// 4x oversampling FIR for true peak detection.
/// 48-tap polyphase filter from BS.1770-5 Annex 2, printed pages 18–19.
/// Published columns are reversed to match oldest-to-newest history storage.
/// https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
///
/// The table is specified for 48 kHz input. `EbuR128::new` logs a warning when
/// true-peak mode is requested at another sample rate rather than failing, since
/// loudness-only and ReplayGain callers commonly analyze native-rate content.
pub(super) const TRUE_PEAK_FIR_PHASES: [[f64; 12]; 4] = [
    [
        -0.0083007812500,
        0.0148925781250,
        -0.0266113281250,
        0.0476074218750,
        -0.1022949218750,
        0.9721679687500,
        0.1373291015625,
        -0.0594482421875,
        0.0332031250000,
        -0.0196533203125,
        0.0109863281250,
        0.0017089843750,
    ],
    [
        -0.0189208984375,
        0.0330810546875,
        -0.0582275390625,
        0.1015625000000,
        -0.2003173828125,
        0.7797851562500,
        0.4650878906250,
        -0.1665039062500,
        0.0891113281250,
        -0.0517578125000,
        0.0292968750000,
        -0.0291748046875,
    ],
    [
        -0.0291748046875,
        0.0292968750000,
        -0.0517578125000,
        0.0891113281250,
        -0.1665039062500,
        0.4650878906250,
        0.7797851562500,
        -0.2003173828125,
        0.1015625000000,
        -0.0582275390625,
        0.0330810546875,
        -0.0189208984375,
    ],
    [
        0.0017089843750,
        0.0109863281250,
        -0.0196533203125,
        0.0332031250000,
        -0.0594482421875,
        0.1373291015625,
        0.9721679687500,
        -0.1022949218750,
        0.0476074218750,
        -0.0266113281250,
        0.0148925781250,
        -0.0083007812500,
    ],
];

pub(super) const TRUE_PEAK_FIR_LEN: usize = 12;
