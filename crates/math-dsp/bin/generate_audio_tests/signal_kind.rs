use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(super) enum SignalKind {
    Id,
    Thd1k,
    Thd100,
    ImdSmpte,
    ImdCcif,
    Sweep,
    WhiteNoise,
    PinkNoise,
    MNoise,
    Stipa,
    FullSti,
}

impl SignalKind {
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::Id => "id",
            Self::Thd1k => "thd1k",
            Self::Thd100 => "thd100",
            Self::ImdSmpte => "imd_smpte",
            Self::ImdCcif => "imd_ccif",
            Self::Sweep => "sweep",
            Self::WhiteNoise => "white_noise",
            Self::PinkNoise => "pink_noise",
            Self::MNoise => "m_noise",
            Self::Stipa => "stipa",
            Self::FullSti => "full_sti",
        }
    }

    pub(super) fn all() -> Vec<Self> {
        // FullSti stays opt-in: 98 segments per file would dwarf a default
        // run (at 10 s per segment, one mono file already holds 980 s).
        vec![
            Self::Id,
            Self::Thd1k,
            Self::Thd100,
            Self::ImdSmpte,
            Self::ImdCcif,
            Self::Sweep,
            Self::WhiteNoise,
            Self::PinkNoise,
            Self::MNoise,
            Self::Stipa,
        ]
    }
}
