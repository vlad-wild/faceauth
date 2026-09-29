//! Per-frame verdicts (kept free of OpenCV types so the GUI can use them anywhere).

/// Why a frame was (not) usable. Drives auth logs, CLI hints and the GUI overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FaceVerdict {
    Ok,
    NoFace,
    TooDark,
    LowConfidence,
    TooSmall,
    TooLarge,
    Blurry,
    LookStraight,
    NotLive,
}

impl FaceVerdict {
    /// Stable identifier (i18n key suffix, log field).
    pub fn key(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::NoFace => "no_face",
            Self::TooDark => "too_dark",
            Self::LowConfidence => "low_confidence",
            Self::TooSmall => "too_small",
            Self::TooLarge => "too_large",
            Self::Blurry => "blurry",
            Self::LookStraight => "look_straight",
            Self::NotLive => "not_live",
        }
    }

    /// i18n key of the user-facing message (see `crate::i18n`).
    pub fn message_key(self) -> &'static str {
        match self {
            Self::Ok => "verdict.ok",
            Self::NoFace => "verdict.no_face",
            Self::TooDark => "verdict.too_dark",
            Self::LowConfidence => "verdict.low_confidence",
            Self::TooSmall => "verdict.too_small",
            Self::TooLarge => "verdict.too_large",
            Self::Blurry => "verdict.blurry",
            Self::LookStraight => "verdict.look_straight",
            Self::NotLive => "verdict.not_live",
        }
    }
}
