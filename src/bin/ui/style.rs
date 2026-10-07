//! Optional desktop palette for `faceauth-ui`.
//!
//! When a palette file exists (`$FACEAUTH_UI_PALETTE`, or the one written by the
//! nothing-rice desktop at `$XDG_STATE_HOME/nothing-rice/palette.json`), the
//! window follows it: colours from the wallpaper, pill-shaped controls, rounded
//! cards and dot meters. Without it the stock Iced look is kept.

use std::path::PathBuf;
use std::time::SystemTime;

use iced::widget::{button, container, row, text_input};
use iced::{Background, Border, Color, Element, Length, Shadow, Theme};
use serde::Deserialize;

/// Colour tokens shared with the desktop shell (hex strings in the file).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Palette {
    pub dark: bool,
    pub surface: Color,
    pub surface_high: Color,
    pub on: Color,
    pub muted: Color,
    pub accent: Color,
    pub outline: Color,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PaletteFile {
    #[serde(default)]
    mode: Option<String>,
    surface: String,
    surface_high: String,
    on: String,
    muted: String,
    accent: String,
    outline: String,
}

fn parse_hex(hex: &str) -> Option<Color> {
    let h = hex.trim().trim_start_matches('#');
    if h.len() != 6 {
        return None;
    }
    let v = u32::from_str_radix(h, 16).ok()?;
    Some(Color::from_rgb8((v >> 16) as u8, (v >> 8) as u8, v as u8))
}

pub fn palette_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("FACEAUTH_UI_PALETTE") {
        return Some(PathBuf::from(p));
    }
    let state = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/state")))?;
    Some(state.join("nothing-rice/palette.json"))
}

/// The palette file and its modification time (to reload only on change).
pub fn load() -> (Option<Palette>, Option<SystemTime>) {
    let Some(path) = palette_path() else {
        return (None, None);
    };
    let stamp = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    let palette = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str::<PaletteFile>(&s).ok())
        .and_then(|f| {
            Some(Palette {
                dark: f.mode.as_deref() != Some("light"),
                surface: parse_hex(&f.surface)?,
                surface_high: parse_hex(&f.surface_high)?,
                on: parse_hex(&f.on)?,
                muted: parse_hex(&f.muted)?,
                accent: parse_hex(&f.accent)?,
                outline: parse_hex(&f.outline)?,
            })
        });
    (palette, stamp)
}

pub fn modified() -> Option<SystemTime> {
    std::fs::metadata(palette_path()?)
        .and_then(|m| m.modified())
        .ok()
}

/// The Iced theme for a palette (stock dark theme without one).
pub fn theme(p: Option<&Palette>) -> Theme {
    let Some(p) = p else {
        return Theme::Dark;
    };
    let mut base = if p.dark {
        iced::theme::Palette::DARK
    } else {
        iced::theme::Palette::LIGHT
    };
    base.background = p.surface;
    base.text = p.on;
    base.primary = p.on;
    base.danger = p.accent;
    Theme::custom("nothing-rice".to_string(), base)
}

const PILL: f32 = 999.0;

fn pill_border(color: Color) -> Border {
    Border {
        color,
        width: 0.0,
        radius: PILL.into(),
    }
}

/// Pill button: filled `on` for primary actions, `surface_high` otherwise.
pub fn button_style(
    p: Option<Palette>,
    primary: bool,
) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |theme, status| {
        let Some(p) = p else {
            return if primary {
                button::primary(theme, status)
            } else {
                button::secondary(theme, status)
            };
        };
        let (bg, fg) = if primary {
            (p.on, p.surface)
        } else {
            (p.surface_high, p.on)
        };
        let bg = match status {
            button::Status::Hovered => mix(bg, p.outline, 0.25),
            button::Status::Pressed => mix(bg, p.outline, 0.45),
            button::Status::Disabled => mix(bg, p.surface, 0.6),
            button::Status::Active => bg,
        };
        let fg = if status == button::Status::Disabled {
            p.muted
        } else {
            fg
        };
        button::Style {
            background: Some(Background::Color(bg)),
            text_color: fg,
            border: pill_border(bg),
            shadow: Shadow::default(),
            snap: false,
        }
    }
}

/// Segmented tab: the selected tab is filled.
pub fn tab_style(
    p: Option<Palette>,
    selected: bool,
) -> impl Fn(&Theme, button::Status) -> button::Style {
    move |theme, status| {
        if p.is_none() {
            return if selected {
                button::primary(theme, status)
            } else {
                button::secondary(theme, status)
            };
        }
        button_style(p, selected)(theme, status)
    }
}

/// Rounded card on `surface_high`.
pub fn card(p: Option<Palette>) -> impl Fn(&Theme) -> container::Style {
    move |_| match p {
        Some(p) => container::Style {
            background: Some(Background::Color(p.surface_high)),
            border: Border {
                color: p.surface_high,
                width: 0.0,
                radius: 28.0.into(),
            },
            text_color: Some(p.on),
            ..container::Style::default()
        },
        None => container::Style::default(),
    }
}

/// Camera preview frame; outlined in the accent colour while the face matches.
pub fn preview_frame(p: Option<Palette>, matched: bool) -> impl Fn(&Theme) -> container::Style {
    move |_| match p {
        Some(p) => container::Style {
            background: Some(Background::Color(p.surface_high)),
            border: Border {
                color: if matched { p.accent } else { p.surface_high },
                width: if matched { 3.0 } else { 0.0 },
                radius: 32.0.into(),
            },
            ..container::Style::default()
        },
        None => container::Style::default(),
    }
}

/// Pill text field: `surface` with an outline, so it shows on cards too.
pub fn input_style(p: Option<Palette>) -> impl Fn(&Theme, text_input::Status) -> text_input::Style {
    move |theme, status| {
        let mut style = text_input::default(theme, status);
        if let Some(p) = p {
            style.background = Background::Color(p.surface);
            style.border = Border {
                color: if matches!(status, text_input::Status::Focused { .. }) {
                    p.on
                } else {
                    p.outline
                },
                width: 1.5,
                radius: PILL.into(),
            };
            style.value = p.on;
            style.placeholder = p.muted;
            style.selection = mix(p.accent, p.surface, 0.5);
        }
        style
    }
}

/// Pill drop-down, same look as the text fields.
pub fn pick_style(
    p: Option<Palette>,
) -> impl Fn(&Theme, iced::widget::pick_list::Status) -> iced::widget::pick_list::Style {
    move |theme, status| {
        let mut style = iced::widget::pick_list::default(theme, status);
        if let Some(p) = p {
            style.background = Background::Color(p.surface);
            style.text_color = p.on;
            style.placeholder_color = p.muted;
            style.handle_color = p.on;
            style.border = Border {
                color: if matches!(status, iced::widget::pick_list::Status::Hovered) {
                    p.on
                } else {
                    p.outline
                },
                width: 1.5,
                radius: PILL.into(),
            };
        }
        style
    }
}

/// Drop-down menu of a [`pick_style`] list.
pub fn menu_style(p: Option<Palette>) -> impl Fn(&Theme) -> iced::overlay::menu::Style {
    move |theme| {
        let mut style = iced::overlay::menu::default(theme);
        if let Some(p) = p {
            style.background = Background::Color(p.surface_high);
            style.border = Border {
                color: p.outline,
                width: 1.0,
                radius: 18.0.into(),
            };
            style.text_color = p.on;
            style.selected_text_color = p.surface;
            style.selected_background = Background::Color(p.on);
        }
        style
    }
}

fn mix(a: Color, b: Color, t: f32) -> Color {
    Color {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

/// One dot of a meter.
fn dot<'a, M: 'a>(color: Color, size: f32) -> Element<'a, M> {
    container(iced::widget::Space::new())
        .width(Length::Fixed(size))
        .height(Length::Fixed(size))
        .style(move |_| container::Style {
            background: Some(Background::Color(color)),
            border: Border {
                color,
                width: 0.0,
                radius: PILL.into(),
            },
            ..container::Style::default()
        })
        .into()
}

/// A row of `total` dots: the first `lit` in `on`, the one at `mark` (if any)
/// in the accent colour, the rest faint.
pub fn dot_meter<'a, M: 'a>(
    p: &Palette,
    total: usize,
    lit: usize,
    mark: Option<usize>,
) -> Element<'a, M> {
    let faint = mix(p.on, p.surface_high, 0.8);
    row((0..total).map(|i| {
        let color = if Some(i) == mark {
            p.accent
        } else if i < lit {
            p.on
        } else {
            faint
        };
        dot(color, 10.0)
    }))
    .spacing(5)
    .into()
}

/// Coloured status dot for doctor checks.
pub fn status_dot<'a, M: 'a>(p: &Palette, kind: faceauth::diagnostics::Status) -> Element<'a, M> {
    use faceauth::diagnostics::Status;
    let color = match kind {
        Status::Ok => p.on,
        Status::Warn => p.muted,
        Status::Fail => p.accent,
    };
    dot(color, 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_colors() {
        assert_eq!(parse_hex("#ffffff"), Some(Color::from_rgb8(255, 255, 255)));
        assert_eq!(
            parse_hex("D71921"),
            Some(Color::from_rgb8(0xd7, 0x19, 0x21))
        );
        assert_eq!(parse_hex("#fff"), None);
        assert_eq!(parse_hex("zzzzzz"), None);
    }
}
