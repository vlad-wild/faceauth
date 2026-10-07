//! The system settings other front ends may change through `faceauth-admin`.
//!
//! Only the keys in [`KEYS`] are accepted, each with a type and range. Edits
//! keep the rest of `/etc/faceauth/config.toml` (comments included) untouched,
//! and the result must still parse as a [`Config`]. Security switches such as
//! `auth.skip_remote` are deliberately not listed.

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use crate::config::Config;

#[derive(Debug, Clone, Copy)]
pub enum Kind {
    Bool,
    Int {
        min: i64,
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
    },
    /// A V4L2 device path (`/dev/videoN`) or a camera index.
    Device,
    /// One of a fixed set of words.
    Choice(&'static [&'static str]),
}

#[derive(Debug, Clone, Copy)]
pub struct Key {
    pub name: &'static str,
    pub kind: Kind,
}

pub const KEYS: &[Key] = &[
    Key {
        name: "video.device_path",
        kind: Kind::Device,
    },
    Key {
        name: "video.ir_mode",
        kind: Kind::Bool,
    },
    Key {
        name: "video.timeout",
        kind: Kind::Int { min: 2, max: 30 },
    },
    Key {
        name: "recognition.distance_threshold",
        kind: Kind::Float { min: 0.2, max: 1.2 },
    },
    Key {
        name: "recognition.required_matches",
        kind: Kind::Int { min: 1, max: 10 },
    },
    Key {
        name: "liveness.ir_check",
        kind: Kind::Bool,
    },
    Key {
        name: "auth.skip_lid_closed",
        kind: Kind::Bool,
    },
    Key {
        name: "openvino.device",
        kind: Kind::Choice(&["AUTO", "NPU", "GPU", "CPU"]),
    },
];

pub fn find_key(name: &str) -> Result<&'static Key> {
    KEYS.iter()
        .find(|k| k.name == name)
        .with_context(|| format!("{name} cannot be changed with faceauth-admin"))
}

/// Current values of every listed key.
pub fn get_all(cfg: &Config) -> Result<Map<String, Value>> {
    let full = serde_json::to_value(cfg)?;
    let mut out = Map::new();
    for key in KEYS {
        let (section, field) = key.name.split_once('.').expect("keys are section.field");
        let value = full
            .get(section)
            .and_then(|s| s.get(field))
            .cloned()
            .unwrap_or(Value::Null);
        out.insert(key.name.to_string(), value);
    }
    Ok(out)
}

/// Parse and validate `raw` for `key` into a TOML value.
fn parse_value(key: &Key, raw: &str) -> Result<toml_edit::Value> {
    let raw = raw.trim();
    Ok(match key.kind {
        Kind::Bool => match raw {
            "true" | "on" | "1" => true.into(),
            "false" | "off" | "0" => false.into(),
            _ => bail!("{}: expected true or false", key.name),
        },
        Kind::Int { min, max } => {
            let v: i64 = raw
                .parse()
                .with_context(|| format!("{}: expected an integer", key.name))?;
            if !(min..=max).contains(&v) {
                bail!("{}: {v} is outside {min}..={max}", key.name);
            }
            v.into()
        }
        Kind::Float { min, max } => {
            let v: f64 = raw
                .parse()
                .with_context(|| format!("{}: expected a number", key.name))?;
            if !v.is_finite() || v < min || v > max {
                bail!("{}: {v} is outside {min}..={max}", key.name);
            }
            v.into()
        }
        Kind::Device => {
            let ok = raw.parse::<u32>().is_ok()
                || raw
                    .strip_prefix("/dev/video")
                    .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
            if !ok {
                bail!("{}: expected /dev/videoN or a camera index", key.name);
            }
            raw.into()
        }
        Kind::Choice(options) => {
            let upper = raw.to_ascii_uppercase();
            if !options.contains(&upper.as_str()) {
                bail!("{}: expected one of {}", key.name, options.join(", "));
            }
            upper.into()
        }
    })
}

/// Return `text` with `key` set to `raw`, keeping everything else as it was.
pub fn set_in_toml(text: &str, key_name: &str, raw: &str) -> Result<String> {
    let key = find_key(key_name)?;
    let value = parse_value(key, raw)?;
    let mut doc: toml_edit::DocumentMut = text.parse().context("config.toml is not valid TOML")?;
    let (section, field) = key.name.split_once('.').expect("keys are section.field");
    if !doc.contains_key(section) {
        doc[section] = toml_edit::table();
    }
    let table = doc[section]
        .as_table_like_mut()
        .with_context(|| format!("[{section}] is not a table"))?;
    // Keep a trailing comment on the line, if any.
    let decor = table
        .get(field)
        .and_then(|item| item.as_value())
        .map(|v| v.decor().clone());
    let mut item = toml_edit::Item::Value(value);
    if let (Some(decor), Some(v)) = (decor, item.as_value_mut()) {
        *v.decor_mut() = decor;
    }
    table.insert(field, item);
    let out = doc.to_string();
    toml::from_str::<Config>(&out).context("The edited config would not load")?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "# faceauth config\n[video]\n# camera\ndevice_path = \"/dev/video0\" # rgb\ntimeout = 4\n\n[recognition]\ndistance_threshold = 0.6\n";

    #[test]
    fn edits_keep_comments() {
        let out = set_in_toml(SAMPLE, "recognition.distance_threshold", "0.55").unwrap();
        assert!(out.contains("# faceauth config"));
        assert!(out.contains("# camera"));
        assert!(out.contains("distance_threshold = 0.55"));
        let out = set_in_toml(&out, "video.device_path", "/dev/video2").unwrap();
        assert!(out.contains("device_path = \"/dev/video2\" # rgb"), "{out}");
        let out = set_in_toml(&out, "auth.skip_lid_closed", "off").unwrap();
        assert!(out.contains("[auth]") && out.contains("skip_lid_closed = false"));
        let out = set_in_toml(&out, "openvino.device", "npu").unwrap();
        assert!(out.contains("device = \"NPU\""));
    }

    #[test]
    fn rejects_unknown_keys_and_bad_values() {
        assert!(set_in_toml(SAMPLE, "auth.skip_remote", "false").is_err());
        assert!(set_in_toml(SAMPLE, "recognition.model_path", "/tmp/x.onnx").is_err());
        assert!(set_in_toml(SAMPLE, "recognition.distance_threshold", "5").is_err());
        assert!(set_in_toml(SAMPLE, "recognition.distance_threshold", "NaN").is_err());
        assert!(set_in_toml(SAMPLE, "video.timeout", "1").is_err());
        assert!(set_in_toml(SAMPLE, "video.device_path", "/etc/shadow").is_err());
        assert!(set_in_toml(SAMPLE, "video.ir_mode", "maybe").is_err());
        assert!(set_in_toml(SAMPLE, "openvino.device", "TPU").is_err());
    }

    #[test]
    fn get_all_lists_every_key() {
        let values = get_all(&Config::default()).unwrap();
        assert_eq!(values.len(), KEYS.len());
        assert!(values.values().all(|v| !v.is_null()));
    }
}
