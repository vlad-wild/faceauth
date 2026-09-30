//! Face model storage.
//!
//! Models live in a root-only store ([`MODELS_DIR`], one `<user>.json` per user).
//! `faceauth-auth` runs as root from PAM, so it only trusts files that the user
//! cannot modify: owned by root, not group/other accessible, not a symlink.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::matching::set_score;
use crate::privilege::lookup_user;

/// Root-only model store.
pub const MODELS_DIR: &str = "/var/lib/faceauth/models";
/// When this file exists `faceauth-auth` exits immediately (PAM falls through to the password).
pub const DISABLED_FLAG: &str = "/etc/faceauth/disabled";
/// Upper bound on vectors accepted from one enrollment / probe payload.
pub const MAX_PAYLOAD_VECTORS: usize = 200;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedEmbeddings {
    pub label: String,
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceModel {
    pub label: String,
    /// Primary embedding set (e.g. default appearance).
    pub embeddings: Vec<Vec<f32>>,
    /// Extra sets (e.g. `glasses`, `hat`) — auth succeeds if any set matches.
    #[serde(default)]
    pub extensions: Vec<NamedEmbeddings>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Fingerprint of the recognition model that produced the embeddings
    /// (`None` for models enrolled before fingerprints existed).
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub embedding_dim: Option<usize>,
}

/// How a new capture is merged into an existing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnrollMerge {
    /// Drop previous model for this user; save only this capture (clears extensions).
    #[default]
    ReplaceAll,
    /// Append new vectors to primary `embeddings`; keep extensions and primary label.
    AppendPrimary,
    /// Replace or create named extension (`--variant`); primary unchanged unless new user.
    ReplaceVariant,
    /// Append into named extension (`--variant` + `--append`).
    AppendVariant,
}

impl EnrollMerge {
    pub fn from_flags(variant: bool, append: bool) -> Self {
        match (variant, append) {
            (true, true) => Self::AppendVariant,
            (true, false) => Self::ReplaceVariant,
            (false, true) => Self::AppendPrimary,
            (false, false) => Self::ReplaceAll,
        }
    }

    pub fn is_variant(self) -> bool {
        matches!(self, Self::ReplaceVariant | Self::AppendVariant)
    }
}

impl FaceModel {
    pub fn new(label: String, embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            label,
            embeddings,
            extensions: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: None,
            model_id: None,
            embedding_dim: None,
        }
    }

    fn sets(&self) -> impl Iterator<Item = &[Vec<f32>]> {
        std::iter::once(self.embeddings.as_slice())
            .chain(self.extensions.iter().map(|e| e.embeddings.as_slice()))
    }

    /// Top-k score: per set, the mean of the `k` smallest distances; then the best set.
    pub fn match_score(&self, probe: &[f32], k: usize) -> f32 {
        self.sets()
            .map(|s| set_score(s, probe, k))
            .fold(f32::INFINITY, f32::min)
    }

    /// Nearest-neighbour distance over all sets (diagnostics).
    pub fn best_match_distance(&self, probe: &[f32]) -> f32 {
        self.match_score(probe, 1)
    }

    pub fn sample_count(&self) -> usize {
        self.sets().map(<[_]>::len).sum()
    }

    /// Refuse to compare against embeddings from a different recognizer.
    pub fn check_compatible(&self, model_id: &str, dim: usize) -> Result<()> {
        match &self.model_id {
            Some(id) if id != model_id => bail!(
                "Stored face model was enrolled with a different recognition model \
                 ({id} ≠ {model_id}); enroll again with `sudo faceauth add`"
            ),
            None => log::warn!(
                "Face model has no recognizer fingerprint (legacy/migrated); \
                 re-enroll to enable the check"
            ),
            _ => {}
        }
        if let Some(bad) = self.sets().flatten().find(|v| v.len() != dim) {
            bail!(
                "Stored embeddings have {} dimensions but the recognizer produces {dim}; enroll again",
                bad.len()
            );
        }
        Ok(())
    }

    /// Upsert extension `label`. If `append`, merge vectors into existing; else replace.
    pub fn upsert_extension(&mut self, label: String, vectors: Vec<Vec<f32>>, append: bool) {
        let label_trim = label.trim().to_string();
        if label_trim.is_empty() {
            return;
        }
        if let Some(ext) = self.extensions.iter_mut().find(|e| e.label == label_trim) {
            if append {
                ext.embeddings.extend(vectors);
            } else {
                ext.embeddings = vectors;
            }
            return;
        }
        self.extensions.push(NamedEmbeddings {
            label: label_trim,
            embeddings: vectors,
        });
    }

    pub fn remove_variant(&mut self, label: &str) -> Result<()> {
        let before = self.extensions.len();
        self.extensions.retain(|e| e.label != label);
        if self.extensions.len() == before {
            bail!("No variant named {label:?}");
        }
        Ok(())
    }

    pub fn rename_variant(&mut self, from: &str, to: &str) -> Result<()> {
        let to = to.trim();
        if to.is_empty() {
            bail!("New variant name is empty");
        }
        if self.extensions.iter().any(|e| e.label == to) {
            bail!("Variant {to:?} already exists");
        }
        let ext = self
            .extensions
            .iter_mut()
            .find(|e| e.label == from)
            .with_context(|| format!("No variant named {from:?}"))?;
        ext.label = to.to_string();
        Ok(())
    }
}

/// Metadata about a model without the vectors (`faceauth list --json`, GUI).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelSummary {
    pub user: String,
    pub label: String,
    pub primary_samples: usize,
    pub variants: Vec<VariantSummary>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub model_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VariantSummary {
    pub label: String,
    pub samples: usize,
}

impl FaceModel {
    pub fn summary(&self, user: &str) -> ModelSummary {
        ModelSummary {
            user: user.to_string(),
            label: self.label.clone(),
            primary_samples: self.embeddings.len(),
            variants: self
                .extensions
                .iter()
                .map(|e| VariantSummary {
                    label: e.label.clone(),
                    samples: e.embeddings.len(),
                })
                .collect(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            model_id: self.model_id.clone(),
        }
    }
}

/// Enrollment sent by the GUI to `faceauth import` (stdin, JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportPayload {
    pub merge: EnrollMerge,
    #[serde(default)]
    pub variant: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    pub model_id: String,
    pub embeddings: Vec<Vec<f32>>,
}

/// Probe vectors sent by the GUI to `faceauth verify` (stdin, JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyPayload {
    pub model_id: String,
    pub embeddings: Vec<Vec<f32>>,
}

/// Answer of `faceauth verify` (stdout, JSON): scores per probe vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyResult {
    pub scores: Vec<f32>,
    pub threshold: f32,
    pub required_matches: u32,
}

/// A capture to merge into a user's model.
#[derive(Debug, Clone)]
pub struct Enrollment {
    pub merge: EnrollMerge,
    pub variant: Option<String>,
    pub label: Option<String>,
    pub vectors: Vec<Vec<f32>>,
    pub model_id: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Database {
    pub users: HashMap<String, FaceModel>,
}

impl Database {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_json(content: &str) -> Result<Self> {
        Ok(serde_json::from_str(content)?)
    }

    pub fn get_user(&self, username: &str) -> Option<&FaceModel> {
        self.users.get(username)
    }

    pub fn get_user_mut(&mut self, username: &str) -> Option<&mut FaceModel> {
        self.users.get_mut(username)
    }

    pub fn remove_user(&mut self, username: &str) -> Option<FaceModel> {
        self.users.remove(username)
    }

    /// Merge a capture into `username`'s model (shared by CLI, GUI import and tests).
    pub fn apply_enrollment(&mut self, username: &str, e: Enrollment) -> Result<()> {
        let Enrollment {
            merge,
            variant,
            label,
            vectors,
            model_id,
        } = e;
        if vectors.is_empty() {
            bail!("No face samples to save");
        }
        let dim = vectors[0].len();
        let variant = variant
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        if merge.is_variant() && variant.is_none() {
            bail!("Variant name is empty");
        }
        if merge == EnrollMerge::ReplaceAll || !self.users.contains_key(username) {
            let label = label.unwrap_or_else(|| format!("{username}-default"));
            self.users
                .insert(username.to_string(), FaceModel::new(label, Vec::new()));
        }
        let model = self.users.get_mut(username).expect("inserted above");
        if let Some(id) = &model.model_id
            && *id != model_id
        {
            bail!(
                "Existing model was enrolled with a different recognition model; \
                 replace it instead of appending"
            );
        }
        if model.sets().flatten().any(|v| v.len() != dim) {
            bail!("New samples have a different dimension than the stored model");
        }

        match merge {
            EnrollMerge::ReplaceAll | EnrollMerge::AppendPrimary => {
                model.embeddings.extend(vectors)
            }
            EnrollMerge::ReplaceVariant | EnrollMerge::AppendVariant => model.upsert_extension(
                variant.expect("checked above"),
                vectors,
                merge == EnrollMerge::AppendVariant,
            ),
        }
        model.model_id = Some(model_id);
        model.embedding_dim = Some(dim);
        model.updated_at = Some(chrono::Utc::now());
        Ok(())
    }

    /// Load `path` only if it is a trusted root-only file. Missing file → `None`.
    pub fn load_trusted(path: &Path) -> Result<Option<Self>> {
        match storage::read_trusted(path, 0)? {
            Some(content) => {
                Ok(Some(Self::from_json(&content).with_context(|| {
                    format!("Corrupt face model {}", path.display())
                })?))
            }
            None => Ok(None),
        }
    }

    /// Atomically write to `path` (root-only permissions).
    pub fn save_secure(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        storage::write_private(path, content.as_bytes())
    }
}

/// Path of `username`'s model in the root store (the account must exist).
pub fn user_model_path(username: &str) -> Result<PathBuf> {
    lookup_user(username)?;
    Ok(Path::new(MODELS_DIR).join(format!("{username}.json")))
}

/// Where versions ≤ 0.2 kept the model (user-writable home directory).
pub fn legacy_model_path(username: &str) -> Result<(PathBuf, u32)> {
    let info = lookup_user(username)?;
    let path = info
        .home
        .join(".local/share/faceauth/models")
        .join(format!("{username}.json"));
    Ok((path, info.uid))
}

/// Load a user's model from the root store, or `None` if nothing is enrolled.
pub fn load_user_model(username: &str) -> Result<Option<FaceModel>> {
    let path = user_model_path(username)?;
    Ok(Database::load_trusted(&path)?.and_then(|mut db| db.remove_user(username)))
}

/// Read a legacy model owned by `owner_uid` (refuses symlinks).
pub fn load_legacy(path: &Path, owner_uid: u32) -> Result<Option<Database>> {
    match storage::read_owned_by(path, owner_uid)? {
        Some(content) => Ok(Some(Database::from_json(&content)?)),
        None => Ok(None),
    }
}

/// Validate vectors received from an unprivileged process (GUI via pkexec).
pub fn validate_vectors(vectors: &[Vec<f32>]) -> Result<usize> {
    if vectors.is_empty() {
        bail!("No vectors supplied");
    }
    if vectors.len() > MAX_PAYLOAD_VECTORS {
        bail!(
            "Too many vectors ({} > {MAX_PAYLOAD_VECTORS})",
            vectors.len()
        );
    }
    let dim = vectors[0].len();
    if !(16..=4096).contains(&dim) {
        bail!("Unexpected embedding dimension {dim}");
    }
    for v in vectors {
        if v.len() != dim {
            bail!("Vectors have different dimensions");
        }
        if !v.iter().all(|x| x.is_finite()) {
            bail!("Vector contains non-finite values");
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if (norm - 1.0).abs() > 0.01 {
            bail!("Vector is not L2-normalized (norm {norm:.3})");
        }
    }
    Ok(dim)
}

/// Pure permission policy for model files, split out for testing.
pub fn check_trusted(is_file: bool, uid: u32, mode: u32, expected_uid: u32) -> Result<()> {
    if !is_file {
        bail!("not a regular file");
    }
    if uid != expected_uid {
        bail!("owned by uid {uid}, expected {expected_uid}");
    }
    if mode & 0o077 != 0 {
        bail!(
            "permissions {:o} allow group/other access, expected 600",
            mode & 0o777
        );
    }
    Ok(())
}

#[cfg(unix)]
mod storage {
    use super::check_trusted;
    use anyhow::{Context, Result, bail};
    use std::fs::{self, DirBuilder, File, OpenOptions};
    use std::io::{ErrorKind, Read, Write};
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
    use std::path::Path;

    fn open_nofollow(path: &Path) -> Result<Option<File>> {
        match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
        {
            Ok(f) => Ok(Some(f)),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) if e.raw_os_error() == Some(libc::ELOOP) => {
                bail!("Refusing symlinked model file {}", path.display())
            }
            Err(e) => Err(e).with_context(|| format!("Failed to open {}", path.display())),
        }
    }

    fn read_all(mut f: File) -> Result<String> {
        let mut s = String::new();
        f.read_to_string(&mut s)?;
        Ok(s)
    }

    /// Read a file that must be root-controlled, including its directory.
    pub fn read_trusted(path: &Path, expected_uid: u32) -> Result<Option<String>> {
        if let Some(dir) = path.parent() {
            match fs::symlink_metadata(dir) {
                Ok(meta) => {
                    if !meta.is_dir() || meta.uid() != expected_uid || meta.mode() & 0o022 != 0 {
                        bail!(
                            "Untrusted model directory {} (must be a root-owned directory, not group/other writable)",
                            dir.display()
                        );
                    }
                }
                Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
                Err(e) => return Err(e.into()),
            }
        }
        let Some(file) = open_nofollow(path)? else {
            return Ok(None);
        };
        let meta = file.metadata()?;
        check_trusted(meta.is_file(), meta.uid(), meta.mode(), expected_uid)
            .with_context(|| format!("Untrusted model file {}", path.display()))?;
        read_all(file).map(Some)
    }

    /// Read a (legacy) file that must belong to `owner_uid`.
    pub fn read_owned_by(path: &Path, owner_uid: u32) -> Result<Option<String>> {
        let Some(file) = open_nofollow(path)? else {
            return Ok(None);
        };
        let meta = file.metadata()?;
        if !meta.is_file() || meta.uid() != owner_uid {
            bail!(
                "{} is not a regular file owned by uid {owner_uid}",
                path.display()
            );
        }
        read_all(file).map(Some)
    }

    /// Atomic write: `<path>.tmp` (0600) → fsync → rename. Parent is created 0700.
    pub fn write_private(path: &Path, data: &[u8]) -> Result<()> {
        let dir = path
            .parent()
            .context("Model path has no parent directory")?;
        DirBuilder::new().recursive(true).mode(0o700).create(dir)?;
        let tmp = path.with_extension("json.tmp");
        let _ = fs::remove_file(&tmp);
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&tmp)
                .with_context(|| format!("Failed to create {}", tmp.display()))?;
            f.write_all(data)?;
            f.sync_all()?;
        }
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600))?;
        fs::rename(&tmp, path)?;
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
        Ok(())
    }
}

#[cfg(not(unix))]
mod storage {
    use anyhow::{Result, bail};
    use std::path::Path;

    pub fn read_trusted(_: &Path, _: u32) -> Result<Option<String>> {
        bail!("Model storage is only supported on Unix")
    }
    pub fn read_owned_by(_: &Path, _: u32) -> Result<Option<String>> {
        bail!("Model storage is only supported on Unix")
    }
    pub fn write_private(_: &Path, _: &[u8]) -> Result<()> {
        bail!("Model storage is only supported on Unix")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[hot] = 1.0;
        v
    }

    fn enrollment(merge: EnrollMerge, variant: Option<&str>, n: usize) -> Enrollment {
        Enrollment {
            merge,
            variant: variant.map(str::to_string),
            label: None,
            vectors: (0..n).map(|i| unit(16, i)).collect(),
            model_id: "m1".into(),
        }
    }

    #[test]
    fn merge_modes() {
        let mut db = Database::new();
        db.apply_enrollment("u", enrollment(EnrollMerge::ReplaceAll, None, 2))
            .unwrap();
        assert_eq!(db.get_user("u").unwrap().embeddings.len(), 2);
        assert_eq!(db.get_user("u").unwrap().label, "u-default");

        db.apply_enrollment("u", enrollment(EnrollMerge::AppendPrimary, None, 3))
            .unwrap();
        assert_eq!(db.get_user("u").unwrap().embeddings.len(), 5);

        db.apply_enrollment(
            "u",
            enrollment(EnrollMerge::ReplaceVariant, Some("glasses"), 2),
        )
        .unwrap();
        db.apply_enrollment(
            "u",
            enrollment(EnrollMerge::AppendVariant, Some("glasses"), 1),
        )
        .unwrap();
        let m = db.get_user("u").unwrap();
        assert_eq!(m.extensions[0].embeddings.len(), 3);
        assert_eq!(m.sample_count(), 8);

        db.apply_enrollment(
            "u",
            enrollment(EnrollMerge::ReplaceVariant, Some("glasses"), 1),
        )
        .unwrap();
        assert_eq!(db.get_user("u").unwrap().extensions[0].embeddings.len(), 1);

        db.apply_enrollment("u", enrollment(EnrollMerge::ReplaceAll, None, 1))
            .unwrap();
        let m = db.get_user("u").unwrap();
        assert_eq!(m.embeddings.len(), 1);
        assert!(m.extensions.is_empty());
        assert_eq!(m.model_id.as_deref(), Some("m1"));
        assert_eq!(m.embedding_dim, Some(16));
    }

    #[test]
    fn variant_only_new_user() {
        let mut db = Database::new();
        db.apply_enrollment("u", enrollment(EnrollMerge::AppendVariant, Some("hat"), 1))
            .unwrap();
        let m = db.get_user("u").unwrap();
        assert!(m.embeddings.is_empty());
        assert_eq!(m.extensions[0].label, "hat");
    }

    #[test]
    fn merge_rejects_bad_input() {
        let mut db = Database::new();
        assert!(
            db.apply_enrollment("u", enrollment(EnrollMerge::ReplaceAll, None, 0))
                .is_err()
        );
        assert!(
            db.apply_enrollment("u", enrollment(EnrollMerge::ReplaceVariant, Some("  "), 1))
                .is_err()
        );
        db.apply_enrollment("u", enrollment(EnrollMerge::ReplaceAll, None, 1))
            .unwrap();
        let mut other = enrollment(EnrollMerge::AppendPrimary, None, 1);
        other.model_id = "m2".into();
        assert!(db.apply_enrollment("u", other).is_err());
        let mut wrong_dim = enrollment(EnrollMerge::AppendPrimary, None, 1);
        wrong_dim.vectors = vec![vec![1.0; 32]];
        assert!(db.apply_enrollment("u", wrong_dim).is_err());
    }

    #[test]
    fn variants_remove_rename() {
        let mut m = FaceModel::new("x".into(), vec![unit(16, 0)]);
        m.upsert_extension("glasses".into(), vec![unit(16, 1)], false);
        m.upsert_extension("hat".into(), vec![unit(16, 2)], false);
        assert!(m.rename_variant("glasses", "hat").is_err());
        m.rename_variant("glasses", "sunglasses").unwrap();
        assert!(m.remove_variant("glasses").is_err());
        m.remove_variant("sunglasses").unwrap();
        assert_eq!(m.extensions.len(), 1);
    }

    #[test]
    fn scoring_uses_best_set_and_top_k() {
        let mut m = FaceModel::new("x".into(), vec![unit(16, 0), unit(16, 1)]);
        m.upsert_extension("glasses".into(), vec![unit(16, 2)], false);
        assert_eq!(m.match_score(&unit(16, 2), 3), 0.0);
        let s = m.match_score(&unit(16, 0), 2);
        assert!((s - std::f32::consts::SQRT_2 / 2.0).abs() < 1e-6);
        assert_eq!(m.best_match_distance(&unit(16, 0)), 0.0);
        assert_eq!(m.match_score(&[1.0; 8], 1), f32::INFINITY);
    }

    #[test]
    fn compatibility_check() {
        let mut m = FaceModel::new("x".into(), vec![unit(16, 0)]);
        assert!(m.check_compatible("any", 16).is_ok()); // legacy: warn only
        assert!(m.check_compatible("any", 32).is_err());
        m.model_id = Some("m1".into());
        assert!(m.check_compatible("m1", 16).is_ok());
        assert!(m.check_compatible("m2", 16).is_err());
    }

    #[test]
    fn legacy_json_parses() {
        let json = r#"{"users":{"vlad":{"label":"vlad-default","embeddings":[[0.1,0.2]],
            "created_at":"2025-01-01T00:00:00Z"}}}"#;
        let db = Database::from_json(json).unwrap();
        let m = db.get_user("vlad").unwrap();
        assert!(m.extensions.is_empty() && m.model_id.is_none());
    }

    #[test]
    fn trusted_policy() {
        assert!(check_trusted(true, 0, 0o100600, 0).is_ok());
        assert!(check_trusted(true, 0, 0o100400, 0).is_ok());
        assert!(check_trusted(false, 0, 0o600, 0).is_err());
        assert!(check_trusted(true, 1000, 0o600, 0).is_err());
        assert!(check_trusted(true, 0, 0o644, 0).is_err());
        assert!(check_trusted(true, 0, 0o620, 0).is_err());
    }

    #[test]
    fn payload_validation() {
        let good: Vec<Vec<f32>> = (0..3).map(|i| unit(128, i)).collect();
        assert_eq!(validate_vectors(&good).unwrap(), 128);
        assert!(validate_vectors(&[]).is_err());
        assert!(validate_vectors(&[vec![1.0; 128]]).is_err()); // not normalized
        let mut nan = unit(128, 0);
        nan[5] = f32::NAN;
        assert!(validate_vectors(&[nan]).is_err());
        assert!(validate_vectors(&[unit(128, 0), unit(64, 0)]).is_err());
        assert!(validate_vectors(&[unit(4, 0)]).is_err());
        let many: Vec<Vec<f32>> = (0..=MAX_PAYLOAD_VECTORS).map(|_| unit(128, 0)).collect();
        assert!(validate_vectors(&many).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn secure_roundtrip_with_current_uid() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models").join("u.json");
        let mut db = Database::new();
        db.apply_enrollment("u", enrollment(EnrollMerge::ReplaceAll, None, 1))
            .unwrap();
        db.save_secure(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        let uid = unsafe { libc::geteuid() };
        let back = storage::read_trusted(&path, uid).unwrap().unwrap();
        assert!(back.contains("u-default"));
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(storage::read_trusted(&path, uid).is_err());
        let link = dir.path().join("models").join("link.json");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(storage::read_owned_by(&link, uid).is_err());
    }
}
