use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::recognition::FaceEmbedding;

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
}

impl FaceModel {
    pub fn new(label: String, embeddings: Vec<Vec<f32>>) -> Self {
        Self {
            label,
            embeddings,
            extensions: Vec::new(),
            created_at: chrono::Utc::now(),
        }
    }

    fn best_distance_in_set(embeddings: &[Vec<f32>], probe: &FaceEmbedding) -> f32 {
        if embeddings.is_empty() {
            return f32::INFINITY;
        }
        embeddings
            .iter()
            .map(|emb| {
                let emb_obj = FaceEmbedding::new(emb.clone());
                probe.euclidean_distance(&emb_obj)
            })
            .fold(f32::INFINITY, f32::min)
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

    /// Average embedding (simple mean) — primary set only (legacy helper).
    pub fn average_embedding(&self) -> Vec<f32> {
        if self.embeddings.is_empty() {
            return Vec::new();
        }
        let dim = self.embeddings[0].len();
        let mut sum = vec![0.0; dim];
        for emb in &self.embeddings {
            for (i, &val) in emb.iter().enumerate() {
                sum[i] += val;
            }
        }
        let count = self.embeddings.len() as f32;
        sum.iter().map(|&x| x / count).collect()
    }

    /// Best match distance to a probe embedding (minimum over primary + all extensions).
    pub fn best_match_distance(&self, probe: &FaceEmbedding) -> f32 {
        let mut best = Self::best_distance_in_set(&self.embeddings, probe);
        for ext in &self.extensions {
            best = best.min(Self::best_distance_in_set(&ext.embeddings, probe));
        }
        best
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Database {
    pub users: HashMap<String, FaceModel>,
}

impl Database {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let content = fs::read_to_string(path)?;
        let db: Database = serde_json::from_str(&content)?;
        Ok(db)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn add_model(&mut self, username: String, model: FaceModel) {
        self.users.insert(username, model);
    }

    pub fn remove_user(&mut self, username: &str) -> Option<FaceModel> {
        self.users.remove(username)
    }

    pub fn get_user(&self, username: &str) -> Option<&FaceModel> {
        self.users.get(username)
    }

    /// Find the best matching user for a probe embedding.
    /// Returns (username, distance) if distance < threshold.
    pub fn identify(&self, probe: &FaceEmbedding, threshold: f32) -> Option<(String, f32)> {
        let mut best_user = None;
        let mut best_distance = f32::INFINITY;

        for (username, model) in &self.users {
            let dist = model.best_match_distance(probe);
            if dist < best_distance {
                best_distance = dist;
                best_user = Some(username.clone());
            }
        }

        if best_distance < threshold {
            best_user.map(|user| (user, best_distance))
        } else {
            None
        }
    }

    /// Verify if the probe belongs to a specific user.
    pub fn verify(&self, username: &str, probe: &FaceEmbedding, threshold: f32) -> bool {
        match self.get_user(username) {
            Some(model) => model.best_match_distance(probe) < threshold,
            None => false,
        }
    }
}

/// Path utilities for faceauth data
pub fn get_data_dir() -> Result<PathBuf> {
    let dir = dirs::data_dir()
        .context("Could not determine data directory")?
        .join("faceauth");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn get_models_dir() -> Result<PathBuf> {
    let dir = get_data_dir()?.join("models");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn get_user_model_path(username: &str) -> Result<PathBuf> {
    let dir = get_models_dir()?;
    Ok(dir.join(format!("{}.json", username)))
}