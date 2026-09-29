//! Embedding comparison, scoring and the consecutive-match tracker.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Euclidean distance. Vectors of different (or zero) length never match.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Mean of the `k` smallest distances from `probe` to the vectors of one sample set.
/// With `k == 1` this is the nearest-neighbour distance. Empty sets never match.
///
/// Averaging several neighbours means one stray enrollment sample cannot, on its
/// own, make an unrelated face match.
pub fn set_score(set: &[Vec<f32>], probe: &[f32], k: usize) -> f32 {
    if set.is_empty() {
        return f32::INFINITY;
    }
    let mut d: Vec<f32> = set.iter().map(|v| l2_distance(v, probe)).collect();
    d.sort_by(f32::total_cmp);
    let k = k.clamp(1, d.len());
    d[..k].iter().sum::<f32>() / k as f32
}

/// Scale `v` to unit length in place (no-op for a zero vector).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v {
            *x /= norm;
        }
    }
}

/// Requires `required` consecutive matching frames. A frame whose face did not
/// match resets the streak; frames without a usable face are ignored.
#[derive(Debug, Clone)]
pub struct MatchTracker {
    required: u32,
    consecutive: u32,
    best_score: f32,
}

impl MatchTracker {
    pub fn new(required: u32) -> Self {
        Self {
            required: required.max(1),
            consecutive: 0,
            best_score: f32::INFINITY,
        }
    }

    /// Record a frame with a face; returns true once the streak is long enough.
    pub fn observe(&mut self, score: f32, matched: bool) -> bool {
        self.best_score = self.best_score.min(score);
        if matched {
            self.consecutive += 1;
        } else {
            self.consecutive = 0;
        }
        self.is_satisfied()
    }

    pub fn is_satisfied(&self) -> bool {
        self.consecutive >= self.required
    }

    pub fn consecutive(&self) -> u32 {
        self.consecutive
    }

    pub fn required(&self) -> u32 {
        self.required
    }

    pub fn best_score(&self) -> f32 {
        self.best_score
    }
}

/// Which way the user should turn to fill the missing pose bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoseHint {
    TurnLeft,
    TurnRight,
    LookStraight,
}

impl PoseHint {
    /// i18n key of the user-facing message (see `crate::i18n`).
    pub fn message_key(self) -> &'static str {
        match self {
            Self::TurnLeft => "hint.turn_left",
            Self::TurnRight => "hint.turn_right",
            Self::LookStraight => "hint.look_straight",
        }
    }
}

/// Outcome of offering one enrollment sample to [`PoseCollector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleDecision {
    Accepted,
    /// Nearly identical to a sample already collected.
    Duplicate,
    /// This pose already has enough samples; the hint names the missing one.
    PoseFull(PoseHint),
}

/// Collects enrollment samples spread over three yaw buckets (turned left,
/// frontal, turned right) and drops near-duplicates. After `relax_after` of the
/// time budget has passed, any non-duplicate pose is accepted.
#[derive(Debug, Clone)]
pub struct PoseCollector {
    target: usize,
    quota: usize,
    counts: [usize; 3],
    vectors: Vec<Vec<f32>>,
    relax_after: f32,
}

/// Minimum L2 distance between two kept samples.
pub const DUPLICATE_DISTANCE: f32 = 0.08;
/// Yaw (degrees) beyond which a sample counts as turned.
pub const POSE_YAW_SPLIT: f32 = 8.0;

impl PoseCollector {
    pub fn new(target: usize) -> Self {
        let target = target.max(1);
        Self {
            target,
            quota: target.div_ceil(3),
            counts: [0; 3],
            vectors: Vec::with_capacity(target),
            relax_after: 0.6,
        }
    }

    /// Bucket 0 = turned left (yaw > split), 1 = frontal, 2 = turned right.
    fn bucket(yaw: Option<f32>) -> usize {
        match yaw {
            Some(y) if y > POSE_YAW_SPLIT => 0,
            Some(y) if y < -POSE_YAW_SPLIT => 2,
            _ => 1,
        }
    }

    /// The pose still missing the most samples.
    pub fn missing_hint(&self) -> Option<PoseHint> {
        if self.is_complete() {
            return None;
        }
        let (idx, _) = self
            .counts
            .iter()
            .enumerate()
            .min_by_key(|&(i, c)| (*c, if i == 1 { 0 } else { 1 }))?;
        Some(match idx {
            0 => PoseHint::TurnLeft,
            2 => PoseHint::TurnRight,
            _ => PoseHint::LookStraight,
        })
    }

    /// `elapsed_fraction` = elapsed / timeout, in `[0, 1]`.
    pub fn offer(
        &mut self,
        vector: Vec<f32>,
        yaw: Option<f32>,
        elapsed_fraction: f32,
    ) -> SampleDecision {
        if self
            .vectors
            .iter()
            .any(|v| l2_distance(v, &vector) < DUPLICATE_DISTANCE)
        {
            return SampleDecision::Duplicate;
        }
        let b = Self::bucket(yaw);
        let relaxed = elapsed_fraction >= self.relax_after;
        if !relaxed && self.counts[b] >= self.quota {
            return SampleDecision::PoseFull(self.missing_hint().unwrap_or(PoseHint::LookStraight));
        }
        self.counts[b] += 1;
        self.vectors.push(vector);
        SampleDecision::Accepted
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn target(&self) -> usize {
        self.target
    }

    pub fn is_complete(&self) -> bool {
        self.vectors.len() >= self.target
    }

    pub fn into_vectors(self) -> Vec<Vec<f32>> {
        self.vectors
    }
}

/// min / median / p95 of a list of scores (ignores non-finite values).
pub fn score_stats(scores: &[f32]) -> Option<(f32, f32, f32)> {
    let mut s: Vec<f32> = scores.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return None;
    }
    s.sort_by(f32::total_cmp);
    let at = |q: f32| s[(((s.len() - 1) as f32) * q).round() as usize];
    Some((s[0], at(0.5), at(0.95)))
}

/// Identity of a model file: first 16 bytes of its SHA-256, hex encoded.
/// Stored with enrolled embeddings so a swapped recognizer is detected.
pub fn file_fingerprint(path: &Path) -> Result<String> {
    let data =
        std::fs::read(path).with_context(|| format!("Failed to read model {}", path.display()))?;
    let digest = Sha256::digest(&data);
    Ok(digest[..16].iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_basics() {
        assert_eq!(l2_distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
        assert_eq!(l2_distance(&[1.0], &[1.0, 0.0]), f32::INFINITY);
        assert_eq!(l2_distance(&[], &[]), f32::INFINITY);
    }

    #[test]
    fn top_k_score() {
        let set = vec![vec![0.0], vec![1.0], vec![2.0], vec![10.0]];
        assert_eq!(set_score(&set, &[0.0], 1), 0.0);
        assert_eq!(set_score(&set, &[0.0], 3), 1.0);
        // k larger than the set uses every vector
        assert_eq!(set_score(&set, &[0.0], 10), 13.0 / 4.0);
        assert_eq!(set_score(&[], &[0.0], 3), f32::INFINITY);
        // k = 0 behaves like 1
        assert_eq!(set_score(&set, &[0.0], 0), 0.0);
    }

    #[test]
    fn normalize() {
        let mut v = [3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = [0.0, 0.0];
        l2_normalize(&mut z);
        assert_eq!(z, [0.0, 0.0]);
    }

    #[test]
    fn tracker_needs_consecutive_matches() {
        let mut t = MatchTracker::new(3);
        assert!(!t.observe(0.3, true));
        assert!(!t.observe(0.4, true));
        assert!(!t.observe(0.9, false)); // reset
        assert_eq!(t.consecutive(), 0);
        assert!(!t.observe(0.3, true));
        assert!(!t.observe(0.3, true));
        assert!(t.observe(0.2, true));
        assert_eq!(t.best_score(), 0.2);
    }

    #[test]
    fn tracker_required_at_least_one() {
        let mut t = MatchTracker::new(0);
        assert_eq!(t.required(), 1);
        assert!(t.observe(0.1, true));
    }

    fn unit(i: usize) -> Vec<f32> {
        let mut v = vec![0.0; 8];
        v[i] = 1.0;
        v
    }

    #[test]
    fn pose_collector_spreads_samples() {
        let mut c = PoseCollector::new(3);
        assert_eq!(c.missing_hint(), Some(PoseHint::LookStraight));
        assert_eq!(c.offer(unit(0), Some(0.0), 0.0), SampleDecision::Accepted);
        // frontal bucket is full (quota 1) → ask for a turned pose
        assert!(matches!(
            c.offer(unit(1), Some(1.0), 0.1),
            SampleDecision::PoseFull(_)
        ));
        assert_eq!(c.offer(unit(0), Some(20.0), 0.1), SampleDecision::Duplicate);
        assert_eq!(c.offer(unit(2), Some(20.0), 0.1), SampleDecision::Accepted);
        assert_eq!(c.missing_hint(), Some(PoseHint::TurnRight));
        assert_eq!(c.offer(unit(3), Some(-20.0), 0.2), SampleDecision::Accepted);
        assert!(c.is_complete());
        assert_eq!(c.missing_hint(), None);
        assert_eq!(c.into_vectors().len(), 3);
    }

    #[test]
    fn pose_collector_relaxes_late() {
        let mut c = PoseCollector::new(3);
        c.offer(unit(0), None, 0.0);
        assert_eq!(c.offer(unit(1), None, 0.9), SampleDecision::Accepted);
    }

    #[test]
    fn stats() {
        assert_eq!(score_stats(&[]), None);
        assert_eq!(score_stats(&[f32::INFINITY]), None);
        let s: Vec<f32> = (1..=21).map(|x| x as f32).collect();
        assert_eq!(score_stats(&s), Some((1.0, 11.0, 20.0)));
    }

    #[test]
    fn fingerprint_is_stable() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.onnx");
        std::fs::write(&p, b"abc").unwrap();
        let id = file_fingerprint(&p).unwrap();
        assert_eq!(id.len(), 32);
        // sha256("abc") = ba7816bf8f01cfea414140de5dae2223...
        assert_eq!(id, "ba7816bf8f01cfea414140de5dae2223");
    }
}
