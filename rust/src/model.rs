use serde_json::Value;
use std::path::Path;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn is_movement_letter(letter: &str) -> bool {
    matches!(letter, "H" | "J" | "K" | "X" | "Z")
}

#[derive(Default)]
pub struct SpeechSchedule(Option<(f64, String)>);

impl SpeechSchedule {
    pub fn schedule(&mut self, letter: &str, now: f64, movement_delay: f64) {
        let delay = if is_movement_letter(letter) {
            movement_delay
        } else {
            0.0
        };
        self.0 = Some((now + delay, letter.to_owned()));
    }

    pub fn take_ready(&mut self, now: f64) -> Option<String> {
        if self.0.as_ref().is_some_and(|(due, _)| now >= *due) {
            self.0.take().map(|(_, letter)| letter)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.0 = None;
    }
}

pub fn normalize(points: &[[f32; 3]; 21]) -> Vec<f32> {
    let mut values: Vec<f32> = points
        .iter()
        .flat_map(|p| (0..3).map(|d| p[d] - points[0][d]))
        .collect();
    let scale = values
        .chunks(3)
        .map(|p| p.iter().map(|x| x * x).sum::<f32>().sqrt())
        .fold(0.0, f32::max);
    if scale >= 1e-6 {
        for x in &mut values {
            *x /= scale;
        }
    }
    values
}

pub fn softmax(mut x: Vec<f32>) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for v in &mut x {
        *v = (*v - max).exp();
    }
    let sum: f32 = x.iter().sum();
    for v in &mut x {
        *v /= sum;
    }
    x
}

pub struct Mlp {
    pub classes: Vec<String>,
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,
}
impl Mlp {
    pub fn load(path: &Path) -> Result<Self> {
        let data: Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let count = data["n_layers"].as_u64().ok_or("n_layers ausente")? as usize;
        let classes: Vec<String> = serde_json::from_value(data["classes"].clone())?;
        let mut weights: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut biases: Vec<Vec<f32>> = Vec::new();
        let mut input = 63;
        for i in 0..count {
            let w: Vec<Vec<f32>> = serde_json::from_value(data[format!("W{i}")].clone())?;
            let b: Vec<f32> = serde_json::from_value(data[format!("b{i}")].clone())?;
            if w.len() != input
                || b.is_empty()
                || w.iter()
                    .any(|r| r.len() != b.len() || r.iter().any(|x| !x.is_finite()))
                || b.iter().any(|x| !x.is_finite())
            {
                return Err("Dimensões/pesos MLP inválidos".into());
            }
            input = b.len();
            weights.push(w);
            biases.push(b);
        }
        if count == 0 || input != classes.len() {
            return Err("Saída MLP incompatível".into());
        }
        Ok(Self {
            classes,
            weights,
            biases,
        })
    }
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), 63);
        let mut a = x.to_vec();
        for (i, (w, b)) in self.weights.iter().zip(&self.biases).enumerate() {
            let mut next = b.clone();
            for (value, row) in a.iter().zip(w) {
                for (out, weight) in next.iter_mut().zip(row) {
                    *out += value * weight;
                }
            }
            if i + 1 < self.weights.len() {
                for v in &mut next {
                    *v = v.max(0.0);
                }
            }
            a = next;
        }
        softmax(a)
    }

    pub fn classify(&self, x: &[f32], threshold: f32) -> (Option<String>, f32) {
        let probabilities = self.predict(x);
        let (index, &confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let label = &self.classes[index];
        let letter = (label != "_" && confidence >= threshold).then(|| label.clone());
        (letter, confidence)
    }
}

#[derive(Default)]
pub struct Stabilizer {
    current: Option<String>,
    since: f64,
    confirmed_current: bool,
    last: Option<String>,
    last_at: f64,
}
impl Stabilizer {
    pub fn update(
        &mut self,
        letter: Option<String>,
        now: f64,
        hold_seconds: f64,
    ) -> (Option<String>, f32, bool) {
        if letter != self.current {
            self.current = letter.clone();
            self.since = now;
            self.confirmed_current = false;
        }
        let Some(ref l) = letter else {
            return (None, 0.0, false);
        };
        let held = now - self.since;
        let progress = if hold_seconds > 0.0 {
            (held / hold_seconds).clamp(0.0, 1.0) as f32
        } else {
            1.0
        };
        let confirmed = held >= hold_seconds && !self.confirmed_current && self.confirm(l, now);
        if confirmed {
            self.confirmed_current = true;
        }
        (letter, progress, confirmed)
    }

    pub fn confirm(&mut self, letter: &str, now: f64) -> bool {
        if self.last.as_deref() == Some(letter) && now - self.last_at < 1.8 {
            return false;
        }
        self.last = Some(letter.to_owned());
        self.last_at = now;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalization_invariant() {
        let points = std::array::from_fn(|i| [i as f32 * 0.01, i as f32 * 0.02, i as f32 * 0.003]);
        let moved = points.map(|p| [p[0] * 2.0 + 0.3, p[1] * 2.0 + 0.1, p[2] * 2.0 - 0.2]);
        for (a, b) in normalize(&points).iter().zip(normalize(&moved)) {
            assert!((a - b).abs() < 1e-5);
        }
    }
    #[test]
    fn stability_and_cooldown() {
        let mut s = Stabilizer::default();
        assert!(!s.update(Some("A".into()), 10.0, 0.8).2);
        assert!(s.update(Some("A".into()), 10.81, 0.8).2);
        assert!(!s.update(Some("A".into()), 12.7, 0.8).2);
        assert!(!s.update(Some("L".into()), 13.0, 0.45).2);
        assert!(!s.update(Some("L".into()), 13.44, 0.45).2);
        assert!(s.update(Some("L".into()), 13.46, 0.45).2);
        assert!(!s.update(Some("L".into()), 20.0, 0.45).2);
        assert_eq!(s.update(None, 21.0, 0.8).1, 0.0);
        assert!(!s.update(Some("L".into()), 22.0, 0.8).2);
        assert!(s.update(Some("L".into()), 22.81, 0.8).2);
    }

    #[test]
    fn delayed_movement_speech_is_replaceable() {
        let mut speech = SpeechSchedule::default();
        speech.schedule("H", 0.0, 0.5);
        assert_eq!(speech.take_ready(0.49), None);
        assert_eq!(speech.take_ready(0.5).as_deref(), Some("H"));
        assert_eq!(speech.take_ready(0.6), None);
        speech.schedule("J", 1.0, 0.5);
        speech.schedule("A", 1.2, 0.5);
        assert_eq!(speech.take_ready(1.2).as_deref(), Some("A"));
        assert_eq!(speech.take_ready(1.5), None);
        speech.schedule("K", 2.0, 0.5);
        speech.clear();
        assert_eq!(speech.take_ready(3.0), None);
    }

    #[test]
    fn null_class_rejects_even_at_high_confidence() {
        let mut model = Mlp {
            classes: vec!["A".into(), "_".into()],
            weights: vec![vec![vec![0.0; 2]; 63]],
            biases: vec![vec![0.0, 12.0]],
        };
        let (letter, confidence) = model.classify(&[0.0; 63], 0.7);
        assert!(confidence > 0.99);
        assert_eq!(letter, None);
        model.biases[0].swap(0, 1);
        assert_eq!(model.classify(&[0.0; 63], 0.7).0.as_deref(), Some("A"));
    }
}
