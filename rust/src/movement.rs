use crate::model::{Result, normalize, softmax};
use serde::Deserialize;
use std::{collections::VecDeque, path::Path};

const STEPS: usize = 20;

#[derive(Deserialize)]
pub struct Lstm {
    pub classes: Vec<String>,
    pub feature_count: usize,
    hidden_size: usize,
    steps: usize,
    #[serde(rename = "W_ih")]
    input_weights: Vec<Vec<f32>>,
    #[serde(rename = "W_hh")]
    hidden_weights: Vec<Vec<f32>>,
    b_ih: Vec<f32>,
    b_hh: Vec<f32>,
    #[serde(rename = "W_fc")]
    output_weights: Vec<Vec<f32>>,
    b_fc: Vec<f32>,
}

impl Lstm {
    pub fn load(path: &Path) -> Result<Self> {
        let model: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        let h = model.hidden_size;
        let gates = 4 * h;
        let matrix = |rows: &[Vec<f32>], height: usize, width: usize| {
            rows.len() == height
                && rows
                    .iter()
                    .all(|row| row.len() == width && row.iter().all(|x| x.is_finite()))
        };
        if !(1..=256).contains(&h)
            || model.steps != STEPS
            || ![63, 65].contains(&model.feature_count)
            || !matrix(&model.input_weights, gates, model.feature_count)
            || !matrix(&model.hidden_weights, gates, h)
            || !matrix(&model.output_weights, model.classes.len(), h)
            || model.b_ih.len() != gates
            || model.b_hh.len() != gates
            || model.b_fc.len() != model.classes.len()
            || model.classes.is_empty()
        {
            return Err("Modelo LSTM incompatível".into());
        }
        Ok(model)
    }

    pub fn predict(&self, sequence: &[Vec<f32>]) -> Vec<f32> {
        let h = self.hidden_size;
        let mut hidden = vec![0.0; h];
        let mut cell = vec![0.0; h];
        let mut gates = vec![0.0; 4 * h];
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        for x in sequence {
            for (j, gate) in gates.iter_mut().enumerate() {
                *gate = dot(&self.input_weights[j], x)
                    + self.b_ih[j]
                    + dot(&self.hidden_weights[j], &hidden)
                    + self.b_hh[j];
            }
            for j in 0..h {
                cell[j] =
                    sigmoid(gates[h + j]) * cell[j] + sigmoid(gates[j]) * gates[2 * h + j].tanh();
                hidden[j] = sigmoid(gates[3 * h + j]) * cell[j].tanh();
            }
        }
        softmax(
            self.output_weights
                .iter()
                .zip(&self.b_fc)
                .map(|(row, bias)| dot(row, &hidden) + bias)
                .collect(),
        )
    }
}

#[derive(Clone)]
pub struct Sample {
    time: f64,
    pose: Vec<f32>,
    wrist: [f32; 2],
    scale: f32,
}

pub fn sequence_features(samples: &[Sample], features: usize) -> Vec<Vec<f32>> {
    let scale = samples.iter().map(|s| s.scale).sum::<f32>() / samples.len() as f32;
    let first = &samples[0];
    let value = |sample: &Sample, feature: usize| {
        if feature < 63 {
            sample.pose[feature]
        } else {
            (sample.wrist[feature - 63] - first.wrist[feature - 63]) / scale.max(1e-6)
        }
    };
    let mut index = 0;
    (0..STEPS)
        .map(|step| {
            let ratio = step as f64 / (STEPS - 1) as f64;
            let weight = if features == 63 {
                let position = ratio * (samples.len() - 1) as f64;
                index = (position as usize).min(samples.len() - 2);
                (position - index as f64) as f32
            } else {
                let time = first.time + ratio * (samples.last().unwrap().time - first.time);
                while index + 2 < samples.len() && samples[index + 1].time < time {
                    index += 1;
                }
                ((time - samples[index].time) / (samples[index + 1].time - samples[index].time))
                    as f32
            };
            (0..features)
                .map(|f| {
                    value(&samples[index], f) * (1.0 - weight)
                        + value(&samples[index + 1], f) * weight
                })
                .collect()
        })
        .collect()
}

#[derive(Default)]
pub struct Segmenter {
    recent: VecDeque<Sample>,
    active: Vec<Sample>,
    started: Option<f64>,
    last_motion: f64,
    speed: f32,
}

impl Segmenter {
    pub fn update(
        &mut self,
        points: Option<&[[f32; 3]; 21]>,
        now: f64,
    ) -> (Option<Vec<Sample>>, bool) {
        let Some(points) = points else {
            *self = Self::default();
            return (None, false);
        };
        let distance = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        let scale = points
            .iter()
            .map(|p| distance(p, &points[0]))
            .fold(1e-6, f32::max);
        let sample = Sample {
            time: now,
            pose: normalize(points),
            wrist: [points[0][0], points[0][1]],
            scale,
        };
        let mut moving = false;
        if let Some(previous) = self.recent.back() {
            let dt = now - previous.time;
            if dt > 0.35 {
                *self = Self::default();
            } else if dt > 0.0 {
                let shape = distance(&sample.pose, &previous.pose) / 21_f32.sqrt();
                let wrist = distance(&sample.wrist, &previous.wrist) / scale.max(previous.scale);
                self.speed = 0.55 * self.speed + 0.45 * shape.max(wrist) / dt.max(0.025) as f32;
                moving = self.speed >= 0.65;
            }
        }
        self.recent.push_back(sample.clone());
        while self.recent.front().is_some_and(|s| now - s.time > 0.12) {
            self.recent.pop_front();
        }
        if self.started.is_none() {
            if !moving {
                return (None, false);
            }
            self.started = Some(now);
            self.last_motion = now;
            self.active = self.recent.iter().cloned().collect();
        } else {
            self.active.push(sample);
            if moving {
                self.last_motion = now;
            }
        }
        let started = self.started.unwrap();
        if now - started < 2.0 && now - self.last_motion < 0.18 {
            return (None, true);
        }
        self.started = None;
        let complete = std::mem::take(&mut self.active);
        (
            (self.last_motion - started >= 0.25 && complete.len() >= 4).then_some(complete),
            false,
        )
    }
}
