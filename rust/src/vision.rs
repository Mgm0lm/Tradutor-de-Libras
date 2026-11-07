//! MediaPipe palm and hand models exported by OpenCV Zoo (Apache-2.0).
//! RGB NHWC inputs. All geometry is in square-padded image pixels until final normalization.
use crate::model::Result;
use image::{RgbImage, imageops};
use ort::{session::Session, value::Tensor};
use std::path::Path;

pub struct Hands {
    palm: Session,
    hand: Session,
    anchors: Vec<[f32; 2]>,
}
impl Hands {
    pub fn load(base: &Path) -> Result<Self> {
        let palm = Session::builder()?
            .with_intra_threads(2)?
            .commit_from_file(base.join("modelo/palm_detection_mediapipe_2023feb.onnx"))?;
        let hand = Session::builder()?
            .with_intra_threads(2)?
            .commit_from_file(base.join("modelo/handpose_estimation_mediapipe_2023feb.onnx"))?;
        let mut anchors = Vec::new();
        for (grid, repeats) in [(24, 2), (12, 6)] {
            for y in 0..grid {
                for x in 0..grid {
                    for _ in 0..repeats {
                        anchors.push([
                            (x as f32 + 0.5) / grid as f32,
                            (y as f32 + 0.5) / grid as f32,
                        ]);
                    }
                }
            }
        }
        Ok(Self {
            palm,
            hand,
            anchors,
        })
    }
    pub fn detect(&mut self, img: &RgbImage) -> Result<Option<[[f32; 3]; 21]>> {
        let side = img.width().max(img.height()) as f32;
        let pad_x = (side - img.width() as f32) / 2.0;
        let pad_y = (side - img.height() as f32) / 2.0;
        let mut square = RgbImage::new(side as u32, side as u32);
        imageops::replace(&mut square, img, pad_x as i64, pad_y as i64);
        let input = imageops::resize(&square, 192, 192, imageops::FilterType::Triangle);
        let values: Vec<f32> = input.as_raw().iter().map(|x| *x as f32 / 255.0).collect();
        let outputs = self.palm.run(ort::inputs![Tensor::from_array((
            [1, 192, 192, 3],
            values
        ))?])?;
        let mut regress = None;
        let mut scores = None;
        for (_, o) in outputs.iter() {
            let (_, data) = o.try_extract_tensor::<f32>()?;
            if data.len() == 2016 * 18 {
                regress = Some(data.to_vec());
            } else if data.len() == 2016 {
                scores = Some(data.to_vec());
            }
        }
        let scores = scores.ok_or("Sortie palm scores absente")?;
        let (index, score) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .ok_or("Palm vide")?;
        if 1.0 / (1.0 + (-score).exp()) < 0.6 {
            return Ok(None);
        }
        let regress = regress.ok_or("Saída de regressão da palma ausente")?;
        let row = &regress[index * 18..index * 18 + 18];
        let anchor = self.anchors[index];
        let key: Vec<[f32; 2]> = (0..7)
            .map(|i| {
                [
                    (row[4 + i * 2] / 192.0 + anchor[0]) * side,
                    (row[5 + i * 2] / 192.0 + anchor[1]) * side,
                ]
            })
            .collect();
        // Rotate wrist -> middle finger base onto the upward image axis.
        let angle =
            std::f32::consts::FRAC_PI_2 - (-(key[2][1] - key[0][1])).atan2(key[2][0] - key[0][0]);
        let (sin, cos) = angle.sin_cos();
        let rotate = |p: [f32; 2]| [cos * p[0] + sin * p[1], -sin * p[0] + cos * p[1]];
        let pts: Vec<_> = key.iter().map(|p| rotate(*p)).collect();
        let minx = pts.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let maxx = pts.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let miny = pts.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let maxy = pts.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
        let size = ((maxx - minx).max(maxy - miny) * 3.0).max(1.0);
        let center = [
            (minx + maxx) / 2.0,
            (miny + maxy) / 2.0 - 0.4 * (maxy - miny),
        ];
        let mut crop = vec![0.0; 224 * 224 * 3];
        for y in 0..224 {
            for x in 0..224 {
                let rx = center[0] + ((x as f32 + 0.5) / 224.0 - 0.5) * size;
                let ry = center[1] + ((y as f32 + 0.5) / 224.0 - 0.5) * size;
                let px = cos * rx - sin * ry - pad_x;
                let py = sin * rx + cos * ry - pad_y;
                let ix = px.floor() as i32;
                let iy = py.floor() as i32;
                // Bilinear sampling with black border.
                for dy in 0..2 {
                    for dx in 0..2 {
                        let xx = ix + dx;
                        let yy = iy + dy;
                        if xx >= 0 && yy >= 0 && xx < img.width() as i32 && yy < img.height() as i32
                        {
                            let wx = if dx == 0 {
                                1.0 - (px - ix as f32)
                            } else {
                                px - ix as f32
                            };
                            let wy = if dy == 0 {
                                1.0 - (py - iy as f32)
                            } else {
                                py - iy as f32
                            };
                            let p = img.get_pixel(xx as u32, yy as u32);
                            for c in 0..3 {
                                crop[(y * 224 + x) * 3 + c] += p[c] as f32 / 255.0 * wx * wy;
                            }
                        }
                    }
                }
            }
        }
        drop(outputs);
        let outputs = self
            .hand
            .run(ort::inputs![Tensor::from_array(([1, 224, 224, 3], crop))?])?;
        // Named outputs of the pinned MediaPipe hand model: Identity=screen coordinates,
        // Identity_1=presence, Identity_2=handedness, Identity_3=world coordinates.
        let (_, confidence) = outputs["Identity_1"].try_extract_tensor::<f32>()?;
        if confidence[0] < 0.6 {
            return Ok(None);
        }
        let (_, landmarks) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let points = std::array::from_fn(|i| {
            let rx = center[0] + (landmarks[i * 3] / 224.0 - 0.5) * size;
            let ry = center[1] + (landmarks[i * 3 + 1] / 224.0 - 0.5) * size;
            [
                (cos * rx - sin * ry - pad_x) / img.width() as f32,
                (sin * rx + cos * ry - pad_y) / img.height() as f32,
                landmarks[i * 3 + 2] / 224.0 * size / img.width() as f32,
            ]
        });
        Ok(Some(points))
    }
}
