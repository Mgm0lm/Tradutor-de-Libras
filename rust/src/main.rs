#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod model;
mod movement;
mod vision;
use eframe::egui;
use image::RgbImage;
use model::{Mlp, Result, Stabilizer};
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

const CONNECTIONS: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];
#[derive(Clone)]
struct Frame {
    id: u64,
    generation: u32,
    captured: Instant,
    image: Arc<RgbImage>,
}
#[derive(Clone)]
struct Detection {
    frame: Frame,
    points: Option<[[f32; 3]; 21]>,
    letter: Option<String>,
    confidence: f32,
    fps: f32,
}
#[derive(Default)]
struct Shared {
    frame: Option<Frame>,
    detection: Option<Detection>,
    error: Option<String>,
    camera_fps: f32,
    gesture: Option<String>,
    movement_status: Option<String>,
}
struct Control {
    stop: AtomicBool,
    camera: AtomicU32,
    threshold: AtomicU32,
}

fn start_workers(
    base: PathBuf,
    shared: Arc<Mutex<Shared>>,
    control: Arc<Control>,
    movement_enabled: bool,
) {
    let capture_shared = shared.clone();
    let capture_control = control.clone();
    std::thread::spawn(move || {
        let mut active = None;
        let mut camera = None;
        let mut frame_id = 0;
        let mut timer = Instant::now();
        let mut count = 0;
        while !capture_control.stop.load(Ordering::Relaxed) {
            let wanted = capture_control.camera.load(Ordering::Relaxed);
            if active != Some(wanted) {
                camera.take();
                let requested = RequestedFormat::new::<RgbFormat>(
                    RequestedFormatType::AbsoluteHighestFrameRate,
                );
                match Camera::new(CameraIndex::Index(wanted), requested).and_then(|mut c| {
                    c.open_stream()?;
                    Ok(c)
                }) {
                    Ok(c) => {
                        camera = Some(c);
                        active = Some(wanted);
                        let mut s = capture_shared.lock().unwrap();
                        s.error = None;
                        s.frame = None;
                        s.detection = None;
                        s.gesture = None;
                    }
                    Err(e) => {
                        capture_shared.lock().unwrap().error =
                            Some(format!("Câmera {wanted}: {e}"));
                        std::thread::sleep(Duration::from_millis(500));
                        continue;
                    }
                }
            }
            let start = Instant::now();
            if let Some(c) = camera.as_mut() {
                match c.frame().and_then(|f| f.decode_image::<RgbFormat>()) {
                    Ok(mut img) => {
                        image::imageops::flip_horizontal_in_place(&mut img);
                        frame_id += 1;
                        count += 1;
                        let mut s = capture_shared.lock().unwrap();
                        s.frame = Some(Frame {
                            id: frame_id,
                            generation: wanted,
                            captured: Instant::now(),
                            image: Arc::new(img),
                        });
                        if timer.elapsed().as_secs_f32() >= 1.0 {
                            s.camera_fps = count as f32 / timer.elapsed().as_secs_f32();
                            count = 0;
                            timer = Instant::now();
                        }
                    }
                    Err(e) => {
                        capture_shared.lock().unwrap().error =
                            Some(format!("Falha de captura: {e}"));
                        camera.take();
                        active = None;
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(33).saturating_sub(start.elapsed()));
        }
    });
    std::thread::spawn(move || {
        let run = || -> Result<()> {
            ort::init_from(base.join("onnxruntime.dll"))?.commit();
            let mut hands = vision::Hands::load(&base)?;
            let mlp = Mlp::load(&base.join("modelo/landmark_model.json"))?;
            let movement_model = if movement_enabled {
                match movement::Lstm::load(&base.join("modelo/movement_model.json")) {
                    Ok(model) => {
                        let labels = model
                            .classes
                            .iter()
                            .filter(|l| l.as_str() != "_")
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ");
                        shared.lock().unwrap().movement_status =
                            Some(format!("Movimento experimental: {labels}"));
                        Some(model)
                    }
                    Err(error) => {
                        shared.lock().unwrap().movement_status =
                            Some(format!("Movimento indisponível: {error}"));
                        None
                    }
                }
            } else {
                None
            };
            let mut segmenter = movement::Segmenter::default();
            let mut generation = None;
            let clock = Instant::now();
            let mut last_id = 0;
            while !control.stop.load(Ordering::Relaxed) {
                let frame = shared.lock().unwrap().frame.clone();
                let Some(frame) = frame.filter(|f| f.id != last_id) else {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                };
                last_id = frame.id;
                let start = Instant::now();
                let points = hands.detect(&frame.image)?;
                let mut letter = None;
                let mut confidence = 0.0;
                let mut gesture = None;
                let mut moving = false;
                if let Some(model) = &movement_model {
                    if generation != Some(frame.generation) {
                        generation = Some(frame.generation);
                        segmenter = movement::Segmenter::default();
                    }
                    let (segment, active) = segmenter.update(
                        points.as_ref(),
                        frame
                            .captured
                            .saturating_duration_since(clock)
                            .as_secs_f64(),
                    );
                    moving = active || segment.is_some();
                    if let Some(segment) = segment {
                        let probabilities = model
                            .predict(&movement::sequence_features(&segment, model.feature_count));
                        let (index, probability) = probabilities
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .unwrap();
                        let label = &model.classes[index];
                        let threshold =
                            f32::from_bits(control.threshold.load(Ordering::Relaxed)).max(0.70);
                        if *probability >= threshold && model::is_movement_letter(label) {
                            gesture = Some(label.clone());
                        }
                    }
                }
                if let Some(p) = &points
                    && !moving
                {
                    let threshold = f32::from_bits(control.threshold.load(Ordering::Relaxed));
                    (letter, confidence) = mlp.classify(&model::normalize(p), threshold);
                }
                let fps = 1.0 / start.elapsed().as_secs_f32();
                if frame.generation == control.camera.load(Ordering::Relaxed) {
                    let mut state = shared.lock().unwrap();
                    if gesture.is_some() {
                        state.gesture = gesture;
                    }
                    state.detection = Some(Detection {
                        frame,
                        points,
                        letter,
                        confidence,
                        fps,
                    });
                }
            }
            Ok(())
        };
        if let Err(e) = run() {
            shared.lock().unwrap().error = Some(format!("Erro no modelo: {e}"));
        }
    });
}

#[cfg(windows)]
#[link(name = "winmm")]
unsafe extern "system" {
    fn mciSendStringW(command: *const u16, result: *mut u16, length: u32, window: isize) -> u32;
    fn PlaySoundW(sound: *const u16, module: isize, flags: u32) -> i32;
}
fn stop_wave_audio() {
    #[cfg(windows)]
    unsafe {
        PlaySoundW(std::ptr::null(), 0, 0);
    }
}

fn play_audio(path: &std::path::Path, asynchronous: bool) -> Result<()> {
    #[cfg(windows)]
    {
        let wav = path.with_extension("wav");
        if wav.exists() {
            let wide: Vec<u16> = wav
                .as_os_str()
                .to_string_lossy()
                .encode_utf16()
                .chain(Some(0))
                .collect();
            const SND_FILENAME: u32 = 0x00020000;
            const SND_NODEFAULT: u32 = 0x0002;
            const SND_ASYNC: u32 = 0x0001;
            let flags = SND_FILENAME | SND_NODEFAULT | if asynchronous { SND_ASYNC } else { 0 };
            if unsafe { PlaySoundW(wide.as_ptr(), 0, flags) } == 0 {
                return Err("Não foi possível reproduzir o áudio WAV".into());
            }
            return Ok(());
        }
        stop_wave_audio();
        let command = |text: String| -> u32 {
            let wide: Vec<u16> = text.encode_utf16().chain(Some(0)).collect();
            unsafe { mciSendStringW(wide.as_ptr(), std::ptr::null_mut(), 0, 0) }
        };
        command("close libras_voice".into());
        if command(format!(
            "open \"{}\" type mpegvideo alias libras_voice",
            path.display()
        )) != 0
        {
            return Err("Não foi possível abrir o áudio".into());
        }
        let status = command("play libras_voice wait".into());
        command("close libras_voice".into());
        if status != 0 {
            return Err("Não foi possível reproduzir o áudio".into());
        }
    }
    Ok(())
}

enum AudioRequest {
    Speak(String),
    Stop,
}

struct App {
    shared: Arc<Mutex<Shared>>,
    control: Arc<Control>,
    texture: Option<egui::TextureHandle>,
    stabilizer: Stabilizer,
    start: Instant,
    last_detection: u64,
    texture_id: u64,
    letter: Option<String>,
    confirmed_letter: Option<String>,
    progress: f32,
    movement_visible_until: f64,
    threshold: f32,
    hold_seconds: f64,
    movement_hold_seconds: f64,
    movement_tts_delay: f64,
    speech_schedule: model::SpeechSchedule,
    desync: bool,
    tts: bool,
    selected: u32,
    cameras: Vec<(u32, String)>,
    audio: mpsc::SyncSender<()>,
    pending_audio: Arc<Mutex<Option<AudioRequest>>>,
    audio_error: Arc<Mutex<Option<String>>>,
}
impl App {
    fn new(cc: &eframe::CreationContext<'_>, base: PathBuf, movement_enabled: bool) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::light());
        let shared = Arc::new(Mutex::new(Shared::default()));
        let control = Arc::new(Control {
            stop: AtomicBool::new(false),
            camera: AtomicU32::new(0),
            threshold: AtomicU32::new(0.7f32.to_bits()),
        });
        let cameras = nokhwa::query(ApiBackend::Auto)
            .unwrap_or_default()
            .iter()
            .filter_map(|c| c.index().as_index().ok().map(|i| (i, c.human_name())))
            .collect::<Vec<_>>();
        let cameras = if cameras.is_empty() {
            vec![(0, "Câmera 0 (padrão)".into())]
        } else {
            cameras
        };
        let selected = cameras[0].0;
        control.camera.store(selected, Ordering::Relaxed);
        let (audio, rx) = mpsc::sync_channel(1);
        let pending_audio = Arc::new(Mutex::new(None::<AudioRequest>));
        let pending = pending_audio.clone();
        let audio_error = Arc::new(Mutex::new(None));
        let errors = audio_error.clone();
        let audio_base = base.clone();
        std::thread::spawn(move || {
            while rx.recv().is_ok() {
                let request = pending.lock().unwrap().take();
                match request {
                    Some(AudioRequest::Speak(letter)) => {
                        let path = audio_base.join("tts_cache").join(format!("{letter}.mp3"));
                        let error = play_audio(&path, true).err().map(|e| e.to_string());
                        *errors.lock().unwrap() = error;
                    }
                    Some(AudioRequest::Stop) => stop_wave_audio(),
                    None => {}
                }
            }
            stop_wave_audio();
        });
        start_workers(base, shared.clone(), control.clone(), movement_enabled);
        Self {
            shared,
            control,
            texture: None,
            stabilizer: Stabilizer::default(),
            start: Instant::now(),
            last_detection: 0,
            texture_id: 0,
            letter: None,
            confirmed_letter: None,
            progress: 0.0,
            movement_visible_until: 0.0,
            threshold: 0.7,
            hold_seconds: 0.8,
            movement_hold_seconds: 0.1,
            movement_tts_delay: 0.5,
            speech_schedule: model::SpeechSchedule::default(),
            desync: true,
            tts: true,
            selected,
            cameras,
            audio,
            pending_audio,
            audio_error,
        }
    }

    fn speak(&self, letter: &str) {
        if self.tts {
            self.request_audio(AudioRequest::Speak(letter.to_owned()));
        }
    }

    fn request_audio(&self, request: AudioRequest) {
        *self.pending_audio.lock().unwrap() = Some(request);
        let _ = self.audio.try_send(());
    }

    fn schedule_confirmed_speech(&mut self, letter: &str, now: f64) {
        if self.tts {
            self.pending_audio.lock().unwrap().take();
            self.speech_schedule
                .schedule(letter, now, self.movement_tts_delay);
        }
    }
}
impl Drop for App {
    fn drop(&mut self) {
        self.control.stop.store(true, Ordering::Relaxed);
    }
}
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(30));
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        let (camera, detection, error, cam_fps, gesture, movement_status) = {
            let mut s = self.shared.lock().unwrap();
            (
                s.frame.clone(),
                s.detection.clone(),
                s.error.clone(),
                s.camera_fps,
                s.gesture.take(),
                s.movement_status.clone(),
            )
        };
        let detection = detection.filter(|d| {
            d.frame.generation == self.selected
                && d.frame.captured.elapsed() < Duration::from_secs(1)
        });
        let now = self.start.elapsed().as_secs_f64();
        if let Some(letter) = gesture
            && self.stabilizer.confirm(&letter, now)
        {
            self.stabilizer.update(None, now, self.hold_seconds);
            self.movement_visible_until = now + 0.65;
            self.letter = Some(letter.clone());
            self.confirmed_letter = Some(letter.clone());
            self.progress = 1.0;
            self.schedule_confirmed_speech(&letter, now);
        }
        if let Some(d) = &detection {
            if d.frame.id != self.last_detection {
                self.last_detection = d.frame.id;
                let hold_seconds = if d.letter.as_deref().is_some_and(model::is_movement_letter) {
                    self.movement_hold_seconds
                } else {
                    self.hold_seconds
                };
                let (l, p, confirmed) = self.stabilizer.update(d.letter.clone(), now, hold_seconds);
                if confirmed || now >= self.movement_visible_until {
                    self.letter = l;
                    self.progress = p;
                }
                if confirmed {
                    self.movement_visible_until = 0.0;
                    self.confirmed_letter = self.letter.clone();
                    if let Some(letter) = self.letter.clone() {
                        self.schedule_confirmed_speech(&letter, now);
                    }
                }
            }
        } else {
            self.stabilizer.update(None, now, self.hold_seconds);
            if now >= self.movement_visible_until {
                self.letter = None;
                self.progress = 0.0;
            }
        }
        let displayed = if self.desync {
            camera.as_ref()
        } else {
            detection.as_ref().map(|d| &d.frame)
        };
        if let Some(f) = displayed {
            if f.id != self.texture_id {
                let image = egui::ColorImage::from_rgb(
                    [f.image.width() as usize, f.image.height() as usize],
                    f.image.as_raw(),
                );
                if let Some(t) = &mut self.texture {
                    t.set(image, egui::TextureOptions::LINEAR);
                } else {
                    self.texture =
                        Some(ctx.load_texture("camera", image, egui::TextureOptions::LINEAR));
                }
                self.texture_id = f.id;
            }
        }
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::WHITE)
                    .inner_margin(24.0),
            )
            .show(ctx, |ui| {
                let avail = ui.available_size();
                let left = avail.x * 0.6 - 12.0;
                let right = avail.x - left - 24.0;
                ui.horizontal_top(|ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(left, avail.y),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            let size = egui::vec2(left, (avail.y - 115.0).max(200.0));
                            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                            ui.painter().rect_filled(
                                rect,
                                14.0,
                                egui::Color32::from_rgb(16, 19, 24),
                            );
                            if let Some(t) = &self.texture {
                                let original = t.size_vec2();
                                let scale = (size.x / original.x).min(size.y / original.y);
                                let video =
                                    egui::Rect::from_center_size(rect.center(), original * scale);
                                ui.painter().image(
                                    t.id(),
                                    video,
                                    egui::Rect::from_min_max(
                                        egui::pos2(0.0, 0.0),
                                        egui::pos2(1.0, 1.0),
                                    ),
                                    egui::Color32::WHITE,
                                );
                                if let Some(points) = detection.as_ref().and_then(|d| d.points) {
                                    let color = if self.progress >= 1.0 {
                                        egui::Color32::from_rgb(80, 220, 80)
                                    } else {
                                        egui::Color32::from_rgb(40, 130, 235)
                                    };
                                    let pos = |i: usize| {
                                        egui::pos2(
                                            video.left() + points[i][0] * video.width(),
                                            video.top() + points[i][1] * video.height(),
                                        )
                                    };
                                    for &(a, b) in CONNECTIONS {
                                        ui.painter().line_segment(
                                            [pos(a), pos(b)],
                                            egui::Stroke::new(2.0_f32, color),
                                        );
                                    }
                                    for i in 0..21 {
                                        ui.painter().circle_filled(pos(i), 4.0, color);
                                    }
                                }
                            }
                            ui.add_space(12.0);
                            egui::Frame::new()
                                .fill(egui::Color32::from_rgb(244, 246, 249))
                                .corner_radius(14.0)
                                .inner_margin(10.0)
                                .show(ui, |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label("Câmera:");
                                        let before = self.selected;
                                        egui::ComboBox::from_id_salt("camera")
                                            .selected_text(format!("Câmera {}", self.selected))
                                            .show_ui(ui, |ui| {
                                                for (id, name) in &self.cameras {
                                                    ui.selectable_value(
                                                        &mut self.selected,
                                                        *id,
                                                        name,
                                                    );
                                                }
                                            });
                                        if before != self.selected {
                                            self.control
                                                .camera
                                                .store(self.selected, Ordering::Relaxed);
                                            self.stabilizer = Stabilizer::default();
                                            self.letter = None;
                                            self.progress = 0.0;
                                            self.movement_visible_until = 0.0;
                                            self.speech_schedule.clear();
                                        }
                                        ui.checkbox(&mut self.desync, "Dessincronizar câmera / IA");
                                        ui.label("Confiança mín.:");
                                        if ui.button("−").clicked() {
                                            self.threshold = (self.threshold - 0.05).max(0.1);
                                        }
                                        ui.label(format!("{:.2}", self.threshold));
                                        if ui.button("+").clicked() {
                                            self.threshold = (self.threshold + 0.05).min(0.98);
                                        }
                                        self.control
                                            .threshold
                                            .store(self.threshold.to_bits(), Ordering::Relaxed);
                                        ui.add(egui::Slider::new(&mut self.hold_seconds, 0.45..=1.20)
                                            .step_by(0.05).fixed_decimals(2).suffix(" s").text("Estabilização"))
                                            .on_hover_text("Tempo de confirmação da letra estática. Tempos menores podem aumentar os falsos positivos.");
                                    });
                                    ui.horizontal_wrapped(|ui| {
                                        ui.add(egui::Slider::new(&mut self.movement_hold_seconds, 0.0..=1.20)
                                            .step_by(0.05).fixed_decimals(2).suffix(" s").text("Confirmação de movimento"))
                                            .on_hover_text("Tempo de estabilidade por quadro para H, J, K, X e Z.");
                                        ui.add(egui::Slider::new(&mut self.movement_tts_delay, 0.0..=2.0)
                                            .step_by(0.05).fixed_decimals(2).suffix(" s").text("Atraso da voz de movimento"))
                                            .on_hover_text("Espera entre a confirmação de H, J, K, X ou Z e o início da fala.");
                                    });
                                });
                            ui.label(format!(
                                "Câmera: {cam_fps:.1} fps | IA: {:.1} fps",
                                detection.as_ref().map_or(0.0, |d| d.fps)
                            ));
                        },
                    );
                    ui.add_space(16.0);
                    ui.allocate_ui_with_layout(
                        egui::vec2(right, avail.y),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            ui.label(egui::RichText::new("Letra reconhecida").size(20.0).strong());
                            ui.add_space(16.0);
                            egui::Frame::new()
                                .fill(egui::Color32::from_rgb(244, 246, 249))
                                .stroke(egui::Stroke::new(
                                    1.0_f32,
                                    egui::Color32::from_rgb(227, 231, 238),
                                ))
                                .corner_radius(14.0)
                                .inner_margin(20.0)
                                .show(ui, |ui| {
                                    ui.set_min_size(egui::vec2(
                                        (right - 40.0).max(100.0),
                                        (avail.y - 130.0).max(200.0),
                                    ));
                                    ui.vertical_centered(|ui| {
                                        ui.add_space((avail.y * 0.2).max(20.0));
                                        let color = if self.letter.is_none() {
                                            egui::Color32::from_gray(156)
                                        } else if self.progress >= 1.0 {
                                            egui::Color32::from_rgb(15, 138, 76)
                                        } else {
                                            egui::Color32::from_rgb(37, 99, 235)
                                        };
                                        ui.label(
                                            egui::RichText::new(
                                                self.letter.as_deref().unwrap_or("–"),
                                            )
                                            .size((right * 0.5).min(220.0))
                                            .strong()
                                            .color(color),
                                        );
                                        ui.add(egui::ProgressBar::new(self.progress));
                                        let status =
                                            error.clone().unwrap_or_else(|| match &self.letter {
                                                Some(l) if self.progress >= 1.0 => {
                                                    format!("Confirmado: {l}")
                                                }
                                                Some(l) => format!("Detectando: {l}..."),
                                                None => "Posicione a mão diante da câmera.".into(),
                                            });
                                        ui.label(
                                            egui::RichText::new(status)
                                                .color(egui::Color32::from_rgb(37, 99, 235)),
                                        );
                                        if let Some(d) = &detection {
                                            if d.points.is_some() {
                                                ui.small(format!(
                                                    "Confiança: {:.0}%",
                                                    d.confidence * 100.0
                                                ));
                                            }
                                        }
                                    });
                                });
                            ui.add_space(16.0);
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .checkbox(&mut self.tts, "Falar em voz alta (TTS)")
                                    .changed()
                                    && !self.tts
                                {
                                    self.speech_schedule.clear();
                                    self.request_audio(AudioRequest::Stop);
                                }
                                if ui.button("Repetir").clicked() && self.tts {
                                    if let Some(l) = &self.confirmed_letter {
                                        self.speech_schedule.clear();
                                        self.speak(l);
                                    }
                                }
                                if ui.button("Sair (ESC)").clicked() {
                                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                                }
                            });
                            if let Some(status) = &movement_status {
                                ui.small(status);
                            }
                            if let Some(e) = self.audio_error.lock().unwrap().as_ref() {
                                ui.colored_label(egui::Color32::RED, e);
                            }
                        },
                    );
                });
            });
        if let Some(letter) = self
            .speech_schedule
            .take_ready(self.start.elapsed().as_secs_f64())
        {
            self.speak(&letter);
        }
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let base = args
        .iter()
        .position(|s| s == "--base")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or(std::env::current_exe()?.parent().unwrap().to_path_buf());
    if let Some(i) = args.iter().position(|s| s == "--audio-test") {
        let letters = &args[i + 1..];
        if letters.is_empty()
            || letters
                .iter()
                .any(|l| l.len() != 1 || !l.as_bytes()[0].is_ascii_uppercase())
        {
            return Err("Informe uma ou mais letras de A a Z após --audio-test".into());
        }
        for (index, letter) in letters.iter().enumerate() {
            let start = Instant::now();
            play_audio(
                &base.join("tts_cache").join(format!("{letter}.mp3")),
                index + 1 < letters.len(),
            )?;
            println!("{letter}: {:.1} ms", start.elapsed().as_secs_f64() * 1000.0);
        }
        return Ok(());
    }
    if let Some(i) = args.iter().position(|s| s == "--evaluate") {
        ort::init_from(base.join("onnxruntime.dll"))?.commit();
        let mut hands = vision::Hands::load(&base)?;
        let mlp = Mlp::load(&base.join("modelo/landmark_model.json"))?;
        let files: Vec<String> =
            serde_json::from_slice(&std::fs::read(args.get(i + 1).ok_or("Lista ausente")?)?)?;
        let mut results = Vec::new();
        for path in files {
            let image = image::open(&path)?.to_rgb8();
            let points = hands.detect(&image)?;
            let probabilities = points.as_ref().map(|p| mlp.predict(&model::normalize(p)));
            results.push(
                serde_json::json!({"file":path,"points":points,"probabilities":probabilities}),
            );
        }
        std::fs::write(
            args.get(i + 2).ok_or("Saída ausente")?,
            serde_json::to_vec(&results)?,
        )?;
        return Ok(());
    }
    // Headless paths exercise the exact native runtime used by the GUI.
    if let Some(i) = args.iter().position(|s| s == "--predict") {
        let input: Vec<Vec<f32>> = serde_json::from_slice(&std::fs::read(
            args.get(i + 1).ok_or("Arquivo de entrada ausente")?,
        )?)?;
        let mlp = Mlp::load(&base.join("modelo/landmark_model.json"))?;
        let output: Vec<_> = input.iter().map(|x| mlp.predict(x)).collect();
        std::fs::write(
            args.get(i + 2).ok_or("Arquivo de saída ausente")?,
            serde_json::to_vec(&output)?,
        )?;
        return Ok(());
    }
    if let Some(i) = args.iter().position(|s| s == "--predict-movement") {
        let model = movement::Lstm::load(&base.join("modelo/movement_model.json"))?;
        let input: Vec<Vec<Vec<f32>>> =
            serde_json::from_slice(&std::fs::read(args.get(i + 1).ok_or("Entrada ausente")?)?)?;
        if input
            .iter()
            .any(|s| s.len() != 20 || s.iter().any(|row| row.len() != model.feature_count))
        {
            return Err("Dimensões de sequência inválidas".into());
        }
        let output: Vec<_> = input
            .iter()
            .map(|sequence| model.predict(sequence))
            .collect();
        std::fs::write(
            args.get(i + 2).ok_or("Saída ausente")?,
            serde_json::to_vec(&output)?,
        )?;
        return Ok(());
    }
    if let Some(i) = args.iter().position(|s| s == "--image") {
        ort::init_from(base.join("onnxruntime.dll"))?.commit();
        let mut hands = vision::Hands::load(&base)?;
        let img = image::open(args.get(i + 1).ok_or("Imagem ausente")?)?.to_rgb8();
        let points = hands.detect(&img)?;
        std::fs::write(
            args.get(i + 2).ok_or("Saída ausente")?,
            serde_json::to_vec(&points)?,
        )?;
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Tradutor de LIBRAS - Alfabeto")
            .with_fullscreen(true)
            .with_min_inner_size([960.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Tradutor de LIBRAS - Alfabeto",
        options,
        Box::new(move |cc| {
            Ok(Box::new(App::new(
                cc,
                base,
                args.iter().any(|arg| arg == "--movement"),
            )))
        }),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        let message = format!("{error}\n");
        eprintln!("{message}");
        if let Ok(exe) = std::env::current_exe() {
            let _ = std::fs::write(exe.with_file_name("app-error.log"), message);
        }
        std::process::exit(1);
    }
}
