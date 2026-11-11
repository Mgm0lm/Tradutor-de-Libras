#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod model;
mod movement;
mod qt;
mod vision;
use image::RgbImage;
use model::{MOVEMENT_LETTERS, Mlp, Result, Stabilizer};
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType},
};
use std::{
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

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
    fps: f32,
}
#[derive(Default)]
struct Shared {
    frame: Option<Frame>,
    detection: Option<Detection>,
    error: Option<String>,
    camera_fps: f32,
    gesture: Option<String>,
    model_ready: bool,
}
struct Control {
    stop: AtomicBool,
    camera: AtomicU32,
    threshold: AtomicU32,
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum MovementSeconds {
    Shared(f64),
    ByLetter(HashMap<String, f64>),
}

impl MovementSeconds {
    fn resolve(&self, default: f64, j_default: Option<f64>) -> HashMap<&'static str, f64> {
        MOVEMENT_LETTERS
            .into_iter()
            .map(|letter| {
                let j_default = j_default.filter(|_| letter == "J");
                let fallback = j_default.unwrap_or(default);
                let value = match self {
                    Self::Shared(value) => j_default.unwrap_or(*value),
                    Self::ByLetter(values) => values.get(letter).copied().unwrap_or(fallback),
                };
                (letter, value.max(0.0))
            })
            .collect()
    }
}

#[derive(serde::Deserialize)]
#[serde(default, rename_all = "SCREAMING_SNAKE_CASE")]
struct Settings {
    app_title: String,
    min_stable_seconds: f64,
    movement_min_stable_seconds: MovementSeconds,
    movement_tts_delay_seconds: MovementSeconds,
    xz_display_hold_seconds: f64,
    conf_threshold_default: f32,
    enable_tts: bool,
    desync_inference: bool,
    show_fps: bool,
    ui_radius: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            app_title: "Tradutor de LIBRAS - Alfabeto".into(),
            min_stable_seconds: 0.8,
            movement_min_stable_seconds: MovementSeconds::ByLetter(HashMap::new()),
            movement_tts_delay_seconds: MovementSeconds::ByLetter(HashMap::new()),
            xz_display_hold_seconds: 0.6,
            conf_threshold_default: 0.7,
            enable_tts: true,
            desync_inference: true,
            show_fps: true,
            ui_radius: 14,
        }
    }
}

impl Settings {
    fn load(base: &Path) -> Result<Self> {
        match std::fs::read_to_string(base.join("config_override.json")) {
            Ok(text) => Ok(serde_json::from_str(text.trim_start_matches('\u{feff}'))?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(error) => Err(error.into()),
        }
    }

    fn movement_hold_times(&self) -> HashMap<&'static str, f64> {
        self.movement_min_stable_seconds.resolve(0.1, None)
    }

    fn movement_speech_delays(&self) -> HashMap<&'static str, f64> {
        self.movement_tts_delay_seconds.resolve(0.5, Some(0.0))
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    #[test]
    fn per_letter_settings_override_only_listed_letters() {
        let settings: Settings = serde_json::from_str(
            r#"{
                "MOVEMENT_MIN_STABLE_SECONDS": {"H": 0.3, "K": 0.6, "Z": 0},
                "MOVEMENT_TTS_DELAY_SECONDS": {"J": 0.25, "X": 0.75}
            }"#,
        )
        .unwrap();
        let hold = settings.movement_hold_times();
        let speech = settings.movement_speech_delays();
        assert_eq!(
            MOVEMENT_LETTERS.map(|letter| hold[letter]),
            [0.3, 0.1, 0.6, 0.1, 0.0]
        );
        assert_eq!(
            MOVEMENT_LETTERS.map(|letter| speech[letter]),
            [0.5, 0.25, 0.5, 0.75, 0.5]
        );
        assert!(!hold.contains_key("A"));
        assert!(!speech.contains_key("A"));
    }

    #[test]
    fn defaults_and_legacy_shared_settings_keep_j_immediate() {
        let defaults = Settings::default();
        let example: Settings =
            serde_json::from_str(include_str!("../../config_override.example.json")).unwrap();
        assert_eq!(
            example.movement_hold_times(),
            defaults.movement_hold_times()
        );
        assert_eq!(
            example.movement_speech_delays(),
            defaults.movement_speech_delays()
        );
        assert_eq!(
            example.xz_display_hold_seconds,
            defaults.xz_display_hold_seconds
        );
        let legacy: Settings = serde_json::from_str(
            r#"{"MOVEMENT_MIN_STABLE_SECONDS": 0.3, "MOVEMENT_TTS_DELAY_SECONDS": 0.7}"#,
        )
        .unwrap();
        let hold = legacy.movement_hold_times();
        let speech = legacy.movement_speech_delays();
        assert_eq!(MOVEMENT_LETTERS.map(|letter| hold[letter]), [0.3; 5]);
        assert_eq!(
            MOVEMENT_LETTERS.map(|letter| speech[letter]),
            [0.7, 0.0, 0.7, 0.7, 0.7]
        );
    }

    #[test]
    fn xz_display_hold_discards_candidates_without_delaying_speech() {
        let settings = Settings::default();
        let mut app = App {
            shared: Arc::default(),
            control: Arc::new(Control {
                stop: AtomicBool::new(false),
                camera: AtomicU32::new(0),
                threshold: AtomicU32::new(0),
            }),
            stabilizer: Stabilizer::default(),
            start: Instant::now(),
            last_detection: 0,
            displayed_frame_id: 0,
            letter: None,
            confirmed_letter: None,
            progress: 0.0,
            movement_visible_until: 0.0,
            xz_display_hold_seconds: settings.xz_display_hold_seconds,
            threshold: 0.7,
            hold_seconds: settings.min_stable_seconds,
            movement_hold_seconds: settings.movement_hold_times(),
            movement_tts_delay: settings.movement_speech_delays(),
            speech_schedule: model::SpeechSchedule::default(),
            desync: true,
            tts: true,
            selected: 0,
            cameras: Vec::new(),
            audio: mpsc::sync_channel(1).0,
            pending_audio: Arc::default(),
        };
        let frame = |id, letter: &str| Detection {
            frame: Frame {
                id,
                generation: 0,
                captured: Instant::now(),
                image: Arc::new(RgbImage::new(1, 1)),
            },
            points: None,
            letter: Some(letter.into()),
            fps: 30.0,
        };

        app.update_recognition(None, Some(&frame(1, "X")), 0.0);
        app.update_recognition(None, Some(&frame(2, "X")), 0.125);
        assert_eq!(app.confirmed_letter.as_deref(), Some("X"));
        app.update_recognition(Some("H".into()), Some(&frame(3, "J")), 0.5);
        assert_eq!(app.speech_schedule.take_ready(0.624), None);
        assert_eq!(app.speech_schedule.take_ready(0.625).as_deref(), Some("X"));
        app.update_recognition(None, None, 0.7);
        assert_eq!(app.letter.as_deref(), Some("X"));
        assert_eq!(app.progress, 1.0);
        app.update_recognition(None, Some(&frame(4, "J")), 0.725);
        assert_eq!(app.letter.as_deref(), Some("J"));
        assert_eq!(app.progress, 0.0);

        app.update_recognition(Some("Z".into()), Some(&frame(5, "A")), 1.0);
        app.update_recognition(Some("H".into()), None, 1.599);
        assert_eq!(app.letter.as_deref(), Some("Z"));
        assert_eq!(app.speech_schedule.take_ready(1.599).as_deref(), Some("Z"));
        app.update_recognition(Some("H".into()), None, 1.6);
        assert_eq!(app.confirmed_letter.as_deref(), Some("H"));
    }
}

fn log_event(base: &Path, message: &str) {
    if let Ok(mut log) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(base.join("app.log"))
    {
        let _ = writeln!(log, "{message}");
    }
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
                        timer = Instant::now();
                        count = 0;
                        let mut s = capture_shared.lock().unwrap();
                        s.error = None;
                        s.frame = None;
                        s.detection = None;
                        s.gesture = None;
                        s.camera_fps = 0.0;
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
                        log_event(&base, &format!("Movimento experimental: {labels}"));
                        Some(model)
                    }
                    Err(error) => {
                        log_event(&base, &format!("Movimento indisponível: {error}"));
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
            let mut fps_epoch = Instant::now();
            let mut fps_frames = 0;
            let mut fps = 0.0;
            shared.lock().unwrap().model_ready = true;
            while !control.stop.load(Ordering::Relaxed) {
                let frame = shared.lock().unwrap().frame.clone();
                let Some(frame) = frame.filter(|f| f.id != last_id) else {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                };
                last_id = frame.id;
                let points = hands.detect(&frame.image)?;
                let mut letter = None;
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
                    letter = mlp.classify(&model::normalize(p), threshold).0;
                }
                fps_frames += 1;
                let elapsed = fps_epoch.elapsed().as_secs_f32();
                if elapsed >= 1.0 {
                    fps = fps_frames as f32 / elapsed;
                    fps_frames = 0;
                    fps_epoch = Instant::now();
                }
                if frame.generation == control.camera.load(Ordering::Relaxed) {
                    let mut state = shared.lock().unwrap();
                    if gesture.is_some() {
                        state.gesture = gesture;
                    }
                    state.detection = Some(Detection {
                        frame,
                        points,
                        letter,
                        fps,
                    });
                }
            }
            Ok(())
        };
        if let Err(e) = run() {
            log_event(&base, &format!("Erro no modelo: {e}"));
            shared.lock().unwrap().error = Some("Erro no modelo (veja app.log)".into());
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
    stabilizer: Stabilizer,
    start: Instant,
    last_detection: u64,
    displayed_frame_id: u64,
    letter: Option<String>,
    confirmed_letter: Option<String>,
    progress: f32,
    movement_visible_until: f64,
    xz_display_hold_seconds: f64,
    threshold: f32,
    hold_seconds: f64,
    movement_hold_seconds: HashMap<&'static str, f64>,
    movement_tts_delay: HashMap<&'static str, f64>,
    speech_schedule: model::SpeechSchedule,
    desync: bool,
    tts: bool,
    selected: u32,
    cameras: Vec<(u32, String)>,
    audio: mpsc::SyncSender<()>,
    pending_audio: Arc<Mutex<Option<AudioRequest>>>,
}
impl App {
    fn new(base: PathBuf, movement_enabled: bool, settings: &Settings) -> Self {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let threshold = settings.conf_threshold_default.clamp(0.1, 0.98);
        let control = Arc::new(Control {
            stop: AtomicBool::new(false),
            camera: AtomicU32::new(0),
            threshold: AtomicU32::new(threshold.to_bits()),
        });
        let cameras = nokhwa::query(ApiBackend::Auto)
            .unwrap_or_default()
            .iter()
            .filter_map(|c| {
                c.index()
                    .as_index()
                    .ok()
                    .map(|i| (i, format!("Câmera {i}")))
            })
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
        let audio_base = base.clone();
        std::thread::spawn(move || {
            while rx.recv().is_ok() {
                let request = pending.lock().unwrap().take();
                match request {
                    Some(AudioRequest::Speak(letter)) => {
                        let path = audio_base.join("tts_cache").join(format!("{letter}.mp3"));
                        if let Err(error) = play_audio(&path, true) {
                            log_event(&audio_base, &format!("TTS ({letter}): {error}"));
                        }
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
            stabilizer: Stabilizer::default(),
            start: Instant::now(),
            last_detection: 0,
            displayed_frame_id: 0,
            letter: None,
            confirmed_letter: None,
            progress: 0.0,
            movement_visible_until: 0.0,
            xz_display_hold_seconds: settings.xz_display_hold_seconds.max(0.0),
            threshold,
            hold_seconds: settings.min_stable_seconds.max(0.0),
            movement_hold_seconds: settings.movement_hold_times(),
            movement_tts_delay: settings.movement_speech_delays(),
            speech_schedule: model::SpeechSchedule::default(),
            desync: settings.desync_inference,
            tts: settings.enable_tts,
            selected,
            cameras,
            audio,
            pending_audio,
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
            let delay = self.movement_tts_delay.get(letter).copied().unwrap_or(0.0);
            self.speech_schedule.schedule(letter, now, delay);
        }
    }

    fn finish_confirmation(&mut self, letter: &str, now: f64, sequence_hold: f64) {
        let hold = if matches!(letter, "X" | "Z") {
            self.xz_display_hold_seconds
        } else {
            sequence_hold
        };
        self.movement_visible_until = now + hold;
        self.confirmed_letter = Some(letter.to_owned());
        self.schedule_confirmed_speech(letter, now);
    }

    fn xz_display_locked(&self, now: f64) -> bool {
        now < self.movement_visible_until && matches!(self.letter.as_deref(), Some("X" | "Z"))
    }
}
impl Drop for App {
    fn drop(&mut self) {
        self.control.stop.store(true, Ordering::Relaxed);
    }
}
impl App {
    fn ui_action(&mut self, action: i32, value: i32) {
        match action {
            0 => {
                self.selected = value as u32;
                self.control.camera.store(self.selected, Ordering::Relaxed);
                self.stabilizer = Stabilizer::default();
                self.letter = None;
                self.progress = 0.0;
                self.movement_visible_until = 0.0;
                self.speech_schedule.clear();
            }
            1 => {
                self.desync = value != 0;
                self.displayed_frame_id = 0;
            }
            2 => {
                self.threshold = (self.threshold + value as f32 * 0.05).clamp(0.10, 0.98);
                self.control
                    .threshold
                    .store(self.threshold.to_bits(), Ordering::Relaxed);
            }
            3 => {
                self.tts = value != 0;
                if !self.tts {
                    self.speech_schedule.clear();
                    self.request_audio(AudioRequest::Stop);
                }
            }
            4 => {
                if let Some(letter) = &self.confirmed_letter {
                    self.speech_schedule.clear();
                    self.speak(letter);
                }
            }
            _ => {}
        }
    }

    fn update_recognition(
        &mut self,
        gesture: Option<String>,
        detection: Option<&Detection>,
        now: f64,
    ) {
        if !self.xz_display_locked(now)
            && let Some(letter) = gesture
            && self.stabilizer.confirm(&letter, now)
        {
            self.stabilizer.update(None, now, self.hold_seconds);
            self.finish_confirmation(&letter, now, 0.65);
            self.letter = Some(letter);
            self.progress = 1.0;
        }
        if self.xz_display_locked(now) {
            if let Some(d) = detection {
                self.last_detection = d.frame.id;
            }
            return;
        }
        if let Some(d) = detection {
            if d.frame.id != self.last_detection {
                self.last_detection = d.frame.id;
                let hold_seconds = d
                    .letter
                    .as_deref()
                    .and_then(|letter| self.movement_hold_seconds.get(letter).copied())
                    .unwrap_or(self.hold_seconds);
                let (l, p, confirmed) = self.stabilizer.update(d.letter.clone(), now, hold_seconds);
                if confirmed && let Some(letter) = &l {
                    self.finish_confirmation(letter, now, 0.0);
                }
                if confirmed || now >= self.movement_visible_until {
                    self.letter = l;
                    self.progress = p;
                }
            }
        } else {
            self.stabilizer.update(None, now, self.hold_seconds);
            if now >= self.movement_visible_until {
                self.letter = None;
                self.progress = 0.0;
            }
        }
    }

    fn tick(&mut self, window: &qt::Window) {
        let (camera, detection, error, cam_fps, gesture, model_ready) = {
            let mut state = self.shared.lock().unwrap();
            (
                state.frame.clone(),
                state.detection.clone(),
                state.error.clone(),
                state.camera_fps,
                state.gesture.take(),
                state.model_ready,
            )
        };
        let detection = detection.filter(|d| {
            d.frame.generation == self.selected
                && d.frame.captured.elapsed() < Duration::from_secs(1)
        });
        self.update_recognition(
            gesture,
            detection.as_ref(),
            self.start.elapsed().as_secs_f64(),
        );
        let displayed = if self.desync {
            camera
                .as_ref()
                .filter(|frame| frame.generation == self.selected)
        } else {
            detection.as_ref().map(|result| &result.frame)
        };
        if let Some(frame) = displayed.filter(|f| f.id != self.displayed_frame_id) {
            let points = detection.as_ref().and_then(|result| result.points.as_ref());
            window.frame(&frame.image, points, self.progress >= 1.0);
            self.displayed_frame_id = frame.id;
        }
        let status = error.unwrap_or_else(|| match &self.letter {
            Some(letter) if self.progress >= 1.0 => format!("Confirmado: {letter}"),
            Some(letter) => format!("Detectando: {letter}..."),
            None if detection.is_some() => "Nenhuma mão reconhecida na área guia.".into(),
            None if model_ready => "Posicione a mão dentro da área guia.".into(),
            None => "Aguardando modelo...".into(),
        });
        let cam_fps = camera
            .as_ref()
            .filter(|frame| {
                frame.generation == self.selected
                    && frame.captured.elapsed() < Duration::from_secs(1)
            })
            .map_or(0.0, |_| cam_fps);
        let ai_fps = detection.as_ref().map_or(0.0, |result| result.fps);
        let fps = format!("Câmera: {cam_fps:.1} fps   |   IA: {ai_fps:.1} fps");
        window.state(
            self.letter.as_deref(),
            self.progress,
            &status,
            &fps,
            self.threshold,
        );
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
    let settings = Settings::load(&base).unwrap_or_else(|error| {
        log_event(
            &base,
            &format!("Configuração inválida, usando padrões: {error}"),
        );
        Settings::default()
    });
    let app = App::new(base, args.iter().any(|arg| arg == "--movement"), &settings);
    let code = qt::run(app, &settings);
    if code != 0 {
        return Err(format!("A janela Qt terminou com código {code}").into());
    }
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
