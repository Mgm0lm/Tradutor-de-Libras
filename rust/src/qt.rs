//! Narrow C ABI boundary for the Qt Widgets frontend. Qt calls back on the UI thread.
use super::{App, Settings};
use image::RgbImage;
use std::{
    ffi::{CString, c_char, c_int, c_void},
    ptr,
};

#[repr(C)]
struct NativeWindow {
    _opaque: [u8; 0],
}

#[repr(C)]
struct WindowSettings {
    title: *const c_char,
    stylesheet: *const c_char,
    camera_ids: *const u32,
    camera_names: *const *const c_char,
    camera_count: usize,
    desync: c_int,
    tts: c_int,
    show_fps: c_int,
}

unsafe extern "C" {
    fn libras_ui_run(
        settings: *const WindowSettings,
        state: *mut c_void,
        tick: extern "C" fn(*mut c_void, *mut NativeWindow),
        action: extern "C" fn(*mut c_void, c_int, c_int),
    ) -> c_int;
    fn libras_ui_state(
        window: *mut NativeWindow,
        letter: *const c_char,
        progress: c_int,
        status: *const c_char,
        fps: *const c_char,
        threshold: f32,
    );
    fn libras_ui_frame(
        window: *mut NativeWindow,
        rgb: *const u8,
        width: c_int,
        height: c_int,
        landmarks: *const f32,
        confirmed: c_int,
    );
}

pub struct Window(*mut NativeWindow);

impl Window {
    pub fn state(
        &self,
        letter: Option<&str>,
        progress: f32,
        status: &str,
        fps: &str,
        threshold: f32,
    ) {
        let letter = letter.map(c_text);
        let status = c_text(status);
        let fps = c_text(fps);
        // Qt copies these UTF-8 strings before returning.
        unsafe {
            libras_ui_state(
                self.0,
                letter.as_ref().map_or(ptr::null(), |text| text.as_ptr()),
                (progress * 100.0) as c_int,
                status.as_ptr(),
                fps.as_ptr(),
                threshold,
            );
        }
    }

    pub fn frame(&self, image: &RgbImage, landmarks: Option<&[[f32; 3]; 21]>, confirmed: bool) {
        let points = landmarks.map_or(ptr::null(), |p| p.as_ptr().cast::<f32>());
        // The QPixmap is created synchronously before the Rust frame can be dropped.
        unsafe {
            libras_ui_frame(
                self.0,
                image.as_raw().as_ptr(),
                image.width() as c_int,
                image.height() as c_int,
                points,
                c_int::from(confirmed),
            );
        }
    }
}

fn c_text(text: &str) -> CString {
    CString::new(text.replace('\0', "")).expect("NULs were removed")
}

fn stylesheet(radius: u32) -> String {
    // Use the same QSS template as Python, including its configurable radius.
    include_str!("../../ui_styles.py")
        .split("\"\"\"")
        .nth(1)
        .expect("Qt stylesheet template missing")
        .replace("{UI_RADIUS - 4}", &radius.saturating_sub(4).to_string())
        .replace("{UI_RADIUS}", &radius.to_string())
        .replace("{{", "{")
        .replace("}}", "}")
}

pub fn run(mut app: App, settings: &Settings) -> c_int {
    extern "C" fn tick(state: *mut c_void, window: *mut NativeWindow) {
        // Qt calls synchronously while `run` owns both objects on this thread.
        unsafe {
            (&mut *state.cast::<App>()).tick(&Window(window));
        }
    }
    extern "C" fn action(state: *mut c_void, kind: c_int, value: c_int) {
        unsafe {
            (&mut *state.cast::<App>()).ui_action(kind, value);
        }
    }
    let title = c_text(&settings.app_title);
    let qss = c_text(&stylesheet(settings.ui_radius));
    let ids: Vec<_> = app.cameras.iter().map(|(id, _)| *id).collect();
    let names: Vec<_> = app.cameras.iter().map(|(_, name)| c_text(name)).collect();
    let name_ptrs: Vec<_> = names.iter().map(|name| name.as_ptr()).collect();
    let init = WindowSettings {
        title: title.as_ptr(),
        stylesheet: qss.as_ptr(),
        camera_ids: ids.as_ptr(),
        camera_names: name_ptrs.as_ptr(),
        camera_count: ids.len(),
        desync: c_int::from(settings.desync_inference),
        tts: c_int::from(settings.enable_tts),
        show_fps: c_int::from(settings.show_fps),
    };
    unsafe { libras_ui_run(&init, (&mut app as *mut App).cast(), tick, action) }
}
