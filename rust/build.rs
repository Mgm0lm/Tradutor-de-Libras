use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=LIBRAS_QT_DIR");
    println!("cargo:rerun-if-changed=src/qt_window.cpp");
    let sdk = env::var_os("LIBRAS_QT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("../tools/downloads/qt-6.11.1")
        });
    assert!(
        sdk.join("lib/Qt6Widgets.lib").is_file(),
        "Qt Widgets ausente. Execute tools/fetch_qt_sdk.py ou defina LIBRAS_QT_DIR."
    );
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag("/permissive-")
        .flag("/Zc:__cplusplus")
        .flag("/Zc:preprocessor")
        .flag("/utf-8")
        .include(sdk.join("include"))
        .file("src/qt_window.cpp")
        .compile("libras_qt_window");
    println!(
        "cargo:rustc-link-search=native={}",
        sdk.join("lib").display()
    );
    for library in ["Qt6Widgets", "Qt6Gui", "Qt6Core", "user32"] {
        println!("cargo:rustc-link-lib={library}");
    }
}
