"""Create a versioned portable Windows bundle without overwriting older releases."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["rust", "python"])
    parser.add_argument("version")
    args = parser.parse_args()
    name = f"TradutorLibras-{args.kind}-{args.version}-windows-x64"
    bundle = ROOT / "dist" / name
    if bundle.exists():
        raise SystemExit(f"Destino já existe, escolha outra versão: {bundle}")
    bundle.mkdir(parents=True)
    (bundle / "modelo").mkdir()
    if args.kind == "rust":
        shutil.copy2(ROOT / "rust/target/release/tradutor-libras.exe", bundle / "TradutorLibras.exe")
        shutil.copy2(ROOT / "tools/downloads/onnxruntime.dll", bundle)
        for filename in ["landmark_model.json", "palm_detection_mediapipe_2023feb.onnx", "handpose_estimation_mediapipe_2023feb.onnx", "rust-assets.json", "palm-LICENSE", "handpose-LICENSE"]:
            shutil.copy2(ROOT / "modelo" / filename, bundle / "modelo")
        if (ROOT / "modelo/movement_model.json").exists():
            shutil.copy2(ROOT / "modelo/movement_model.json", bundle / "modelo")
        for filename in ["LICENSE", "ThirdPartyNotices.txt"]:
            shutil.copy2(ROOT / "tools/downloads" / filename, bundle / ("onnxruntime-" + filename))
    else:
        source = ROOT / "dist/python-build/TradutorLibra"
        shutil.copytree(source, bundle, dirs_exist_ok=True)
        (bundle / "TradutorLibra.exe").rename(bundle / "TradutorLibras.exe")
        shutil.copy2(ROOT / "modelo/landmark_model.npz", bundle / "modelo")
        if (ROOT / "modelo/movement_model.npz").exists():
            shutil.copy2(ROOT / "modelo/movement_model.npz", bundle / "modelo")
    shutil.copytree(ROOT / "tts_cache", bundle / "tts_cache")
    if any((bundle / "modelo").glob("movement_model.*")):
        (bundle / "Iniciar-com-movimento.cmd").write_text(
            '@echo off\nstart "" "%~dp0TradutorLibras.exe" --movement\n', encoding="ascii"
        )
    for filename in ["README.md", "CHANGELOG.md", "LICENSE", "config_override.example.json"]:
        shutil.copy2(ROOT / filename, bundle)
    records = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(bundle.rglob("*")) if p.is_file()}
    (bundle / "SHA256.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(bundle), "zip", bundle.parent, bundle.name)
    print(archive)

if __name__ == "__main__":
    main()
