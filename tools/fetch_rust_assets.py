"""Download official model/runtime assets; record source and SHA256 for packaging."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "modelo"
DOWNLOADS = ROOT / "tools" / "downloads"
ASSETS = {
    "palm_detection_mediapipe_2023feb.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx",
    "handpose_estimation_mediapipe_2023feb.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx",
    "palm-LICENSE": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/palm_detection_mediapipe/LICENSE",
    "handpose-LICENSE": "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/handpose_estimation_mediapipe/LICENSE",
}

def download(url, path):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return {"file": path.name, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    records = [download(url, MODEL / name) for name, url in ASSETS.items()]
    url = "https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-win-x64-1.23.2.zip"
    archive = DOWNLOADS / "onnxruntime-win-x64-1.23.2.zip"
    records.append(download(url, archive))
    with zipfile.ZipFile(archive) as package:
        for name in package.namelist():
            if name.endswith(("onnxruntime.dll", "onnxruntime_providers_shared.dll", "/LICENSE", "/ThirdPartyNotices.txt")):
                dest = DOWNLOADS / Path(name).name
                dest.write_bytes(package.read(name))
    (MODEL / "rust-assets.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Downloaded model and runtime assets.")

if __name__ == "__main__":
    main()
