"""Local integration comparison; results are diagnostic, not independent accuracy."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classifier import MLPNumpy
from landmarks_utils import normalize_landmarks, mediapipe_landmarks_to_xyz

def main():
    output = ROOT / "test-output"
    output.mkdir(exist_ok=True)
    files = [p for folder in sorted((ROOT / "training").iterdir()) if folder.is_dir()
             for p in sorted(folder.glob("*.png"))[:3]]
    if not files:
        raise SystemExit("Nenhuma imagem local disponível")
    source, target = output / "vision-input.json", output / "vision-native.json"
    source.write_text(json.dumps([str(p) for p in files]))
    subprocess.run([str(Path(sys.argv[1]).resolve()), "--base", str(ROOT), "--evaluate", str(source), str(target)], check=True)
    results = json.loads(target.read_text())
    mlp = MLPNumpy(ROOT / "modelo/landmark_model.npz")
    rows = []
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6) as hands:
        for path, native in zip(files, results):
            image = cv2.imread(str(path))
            detected = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            py_letter, rs_letter, distance = None, None, None
            if native["probabilities"] is not None:
                rs_letter = mlp.classes[int(np.argmax(native["probabilities"]))]
            if detected.multi_hand_landmarks:
                pts = mediapipe_landmarks_to_xyz(detected.multi_hand_landmarks[0])
                py_letter = mlp.classes[int(mlp.predict_proba(normalize_landmarks(pts)[None]).argmax())]
                if native["points"] is not None:
                    distance = float(np.abs(normalize_landmarks(pts)-normalize_landmarks(native["points"])).mean())
            rows.append({"label": path.parent.name, "python": py_letter, "rust": rs_letter, "feature_mae": distance})
    report = {"samples": len(rows), "python_detected": sum(r["python"] is not None for r in rows),
              "rust_detected": sum(r["rust"] is not None for r in rows),
              "python_label_matches": sum(r["python"] == r["label"] for r in rows),
              "rust_label_matches": sum(r["rust"] == r["label"] for r in rows), "rows": rows,
              "limitation": "Images from local training set; not an independent accuracy estimate."}
    (output / "vision-comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}))

if __name__ == "__main__":
    main()
