"""Compare native MLP against the actual Python classifier on local data."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classifier import MLPNumpy

def main():
    exe = Path(sys.argv[1]).resolve()
    output = ROOT / "test-output"
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(2026)
    features = rng.uniform(-1, 1, (100, 63)).astype(np.float32)
    data = ROOT / "dataset/landmarks.csv"
    if data.exists():
        import csv
        with data.open(encoding="utf-8") as stream:
            rows = list(csv.reader(stream))[1:]
        real = np.asarray([[float(v) for v in row[:-1]] for row in rows[:300]], dtype=np.float32)
        features = np.concatenate([features, real])
    source, target = output / "features.json", output / "native-probabilities.json"
    source.write_text(json.dumps(features.tolist()))
    subprocess.run([str(exe), "--base", str(ROOT), "--predict", str(source), str(target)], check=True)
    expected = MLPNumpy(ROOT / "modelo/landmark_model.npz").predict_proba(features)
    actual = np.asarray(json.loads(target.read_text()))
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
    assert np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
    report = {"samples": len(features), "max_absolute_error": float(np.abs(actual-expected).max()), "argmax_agreement": 1.0}
    (output / "mlp-parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))

if __name__ == "__main__":
    main()
