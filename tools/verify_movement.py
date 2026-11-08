"""Verify segmentation and compare Rust LSTM probabilities against Python."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classifier import LSTMNumpy
from movement import MotionSegmenter, sequence_features

def check_segmenter():
    shape = np.zeros((21, 3), dtype=np.float32)
    shape[:, 1] = np.linspace(0, .2, 21)
    segmenter = MotionSegmenter()
    segments = []
    for index in range(35):
        offset = min(max(index - 4, 0), 15) * 0.012
        finished, _ = segmenter.update(shape + [offset, 0, 0], index * .05)
        if finished is not None:
            segments.append(finished)
    assert len(segments) == 1
    features = sequence_features(segments[0], 65)
    assert features.shape == (20, 65)
    assert np.allclose(features[:, :63], features[0, :63])
    assert features[0, 63] == 0 and features[-1, 63] > .4
    assert segmenter.update(None, 2.0) == (None, False)
    completed = []
    for index in range(60):
        finished, _ = segmenter.update(shape + [index * .012, 0, 0], 3 + index * .05)
        if finished is not None:
            completed.append(finished)
    assert len(completed) == 1
    assert completed[0][-1][0] - completed[0][0][0] < 2.2

def main():
    check_segmenter()
    model = LSTMNumpy(ROOT / "modelo/movement_model.npz")
    output = ROOT / "test-output"
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    sequence = list(rng.normal(0, .2, (20, 20, model.feature_count)).astype(np.float32))
    if model.feature_count == 63:
        for letter in ["H", "J"]:
            sequence.extend(np.load(path, allow_pickle=False) for path in sorted((ROOT / "dataset/movement_sequences" / letter).glob("*.npy"))[-30:])
    source, target = output / "lstm-input.json", output / "lstm-rust.json"
    source.write_text(json.dumps(np.array(sequence).tolist()), encoding="utf-8")
    subprocess.run([str(Path(sys.argv[1]).resolve()), "--base", str(ROOT), "--predict-movement", str(source), str(target)], check=True)
    expected = np.array([model.predict_proba(sample) for sample in sequence])
    actual = np.array(json.loads(target.read_text(encoding="utf-8")))
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
    assert np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1))
    print(json.dumps({"samples": len(sequence), "max_absolute_error": float(np.max(np.abs(expected - actual))), "argmax_agreement": 1.0}))

if __name__ == "__main__":
    main()
