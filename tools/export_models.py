"""Export existing NPZ weights to portable JSON without changing their values."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def export(source, destination):
    with np.load(source, allow_pickle=False) as data:
        model = {key: data[key].tolist() for key in data.files}
    destination.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print(destination)

if __name__ == "__main__":
    for name in ("landmark_model", "movement_model"):
        source = ROOT / "modelo" / (name + ".npz")
        if source.exists():
            export(source, source.with_suffix(".json"))
