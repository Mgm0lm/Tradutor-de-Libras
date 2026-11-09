"""Segment gestures in real time and retain the wrist path for new models."""
from collections import deque

import numpy as np

from landmarks_utils import normalize_landmarks, resample_sequence

STEPS = 20
TRAJECTORY_FEATURES = 65
REJECT_LABEL = "_"
MIN_SPAN = 0.25
MAX_SPAN = 2.0
QUIET = 0.18
PRE_ROLL = 0.12
SPEED_THRESHOLD = 0.65
MAX_GAP = 0.35


def load_samples(directory, labels, features):
    """Read one complete gesture per NPY file, in capture order within a class."""
    data = {}
    for letter in labels:
        sequences = []
        for path in sorted((directory / letter).glob("*.npy")):
            sample = np.load(path, allow_pickle=False)
            if sample.shape != (STEPS, features) or not np.isfinite(sample).all():
                raise ValueError(f"Amostra inválida: {path}. Esperado: {STEPS} × {features} valores finitos.")
            sequences.append(sample.astype(np.float32))
        if sequences:
            data[letter] = sequences
    return data


def sequence_features(samples, features):
    """Samples contain timestamp, normalized pose, wrist xy and hand scale."""
    times = np.array([s[0] for s in samples])
    poses = np.array([s[1] for s in samples], dtype=np.float32)
    if features == 63:
        return resample_sequence(poses, STEPS)
    wrists = np.array([s[2] for s in samples], dtype=np.float32)
    scale = max(1e-6, float(np.mean([s[3] for s in samples])))
    values = np.column_stack((poses, (wrists - wrists[0]) / scale))
    uniform = np.linspace(times[0], times[-1], STEPS)
    return np.column_stack([np.interp(uniform, times, column) for column in values.T]).astype(np.float32)


def landmark_sample(xyz, now):
    points = np.asarray(xyz, dtype=np.float32)
    scale = max(1e-6, float(np.linalg.norm(points - points[0], axis=1).max()))
    return now, normalize_landmarks(points), points[0, :2].copy(), scale


class MotionSegmenter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.recent = deque()
        self.active = []
        self.started = None
        self.last_motion = None
        self.speed = 0.0

    def update(self, xyz, now):
        """Return a finished segment and whether a gesture is in progress."""
        if xyz is None:
            self.reset()
            return None, False
        current = landmark_sample(xyz, now)
        moving = False
        if self.recent:
            previous = self.recent[-1]
            dt = now - previous[0]
            if dt > MAX_GAP:
                self.reset()
            elif dt > 0:
                shape_change = float(np.linalg.norm(current[1] - previous[1])) / np.sqrt(21)
                wrist_change = float(np.linalg.norm(current[2] - previous[2])) / max(current[3], previous[3])
                speed = max(shape_change, wrist_change) / max(0.025, dt)
                self.speed = 0.55 * self.speed + 0.45 * speed
                moving = self.speed >= SPEED_THRESHOLD
        self.recent.append(current)
        while self.recent and now - self.recent[0][0] > PRE_ROLL:
            self.recent.popleft()

        if self.started is None:
            if not moving:
                return None, False
            self.started = now
            self.last_motion = now
            self.active = list(self.recent)
        else:
            self.active.append(current)
            if moving:
                self.last_motion = now

        if now - self.started < MAX_SPAN and now - self.last_motion < QUIET:
            return None, True
        # Preserve the start of the window and omit excess stationary tail.
        span = self.last_motion - self.started
        result = self.active if span >= MIN_SPAN and len(self.active) >= 4 else None
        self.started = None
        self.last_motion = None
        self.active = []
        return result, False
