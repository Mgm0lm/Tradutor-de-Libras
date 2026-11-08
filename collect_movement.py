"""Collect complete gestures with wrist trajectory using the live segmenter."""
import argparse
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

import config
from landmarks_utils import mediapipe_landmarks_to_xyz
from movement import MotionSegmenter, sequence_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    labels = [*config.MOVEMENT_LETTERS, '_']
    target = 0
    segmenter = MotionSegmenter()
    segment = None
    completed_at = 0.0
    root = Path(config.MOVEMENT_TRAJECTORY_DIR)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(f'Câmera {args.camera} indisponível')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
    print('A/D: letra; ESPAÇO: salvar gesto concluído; ESC: sair. _ = movimento que não é letra.')
    try:
        with mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError('A câmera parou de fornecer quadros.')
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                now = time.monotonic()
                xyz = None
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                    xyz = mediapipe_landmarks_to_xyz(hand)
                finished, active = segmenter.update(xyz, now)
                if active:
                    segment = None
                if finished is not None:
                    segment, completed_at = finished, now
                if now - completed_at > 4:
                    segment = None
                label = labels[target]
                folder = root / label
                count = len(list(folder.glob('*.npy')))
                state = 'gravando' if active else ('ESPACO para salvar' if segment is not None else 'aguardando gesto')
                cv2.putText(frame, f'{label}  |  {count} amostras  |  {state}', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 2)
                cv2.imshow('Coleta de movimento Libras', frame)
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    break
                if key in (ord('a'), ord('d')):
                    target = (target + (1 if key == ord('d') else -1)) % len(labels)
                    segmenter.reset()
                    segment = None
                if key == 32 and segment is not None:
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f'sample_{time.time_ns()}.npy'
                    np.save(path, sequence_features(segment, 65), allow_pickle=False)
                    segment = None
                    print(f'Amostra de {label} salva: {path.name}')
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
