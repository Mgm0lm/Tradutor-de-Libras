"""Collect gesture landmarks while Space is held, then save with S."""
import argparse
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QSizePolicy, QVBoxLayout, QWidget

import config
from landmarks_utils import mediapipe_landmarks_to_xyz
from movement import MAX_GAP, MIN_SPAN, REJECT_LABEL, TRAJECTORY_FEATURES, landmark_sample, sequence_features


class MovementCollector(QWidget):
    def __init__(self, camera, hands, root):
        super().__init__()
        self.camera = camera
        self.hands = hands
        self.root = root
        self.labels = [*config.MOVEMENT_LETTERS, REJECT_LABEL]
        self.target = 0
        self.samples = None
        self.interrupted = False
        self.draft = None

        self.setWindowTitle("Coleta de movimento — Libras")
        self.resize(1000, 650)
        self.heading = QLabel()
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(480, 270)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview.setStyleSheet("background: #161616")
        self.status = QLabel("Segure Espaço para gravar um gesto.")
        self.status.setWordWrap(True)
        controls = QLabel("A/D ou setas: letra   |   Espaço: segure para gravar   |   S: salvar   |   Esc: sair")
        controls.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.heading)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.status)
        layout.addWidget(controls)
        self._update_heading()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._read_frame)
        self.timer.start(15)

    def _update_heading(self):
        label = self.labels[self.target]
        count = sum(1 for _ in (self.root / label).glob("*.npy"))
        name = "_ (outros movimentos)" if label == REJECT_LABEL else label
        self.heading.setText(f"Letra alvo: {name}   |   Amostras salvas: {count}")

    def _read_frame(self):
        ok, frame = self.camera.read()
        if not ok:
            self.timer.stop()
            self.samples = None
            self.status.setText("A câmera parou de fornecer quadros. Reabra o coletor para gravar.")
            return
        now = time.monotonic()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.samples is not None:
                if self.samples and now - self.samples[-1][0] > MAX_GAP:
                    self.interrupted = True
                    self.status.setText("Rastreamento interrompido. Solte Espaço e grave novamente.")
                if not self.interrupted:
                    self.samples.append(landmark_sample(mediapipe_landmarks_to_xyz(hand), now))

        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
        ))

    def keyPressEvent(self, event):
        key = event.key()
        if key not in (Qt.Key.Key_Space, Qt.Key.Key_S, Qt.Key.Key_A, Qt.Key.Key_D,
                       Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Escape):
            super().keyPressEvent(event)
            return
        event.accept()
        if event.isAutoRepeat():
            return
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_Space and self.samples is None and self.timer.isActive():
            self.samples = []
            self.interrupted = False
            self.status.setText("Gravando. Solte Espaço para parar.")
        elif key == Qt.Key.Key_S:
            self._save_draft()
        elif self.samples is None and key in (Qt.Key.Key_A, Qt.Key.Key_D, Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = -1 if key in (Qt.Key.Key_A, Qt.Key.Key_Left) else 1
            self.target = (self.target + step) % len(self.labels)
            self._update_heading()

    def keyReleaseEvent(self, event):
        if event.key() != Qt.Key.Key_Space:
            super().keyReleaseEvent(event)
            return
        event.accept()
        if event.isAutoRepeat() or self.samples is None:
            return
        samples, self.samples = self.samples, None
        if self.interrupted or (samples and time.monotonic() - samples[-1][0] > MAX_GAP):
            self.status.setText("Rastreamento interrompido. Grave novamente.")
        elif len(samples) < 4 or samples[-1][0] - samples[0][0] < MIN_SPAN:
            self.status.setText("Captura muito curta ou sem mão detectada. Grave novamente.")
        else:
            label = self.labels[self.target]
            self.draft = label, sequence_features(samples, TRAJECTORY_FEATURES)
            self.status.setText(f"Gesto de {label} pronto. Pressione S para salvar.")
            return
        if self.draft is not None:
            self.status.setText(f"{self.status.text()} S ainda salva a captura anterior ({self.draft[0]}).")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.samples is not None and not self.isActiveWindow():
            self.samples = None
            self.status.setText("Gravação cancelada ao sair da janela.")

    def _save_draft(self):
        if self.samples is not None or self.draft is None:
            return
        label, sequence = self.draft
        folder = self.root / label
        path = folder / f"sample_{time.time_ns()}.npy"
        temporary = path.with_suffix(".npy.part")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with temporary.open("xb") as stream:
                np.save(stream, sequence, allow_pickle=False)
            temporary.replace(path)
        except OSError as error:
            self.status.setText(f"Erro ao salvar {label}. Pressione S para tentar de novo. {error}")
            return
        self.draft = None
        self._update_heading()
        self.status.setText(f"Amostra de {label} salva. Segure Espaço para gravar outra.")
        print(f"Amostra de {label} salva: {path.name}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    collect(parser.parse_args().camera)


def collect(camera_index):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    try:
        if not camera.isOpened():
            raise SystemExit(f"Câmera {camera_index} indisponível")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
        app = QApplication([])
        with mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
            window = MovementCollector(camera, hands, Path(config.MOVEMENT_TRAJECTORY_DIR))
            window.show()
            app.exec()
    finally:
        camera.release()


if __name__ == "__main__":
    main()
