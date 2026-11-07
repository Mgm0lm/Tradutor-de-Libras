"""
Enumeracao de cameras disponiveis (USB / integrada) e thread de
captura continua de video, independente da thread de inferencia.

(Mesma estrutura usada no projeto do caixa com visao computacional -
funciona bem aqui pelo mesmo motivo: mantem o video fluido mesmo se a
inferencia do CNN demorar mais que 1 frame de camera.)
"""

import platform
import time
from threading import Lock

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


def _backend():
    if platform.system() == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    return None


def list_available_cameras(max_scan=5):
    """Tenta abrir os indices 0..max_scan-1 e retorna os que respondem.

    OBS: o OpenCV nao expoe de forma confiavel e multiplataforma se a
    camera e USB ou integrada, entao os rotulos sao genericos. Teste
    antes da apresentacao para saber qual indice e qual.
    """
    backend = _backend()
    cameras = []
    for idx in range(max_scan):
        cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                cameras.append((idx, f"Câmera {idx}"))
        if cap is not None:
            cap.release()

    if not cameras:
        cameras.append((0, "Câmera 0 (padrão)"))
    return cameras


class SharedFrame:
    """Guarda o frame mais recente da camera de forma thread-safe, para
    que a thread de inferencia sempre consuma o frame mais atual (sem
    fila / sem acumular atraso)."""

    def __init__(self):
        self._lock = Lock()
        self._frame = None
        self._frame_id = 0

    def set(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def get(self):
        with self._lock:
            if self._frame is None:
                return None, -1
            return self._frame.copy(), self._frame_id

    def clear(self):
        with self._lock:
            self._frame = None
            self._frame_id = 0


class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray, int)   # frame BGR, frame_id
    camera_error = Signal(str)

    def __init__(self, camera_index, target_fps, width, height, shared_frame: SharedFrame, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.target_fps = max(1, target_fps)
        self.width = width
        self.height = height
        self.shared_frame = shared_frame
        self._running = False

    def run(self):
        self._running = True
        cap = self._open(self.camera_index)
        if cap is None or not cap.isOpened():
            self.camera_error.emit(f"Não foi possível abrir a câmera {self.camera_index}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        min_frame_time = 1.0 / self.target_fps

        while self._running:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            # Espelha horizontalmente (efeito espelho) - mais natural pra
            # quem esta de frente pra camera sinalizando, como olhar num
            # espelho em vez de ver a propria mao invertida.
            frame = cv2.flip(frame, 1)

            self.shared_frame.set(frame)
            self.frame_ready.emit(frame, self.shared_frame._frame_id)

            elapsed = time.time() - t0
            sleep_time = min_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    def _open(self, index):
        backend = _backend()
        return cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)

    def stop(self):
        self._running = False
