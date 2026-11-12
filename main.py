"""
Tradutor de LIBRAS (alfabeto) - webcam de um lado, letra reconhecida
(com fala via TTS) do outro.
"""

import logging
import platform
import sys
import time

# --------------------------------------------------------------
# FIX DE DPI NO WINDOWS - precisa rodar ANTES de criar a QApplication,
# senao a janela fullscreen e os widgets ficam com tamanho/posicao
# errados em monitores com escala >100% (comum em notebooks).
# --------------------------------------------------------------
if platform.system() == "Windows":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# --------------------------------------------------------------
# FIX shiboken6 (PySide6) + six (mediapipe -> matplotlib -> dateutil
# -> six): no .exe gerado pelo PyInstaller, o próprio bootstrap do
# PyInstaller já toca PySide6/shiboken ANTES deste arquivo começar a
# rodar - então importar mediapipe "antes do PySide6" aqui não
# resolve mais (só ajudava rodando via "python main.py" direto).
#
# A causa raiz: o shiboken instala um hook que inspeciona
# (inspect.getsource) todo módulo importado no processo, pra suportar
# o "feature switching" do PySide. O carregador preguiçoso de módulos
# do six (usado por dateutil, puxado pelo matplotlib, puxado pelo
# mediapipe) não implementa um atributo que o mecanismo de repr de
# módulo do Python 3.12 espera - e isso quebra bem no meio da
# formatação da mensagem de um TypeError comum e esperado (inspect
# não consegue pegar código-fonte de módulo embutido), virando:
#   AttributeError: '_SixMetaPathImporter' object has no attribute '_path'
#
# Solução: substituir inspect.getsource por uma versão que nunca deixa
# essa exceção escapar - qualquer falha vira um OSError "comum" (o
# tipo de erro que o inspect já usa pra "sem código-fonte disponível",
# e que o shiboken já trata normalmente pra módulos embutidos/
# compilados, sem crashar). Como o Python resolve inspect.getsource
# por nome a cada chamada (não guarda uma referência fixa), isso
# funciona não importa quando é aplicado - inclusive no .exe.
import inspect as _inspect

_original_getsource = _inspect.getsource


def _safe_getsource(obj):
    try:
        return _original_getsource(obj)
    except Exception as e:
        raise OSError(f"código-fonte indisponível ({e})") from None


_inspect.getsource = _safe_getsource

try:
    import mediapipe  # noqa: F401  (ajuda no modo "python main.py"; inofensivo no .exe)
except Exception:
    pass  # o erro normal aparecerá no status da UI, via classifier.py

try:
    # Pré-importa pygame na thread principal, ANTES de qualquer QThread
    # começar (TTSWorker importa pygame só quando roda, dentro da sua
    # própria thread). Import de um módulo pela primeira vez a partir
    # de uma thread secundária, enquanto o processo ainda está sendo
    # inicializado (comum logo na abertura do .exe gerado pelo
    # PyInstaller), pode disparar um deadlock no lock interno de import
    # do Python (_ModuleLock). Importar aqui, na thread principal e
    # antes de tudo, deixa o pygame em cache - o "import pygame" feito
    # depois dentro da TTSWorker só reaproveita o módulo já carregado.
    import pygame  # noqa: F401
except Exception:
    pass  # erro tratado normalmente dentro de tts.py

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFrame, QComboBox, QCheckBox, QPushButton, QProgressBar, QSizePolicy,
)

import cv2
import numpy as np

import config
from camera_manager import CameraWorker, SharedFrame, list_available_cameras
from classifier import LibrasClassifierWorker, HAND_CONNECTIONS
from stabilizer import LetterStabilizer
from tts import TTSWorker
from ui_styles import STYLESHEET

# Tempos em segundos. O config_override.json tem prioridade sobre estes valores.
# Estabilidade da previsão por quadro antes da confirmação.
MOVEMENT_MIN_STABLE_SECONDS = {"H": 0.1, "J": 0.05, "K": 0.1, "X": 0, "Z": 0}
# Espera do TTS após a confirmação. Use 0 para falar imediatamente.
MOVEMENT_TTS_DELAY_SECONDS = {"H": 0.5, "J": 0.0, "K": 0.5, "X": 0.4, "Z": 0.75}
# Tempo que X e Z ficam na tela após a confirmação.
XZ_DISPLAY_HOLD_SECONDS = 1.0


def _nonnegative_seconds(value, default):
    if type(value) not in (int, float) or not -sys.float_info.max <= value <= sys.float_info.max:
        value = default
    return max(0.0, value)


def _movement_seconds(defaults, override, *, preserve_j=False):
    # O formato antigo aceitava um único número e mantinha J sem espera no TTS.
    if not isinstance(override, dict):
        override = {letter: override for letter in defaults if letter != "J" or not preserve_j}
    return {letter: _nonnegative_seconds(override.get(letter), default)
            for letter, default in defaults.items()}


logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_TITLE)
        self.setStyleSheet(STYLESHEET)

        self.shared_frame = SharedFrame()
        self.camera_worker = None
        self.classifier_worker = None
        self.stabilizer = LetterStabilizer(_movement_seconds(
            MOVEMENT_MIN_STABLE_SECONDS, getattr(config, "MOVEMENT_MIN_STABLE_SECONDS", None)
        ))
        self._movement_tts_delay = _movement_seconds(
            MOVEMENT_TTS_DELAY_SECONDS, getattr(config, "MOVEMENT_TTS_DELAY_SECONDS", None),
            preserve_j=True,
        )
        self._xz_display_hold = _nonnegative_seconds(
            getattr(config, "XZ_DISPLAY_HOLD_SECONDS", None), XZ_DISPLAY_HOLD_SECONDS
        )
        self._movement_visible_until = 0.0
        self.tts_worker = None
        self._pending_tts_letter = None
        self._tts_delay_timer = QTimer(self)
        self._tts_delay_timer.setSingleShot(True)
        self._tts_delay_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._tts_delay_timer.timeout.connect(self._play_delayed_letter)

        self._last_camera_frame_id = -1
        self._last_landmarks_px = None
        self.stabilizer_confirmed = False
        self._last_shown_qimage = None
        self._cam_fps_count = 0
        self._ai_fps_count = 0

        self._build_ui()
        if config.SHOW_FPS:
            self._fps_last_t = time.monotonic()
            self._fps_timer = QTimer(self)
            self._fps_timer.timeout.connect(self._refresh_fps)
            self._fps_timer.start(1000)
        self._start_tts()
        self._start_camera(0)
        self._start_classifier()

        self.showFullScreen()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        outer = QHBoxLayout(root)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(24)

        outer.addLayout(self._build_left_panel(), stretch=6)
        outer.addLayout(self._build_right_panel(), stretch=4)

    def _build_left_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

        self.video_label = QLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.fps_label = QLabel("Câmera: 0.0 fps   |   IA: 0.0 fps")
        self.fps_label.setObjectName("subtitulo")
        self.fps_label.setVisible(config.SHOW_FPS)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.video_label, stretch=1)

        controls = QFrame()
        controls.setObjectName("card")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(14, 10, 14, 10)

        controls_layout.addWidget(QLabel("Câmera:"))
        self.camera_combo = QComboBox()
        for idx, name in list_available_cameras(config.MAX_CAMERAS_TO_SCAN):
            self.camera_combo.addItem(name, idx)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        controls_layout.addWidget(self.camera_combo)

        controls_layout.addStretch(1)

        self.desync_checkbox = QCheckBox("Dessincronizar câmera / IA")
        self.desync_checkbox.setChecked(config.DESYNC_INFERENCE)
        self.desync_checkbox.setToolTip(
            "Ligado: vídeo sempre fluido, a letra pode atrasar um pouco.\n"
            "Desligado: vídeo só atualiza junto com cada nova inferência."
        )
        controls_layout.addWidget(self.desync_checkbox)

        controls_layout.addStretch(1)

        controls_layout.addWidget(QLabel("Confiança mín.:"))
        self.threshold_minus_btn = QPushButton("-")
        self.threshold_minus_btn.setObjectName("thresholdBtn")
        self.threshold_minus_btn.clicked.connect(lambda: self._adjust_threshold(-config.CONF_THRESHOLD_STEP))
        controls_layout.addWidget(self.threshold_minus_btn)

        self.threshold_label = QLabel(f"{config.CONF_THRESHOLD_DEFAULT:.2f}")
        controls_layout.addWidget(self.threshold_label)

        self.threshold_plus_btn = QPushButton("+")
        self.threshold_plus_btn.setObjectName("thresholdBtn")
        self.threshold_plus_btn.clicked.connect(lambda: self._adjust_threshold(config.CONF_THRESHOLD_STEP))
        controls_layout.addWidget(self.threshold_plus_btn)

        layout.addWidget(controls)

        self._conf_threshold = config.CONF_THRESHOLD_DEFAULT
        return layout

    def _build_right_panel(self):
        layout = QVBoxLayout()
        layout.setSpacing(16)

        title = QLabel("Letra reconhecida")
        title.setObjectName("titulo")
        layout.addWidget(title)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignCenter)
        card_layout.setContentsMargins(20, 20, 20, 20)

        self.letra_label = QLabel("–")
        self.letra_label.setObjectName("letraLabel")
        self.letra_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.letra_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        card_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Aguardando modelo...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.status_label)

        layout.addWidget(card, stretch=1)

        bottom_row = QHBoxLayout()
        self.tts_checkbox = QCheckBox("Falar em voz alta (TTS)")
        self.tts_checkbox.setChecked(config.ENABLE_TTS)
        self.tts_checkbox.stateChanged.connect(self._on_tts_toggled)
        bottom_row.addWidget(self.tts_checkbox)

        bottom_row.addStretch(1)

        self.repeat_btn = QPushButton("Repetir")
        self.repeat_btn.setObjectName("secondaryBtn")
        self.repeat_btn.clicked.connect(self._repeat_last_letter)
        bottom_row.addWidget(self.repeat_btn)

        exit_btn = QPushButton("Sair (ESC)")
        exit_btn.setObjectName("secondaryBtn")
        exit_btn.clicked.connect(self.close)
        bottom_row.addWidget(exit_btn)

        layout.addLayout(bottom_row)
        return layout

    # ------------------------------------------------------------------
    # Câmera
    # ------------------------------------------------------------------
    def _start_camera(self, camera_index):
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker.wait(1000)

        self.shared_frame.clear()
        self._movement_visible_until = 0.0
        self.camera_worker = CameraWorker(
            camera_index=camera_index,
            target_fps=config.CAMERA_FPS_TARGET,
            width=config.CAMERA_FRAME_WIDTH,
            height=config.CAMERA_FRAME_HEIGHT,
            shared_frame=self.shared_frame,
        )
        self.camera_worker.frame_ready.connect(self._on_camera_frame)
        self.camera_worker.camera_error.connect(self._on_camera_error)
        self.camera_worker.start()

    def _on_camera_changed(self, combo_index):
        camera_index = self.camera_combo.itemData(combo_index)
        self._start_camera(camera_index)

    def _on_camera_error(self, message):
        self.status_label.setText(message)

    def _on_camera_frame(self, frame, frame_id):
        self._last_camera_frame_id = frame_id
        self._cam_fps_count += 1
        if self.desync_checkbox.isChecked():
            self._show_frame(frame, self._last_landmarks_px)

    # ------------------------------------------------------------------
    # Classificador (CNN)
    # ------------------------------------------------------------------
    def _start_classifier(self):
        self.classifier_worker = LibrasClassifierWorker(
            model_path=config.MODEL_PATH,
            shared_frame=self.shared_frame,
            conf_threshold=self._conf_threshold,
        )
        self.classifier_worker.detection_ready.connect(self._on_detection)
        self.classifier_worker.model_error.connect(self._on_model_error)
        self.classifier_worker.model_ready.connect(self._on_model_ready)
        self.classifier_worker.start()

    def _on_model_ready(self):
        self.status_label.setText("Posicione a mão dentro da área guia.")

    def _on_model_error(self, message):
        first_line = message.strip().splitlines()[0] if message.strip() else "Erro no modelo"
        self.status_label.setText(f"{first_line} (veja app.log)")
        logger.error(message)

    def _adjust_threshold(self, delta):
        self._conf_threshold = min(
            config.CONF_THRESHOLD_MAX,
            max(config.CONF_THRESHOLD_MIN, self._conf_threshold + delta),
        )
        self.threshold_label.setText(f"{self._conf_threshold:.2f}")
        if self.classifier_worker is not None:
            self.classifier_worker.set_conf_threshold(self._conf_threshold)

    def _on_detection(self, detection, frame, landmarks_px, frame_id):
        self._ai_fps_count += 1
        self._last_landmarks_px = landmarks_px

        if not self.desync_checkbox.isChecked():
            self._show_frame(frame, landmarks_px)

        now = time.monotonic()
        if now < self._movement_visible_until and self._last_confirmed_letter in ("X", "Z"):
            return

        if detection is not None and detection.instant:
            # Letra de MOVIMENTO (H/J/K/X/Z): o gesto inteiro já foi
            # observado pela LSTM - não faz sentido pedir pra "segurar
            # parado" de novo como as estáticas. Confirma direto,
            # respeitando o mesmo cooldown do estabilizador pra não
            # falar a mesma letra em sequência por engano.
            if self.stabilizer.confirm(detection.letter, now):
                self.stabilizer.reset()
                self._update_letter_ui(detection.letter, 1.0, confirmed=True)
                self._finish_confirmation(detection.letter, now, sequence_hold=0.65)
            return

        if now < self._movement_visible_until:
            return

        letter = detection.letter if detection is not None else None
        shown_letter, progress, just_confirmed = self.stabilizer.update(letter)

        self._update_letter_ui(shown_letter, progress, confirmed=(progress >= 1.0 and shown_letter is not None))

        if just_confirmed:
            self._finish_confirmation(shown_letter, now)

    def _finish_confirmation(self, letter, now, sequence_hold=0.0):
        hold = self._xz_display_hold if letter in ("X", "Z") else sequence_hold
        self._movement_visible_until = now + hold
        self._speak_confirmed(letter)

    def _update_letter_ui(self, letter, progress, confirmed):
        self.progress_bar.setValue(int(progress * 100))
        self.stabilizer_confirmed = confirmed

        if letter is None:
            self.letra_label.setText("–")
            self.letra_label.setProperty("estado", "nada")
            self.status_label.setText("Nenhuma mão reconhecida na área guia.")
        else:
            self.letra_label.setText(letter)
            if confirmed:
                self.letra_label.setProperty("estado", "confirmado")
                self.status_label.setText(f"Confirmado: {letter}")
                self._last_confirmed_letter = letter
            else:
                self.letra_label.setProperty("estado", "detectando")
                self.status_label.setText(f"Detectando: {letter}...")

        # forca o Qt a reaplicar o QSS quando a propriedade dinamica muda
        self.letra_label.style().unpolish(self.letra_label)
        self.letra_label.style().polish(self.letra_label)

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------
    def _start_tts(self):
        self._last_confirmed_letter = None
        letters = []
        for path in filter(None, (config.MODEL_PATH, config.selected_movement_model_path())):
            try:
                with np.load(path, allow_pickle=False) as data:
                    letters.extend(str(letter) for letter in data["classes"] if letter in config.LETTERS.values())
            except (OSError, ValueError, KeyError):
                logger.warning("Modelo indisponível para preparar áudio: %s", path)
        self.tts_worker = TTSWorker(letters=list(dict.fromkeys(letters)))
        self.tts_worker.cache_ready.connect(lambda: logger.info("Cache de TTS pronto."))
        self.tts_worker.speak_error.connect(lambda msg: logger.warning("TTS: %s", msg))
        self.tts_worker.start()

    def _on_tts_toggled(self, state):
        config.ENABLE_TTS = bool(state)
        if not config.ENABLE_TTS and self.tts_worker is not None:
            self._cancel_delayed_letter()
            self.tts_worker.clear_pending()

    def _cancel_delayed_letter(self):
        self._tts_delay_timer.stop()
        self._pending_tts_letter = None

    def _play_delayed_letter(self):
        letter, self._pending_tts_letter = self._pending_tts_letter, None
        if letter is not None and self.tts_worker is not None and config.ENABLE_TTS:
            self.tts_worker.enqueue(letter)

    def _speak_confirmed(self, letter):
        self._cancel_delayed_letter()
        if self.tts_worker is None or not config.ENABLE_TTS:
            return
        delay_ms = round(min(self._movement_tts_delay.get(letter, 0) * 1000, 2_147_483_647))
        if delay_ms:
            self.tts_worker.clear_pending()
            self._pending_tts_letter = letter
            self._tts_delay_timer.start(delay_ms)
        else:
            self.tts_worker.enqueue(letter)

    def _repeat_last_letter(self):
        if self._last_confirmed_letter and self.tts_worker is not None:
            self._cancel_delayed_letter()
            self.tts_worker.enqueue(self._last_confirmed_letter)

    # ------------------------------------------------------------------
    # Vídeo / ROI overlay
    # ------------------------------------------------------------------
    def _show_frame(self, frame_bgr, landmarks_px=None):
        frame = frame_bgr.copy()
        h, w = frame.shape[:2]

        if landmarks_px:
            color = (80, 220, 80) if self.stabilizer_confirmed else (235, 130, 40)  # BGR
            for i1, i2 in HAND_CONNECTIONS:
                if i1 < len(landmarks_px) and i2 < len(landmarks_px):
                    cv2.line(frame, landmarks_px[i1], landmarks_px[i2], color, 2)
            for x, y in landmarks_px:
                cv2.circle(frame, (x, y), 4, color, -1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # FPS
    # ------------------------------------------------------------------
    def _refresh_fps(self):
        now = time.monotonic()
        elapsed = now - self._fps_last_t
        self.fps_label.setText(
            f"Câmera: {self._cam_fps_count / elapsed:.1f} fps   |   IA: {self._ai_fps_count / elapsed:.1f} fps"
        )
        self._cam_fps_count = 0
        self._ai_fps_count = 0
        self._fps_last_t = now

    # ------------------------------------------------------------------
    # Encerramento
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._cancel_delayed_letter()
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker.wait(1000)
        if self.classifier_worker is not None:
            self.classifier_worker.stop()
            self.classifier_worker.wait(1000)
        if self.tts_worker is not None:
            self.tts_worker.stop()
            self.tts_worker.wait(1000)
        super().closeEvent(event)


def main():
    if "--movement" in sys.argv and "--movement-mlp" in sys.argv:
        raise SystemExit("Escolha --movement (LSTM) ou --movement-mlp (MLP temporal).")
    if "--movement-mlp" in sys.argv:
        sys.argv.remove("--movement-mlp")
        config.ENABLE_MOVEMENT_MLP = True
    if "--movement" in sys.argv:
        sys.argv.remove("--movement")
        config.ENABLE_MOVEMENT = True
        config.ENABLE_MOVEMENT_MLP = False
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
