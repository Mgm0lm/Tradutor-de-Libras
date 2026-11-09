"""
Thread de inferência do tradutor de LIBRAS: extrai os 21 landmarks da
mão com o MediaPipe Hands (bem mais leve e robusto a fundo/iluminação
que um classificador de imagem treinado do zero) e classifica a letra
com uma MLP treinada por train_classifier.py.

Roda separada da thread de câmera (ver camera_manager.py), na
velocidade que o hardware permitir.
"""

import logging
import time
from threading import Lock

import cv2
import numpy as np
import mediapipe as mp
from PySide6.QtCore import QThread, Signal

import config
from landmarks_utils import mediapipe_landmarks_to_xyz, normalize_landmarks
from movement import MotionSegmenter, REJECT_LABEL, STEPS, TRAJECTORY_FEATURES, sequence_features

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 5

# Pares de índices de landmark que formam o "esqueleto" da mão, na
# mesma convenção do MediaPipe (usado só pra desenhar o overlay no
# vídeo em main.py - não influencia a classificação).
HAND_CONNECTIONS = tuple(mp.solutions.hands.HAND_CONNECTIONS)


class Detection:
    __slots__ = ("letter", "conf", "instant")

    def __init__(self, letter, conf, instant=False):
        self.letter = letter
        self.conf = conf
        # True para letras de MOVIMENTO (H/J/K/X/Z): o gesto inteiro já
        # foi observado quando a LSTM classifica, então não faz sentido
        # exigir "segurar parado" de novo como nas letras estáticas -
        # main.py pula o estabilizador e confirma na hora.
        self.instant = instant


class MLPNumpy:
    """Reimplementação mínima, em numpy puro, da inferência de uma
    MLPClassifier do scikit-learn (ReLU nas camadas ocultas, softmax
    na saída) - lê os pesos brutos exportados por train_classifier.py.
    Não depende de scikit-learn nem scipy em tempo de execução."""

    def __init__(self, npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            self.n_layers = int(data["n_layers"])
            self.classes = [str(c) for c in data["classes"]]
            self.weights = [data[f"W{i}"] for i in range(self.n_layers)]
            self.biases = [data[f"b{i}"] for i in range(self.n_layers)]

    def predict_proba(self, x):
        a = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ W + b
            is_last_layer = (i == self.n_layers - 1)
            if not is_last_layer:
                a = np.maximum(a, 0.0)  # ReLU
        # softmax na camada de saída
        a = a - a.max(axis=1, keepdims=True)
        exp = np.exp(a)
        return exp / exp.sum(axis=1, keepdims=True)


class TemporalMLPNumpy(MLPNumpy):
    """Apply the same MLP to one ordered, flattened gesture window."""

    feature_count = TRAJECTORY_FEATURES

    def __init__(self, npz_path):
        super().__init__(npz_path)
        with np.load(npz_path, allow_pickle=False) as data:
            if int(data["steps"]) != STEPS or int(data["feature_count"]) != self.feature_count:
                raise ValueError("Formato de sequência incompatível com a MLP temporal")
        allowed = {*config.MOVEMENT_LETTERS, REJECT_LABEL}
        if len(self.classes) < 2 or REJECT_LABEL not in self.classes or not set(self.classes) <= allowed:
            raise ValueError("Classes de movimento incompatíveis")
        width = STEPS * self.feature_count
        for weights, bias in zip(self.weights, self.biases):
            if bias.ndim != 1 or weights.shape != (width, bias.size):
                raise ValueError("Dimensões de pesos incompatíveis")
            width = bias.size
        if not self.weights or width != len(self.classes):
            raise ValueError("Saída da MLP temporal incompatível")

    def predict_proba(self, sequence):
        if sequence.shape != (STEPS, self.feature_count):
            raise ValueError("Formato de sequência incompatível com a MLP temporal")
        return super().predict_proba(sequence.reshape(1, -1))[0]


class LSTMNumpy:
    """Reimplementação mínima, em numpy puro, do forward pass de uma
    nn.LSTM de camada única do PyTorch (batch_first) + camada linear
    final - lê os pesos brutos exportados por
    train_movement_classifier.py. Não depende de PyTorch em tempo de
    execução.

    Segue exatamente a convenção de portões do PyTorch: as matrizes
    W_ih/W_hh (e os bias) vêm concatenadas na ordem
    [entrada, esquecimento, candidato, saída] (i, f, g, o)."""

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=False)
        self.W_ih = data["W_ih"]  # (4H, input_size)
        self.W_hh = data["W_hh"]  # (4H, H)
        self.b_ih = data["b_ih"]  # (4H,)
        self.b_hh = data["b_hh"]  # (4H,)
        self.W_fc = data["W_fc"]  # (num_classes, H)
        self.b_fc = data["b_fc"]  # (num_classes,)
        self.hidden_size = int(data["hidden_size"])
        self.feature_count = int(data.get("feature_count", self.W_ih.shape[1]))
        if self.feature_count not in (63, 65) or int(data.get("steps", STEPS)) != STEPS:
            raise ValueError("Formato de sequência incompatível")
        self.classes = [str(c) for c in data["classes"]]

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict_proba(self, sequence):
        """sequence: array (T, input_size) - a janela do gesto inteira."""
        H = self.hidden_size
        h = np.zeros(H, dtype=np.float32)
        c = np.zeros(H, dtype=np.float32)

        for t in range(sequence.shape[0]):
            x_t = sequence[t]
            gates = self.W_ih @ x_t + self.b_ih + self.W_hh @ h + self.b_hh
            i = self._sigmoid(gates[0:H])
            f = self._sigmoid(gates[H:2 * H])
            g = np.tanh(gates[2 * H:3 * H])
            o = self._sigmoid(gates[3 * H:4 * H])
            c = f * c + i * g
            h = o * np.tanh(c)

        logits = self.W_fc @ h + self.b_fc
        logits = logits - logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()


class LibrasClassifierWorker(QThread):
    # Detection ou None, frame usado (BGR np.ndarray), landmarks em
    # pixel (lista de 21 (x,y) int) ou None, frame_id
    detection_ready = Signal(object, np.ndarray, object, int)
    model_error = Signal(str)
    model_ready = Signal()

    def __init__(self, model_path, shared_frame, conf_threshold, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.shared_frame = shared_frame

        self._conf_lock = Lock()
        self._conf_threshold = conf_threshold
        self._running = False
        self._model = None
        self._labels = None
        self._hands = None

        # Estado do detector de movimento / classificador de gestos
        # (opcional - se o modelo não existir, o app funciona só com
        # as letras estáticas, sem erro).
        self._movement_model = None
        self._segmenter = MotionSegmenter()

    def set_conf_threshold(self, value):
        with self._conf_lock:
            self._conf_threshold = value

    def _get_conf_threshold(self):
        with self._conf_lock:
            return self._conf_threshold

    def run(self):
        logger.info("Thread de classificação (LIBRAS/MediaPipe) iniciada. Modelo: %s", self.model_path)

        if not self._load_model():
            return

        if config.selected_movement_model_path() is not None:
            self._load_movement_model()

        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            )
        except Exception:
            logger.exception("Falha ao iniciar o MediaPipe Hands")
            self.model_error.emit(
                "Erro ao iniciar o MediaPipe Hands.\n"
                "Detalhes completos foram salvos em app.log."
            )
            return

        logger.info("MediaPipe Hands + modelo carregados com sucesso. Classes: %s", self._labels)
        self.model_ready.emit()

        self._running = True
        last_frame_id = -1
        consecutive_errors = 0

        while self._running:
            frame, frame_id = self.shared_frame.get()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = frame_id

            try:
                detection, landmarks_px = self._process(frame)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.exception(
                    "Erro durante a inferência (tentativa %d/%d)",
                    consecutive_errors, MAX_CONSECUTIVE_ERRORS
                )
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    self.model_error.emit(
                        f"A detecção parou após vários erros seguidos de inferência:\n{e}\n\n"
                        "Detalhes completos foram salvos em app.log."
                    )
                    self._running = False
                    break
                time.sleep(0.3)
                continue

            self.detection_ready.emit(detection, frame, landmarks_px, frame_id)

        if self._hands is not None:
            self._hands.close()

    def _process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        now = time.monotonic()
        if not results.multi_hand_landmarks:
            self._segmenter.reset()
            return None, None

        xyz = mediapipe_landmarks_to_xyz(results.multi_hand_landmarks[0])
        landmarks_px = [(int(x * w), int(y * h)) for x, y, _z in xyz]
        if self._movement_model is not None:
            segment, active = self._segmenter.update(xyz, now)
            if segment is not None:
                return self._classify_gesture(segment), landmarks_px
            if active:
                return None, landmarks_px
        return self._classify_static(normalize_landmarks(xyz)), landmarks_px

    def _classify_static(self, feature):
        probs = self._model.predict_proba(feature.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        letter = self._labels[best_idx]
        conf_threshold = self._get_conf_threshold()
        if letter == REJECT_LABEL or conf < conf_threshold:
            return None
        return Detection(letter, conf, instant=False)

    def _classify_gesture(self, segment):
        sequence = sequence_features(segment, self._movement_model.feature_count)
        probs = self._movement_model.predict_proba(sequence)
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        letter = self._movement_model.classes[best_idx]
        if letter not in config.MOVEMENT_LETTERS or conf < max(config.MOVEMENT_CONF_THRESHOLD, self._get_conf_threshold()):
            return None
        return Detection(letter, conf, instant=True)

    def _load_model(self):
        try:
            self._model = MLPNumpy(self.model_path)
            self._labels = self._model.classes
            return True
        except FileNotFoundError:
            logger.error("Modelo não encontrado em %s", self.model_path)
            self.model_error.emit(
                f"Modelo não encontrado em {self.model_path}.\n"
                "Rode primeiro: python collect_landmarks.py (colete amostras)\n"
                "e depois: python train_classifier.py (treina o modelo)."
            )
            return False
        except Exception:
            logger.exception("Falha ao carregar o modelo (%s)", self.model_path)
            self.model_error.emit(
                f"Erro ao carregar o modelo ({self.model_path}).\n"
                "Detalhes completos foram salvos em app.log."
            )
            return False

    def _load_movement_model(self):
        path = config.selected_movement_model_path()
        model_class = TemporalMLPNumpy if config.ENABLE_MOVEMENT_MLP else LSTMNumpy
        train_command = "train_classifier.py --movement" if config.ENABLE_MOVEMENT_MLP else "train_movement_classifier.py"
        try:
            self._movement_model = model_class(path)
            logger.info(
                "Modelo de gestos %s carregado. Classes: %s",
                model_class.__name__,
                self._movement_model.classes,
            )
        except FileNotFoundError:
            logger.info(
                "Modelo de gestos não encontrado em %s - app funcionará só com "
                "letras estáticas. Rode collect_landmarks.py --movement e %s.",
                path,
                train_command,
            )
            self._movement_model = None
        except Exception:
            logger.exception("Falha ao carregar o modelo de gestos (%s) - ignorando.", path)
            self._movement_model = None

    def stop(self):
        self._running = False
