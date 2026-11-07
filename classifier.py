"""
Thread de inferência do tradutor de LIBRAS: extrai os 21 landmarks da
mão com o MediaPipe Hands (bem mais leve e robusto a fundo/iluminação
que um classificador de imagem treinado do zero) e classifica a letra
com um modelo leve (RandomForest, treinado por train_classifier.py).

Roda separada da thread de câmera (ver camera_manager.py), na
velocidade que o hardware permitir.
"""

import logging
import time
from collections import deque
from threading import Lock

import cv2
import numpy as np
import mediapipe as mp
from PySide6.QtCore import QThread, Signal

import config
from landmarks_utils import mediapipe_landmarks_to_xyz, normalize_landmarks, resample_sequence

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
        data = np.load(npz_path, allow_pickle=False)
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
        self._motion_buffer = deque(maxlen=config.MOTION_DETECT_FRAMES)
        self._gesture_window = deque()  # lista de (timestamp, feature) - poda por tempo, não por contagem

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

        self._load_movement_model()  # opcional - não falha o app se ausente

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
        now = time.time()

        if not results.multi_hand_landmarks:
            # Sem mão na cena - reseta os buffers de gesto (um gesto
            # precisa de mão contínua; se sumiu, o que tiver no buffer
            # não é mais um gesto válido).
            self._motion_buffer.clear()
            self._gesture_window.clear()
            return None, None

        hand_landmarks = results.multi_hand_landmarks[0]
        xyz = mediapipe_landmarks_to_xyz(hand_landmarks)
        landmarks_px = [(int(x * w), int(y * h)) for (x, y, _z) in xyz]
        feature = normalize_landmarks(xyz)

        self._motion_buffer.append(feature)
        is_moving = self._is_moving()
        was_gesturing = len(self._gesture_window) > 0

        if is_moving and self._movement_model is not None:
            self._gesture_window.append((now, feature))
            # poda por TEMPO real, não por contagem de frames - assim a
            # duração da janela não depende da velocidade da câmera/máquina
            while self._gesture_window and (now - self._gesture_window[0][0]) > config.MOVEMENT_WINDOW_SECONDS:
                self._gesture_window.popleft()

            span = now - self._gesture_window[0][0] if len(self._gesture_window) >= 2 else 0.0
            if span >= config.MOVEMENT_WINDOW_SECONDS:
                # teto de segurança: gesto não parou sozinho até aqui -
                # classifica com o que já foi capturado e reinicia
                detection = self._classify_gesture()
                self._gesture_window.clear()
                return detection, landmarks_px
            # ainda gesticulando - não roda a MLP estática nesse
            # meio-tempo pra não piscar letra estática errada
            return None, landmarks_px

        if was_gesturing:
            # a mão ACABOU de parar - esse é o momento natural de
            # classificar o gesto que acabou de terminar (em vez de
            # esperar um número fixo de frames)
            span = self._gesture_window[-1][0] - self._gesture_window[0][0] if len(self._gesture_window) >= 2 else 0.0
            detection = self._classify_gesture() if span >= config.MOVEMENT_MIN_SECONDS else None
            self._gesture_window.clear()
            return detection, landmarks_px

        # Mão parada e sem gesto pendente (ou modelo de movimento
        # indisponível): comportamento normal, MLP estática frame a frame.
        detection = self._classify_static(feature)
        return detection, landmarks_px

    def _is_moving(self):
        if len(self._motion_buffer) < 2:
            return False
        diffs = [
            np.linalg.norm(self._motion_buffer[i] - self._motion_buffer[i - 1])
            for i in range(1, len(self._motion_buffer))
        ]
        return float(np.mean(diffs)) > config.MOTION_THRESHOLD

    def _classify_static(self, feature):
        probs = self._model.predict_proba(feature.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        letter = self._labels[best_idx]
        conf_threshold = self._get_conf_threshold()
        if conf < conf_threshold:
            return None
        return Detection(letter, conf, instant=False)

    def _classify_gesture(self):
        raw_sequence = np.array([f for _t, f in self._gesture_window], dtype=np.float32)
        sequence = resample_sequence(raw_sequence, config.MOVEMENT_STEPS)
        probs = self._movement_model.predict_proba(sequence)
        best_idx = int(np.argmax(probs))
        conf = float(probs[best_idx])
        letter = self._movement_model.classes[best_idx]
        if conf < config.MOVEMENT_CONF_THRESHOLD:
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
        try:
            self._movement_model = LSTMNumpy(config.MOVEMENT_MODEL_PATH)
            logger.info(
                "Modelo de gestos carregado com sucesso. Classes: %s",
                self._movement_model.classes,
            )
        except FileNotFoundError:
            logger.info(
                "Modelo de gestos não encontrado em %s - app funcionará só com "
                "letras estáticas. Rode collect_movement.py + "
                "train_movement_classifier.py pra habilitar H/J/K/X/Z.",
                config.MOVEMENT_MODEL_PATH,
            )
            self._movement_model = None
        except Exception:
            logger.exception("Falha ao carregar o modelo de gestos (%s) - ignorando.", config.MOVEMENT_MODEL_PATH)
            self._movement_model = None

    def stop(self):
        self._running = False
