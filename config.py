"""
============================================================
 CONFIGURACOES DO TRADUTOR DE LIBRAS (MediaPipe + RandomForest)
============================================================
Edite os valores abaixo para ajustar o comportamento do
programa sem precisar mexer no resto do codigo.
"""

import os
import sys

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# MODELO (treinado com train_classifier.py a partir de
# dataset/landmarks.csv, gerado por collect_landmarks.py)
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "modelo", "landmark_model.npz")
DATASET_CSV_PATH = os.path.join(BASE_DIR, "dataset", "landmarks.csv")
EXTERNAL_DATASET_CSV_PATH = os.path.join(BASE_DIR, "dataset", "landmarks_external.csv")

# ------------------------------------------------------------------
# LETRAS COM MOVIMENTO (LSTM)
# ------------------------------------------------------------------
# Letras do alfabeto estático (MLP, frame a frame) x letras que exigem
# movimento (LSTM, janela de frames). O detector de movimento decide
# em tempo real qual das duas usar - ver classifier.py.
MOVEMENT_LETTERS = ["H", "J", "K", "X", "Z"]

MOVEMENT_MODEL_PATH = os.path.join(BASE_DIR, "modelo", "movement_model.npz")
MOVEMENT_DATASET_DIR = os.path.join(BASE_DIR, "dataset", "movement_sequences")
MOVEMENT_TRAJECTORY_DIR = os.path.join(BASE_DIR, "dataset", "movement_trajectory")
ENABLE_MOVEMENT = False  # Experimental; opt in through config_override.json.

# Duração MÁXIMA real (em segundos) de um gesto - funciona como teto de
# segurança: se a mão não parar de se mover antes disso, classifica com
# o que já foi capturado até aqui e reinicia. Ajuste conforme o gesto
# mais lento que vocês fizerem na prática.
MOVEMENT_WINDOW_SECONDS = 2.0

# Duração MÍNIMA (segundos) pra considerar um "movimento" como gesto
# válido, e não ruído/tremor de 1-2 frames.
MOVEMENT_MIN_SECONDS = 0.25

# Nº de passos de tempo que a sequência é reamostrada (via interpolação
# - ver landmarks_utils.resample_sequence) antes de entrar na LSTM. Não
# é mais "quantidade de frames capturados" - é sempre esse valor fixo,
# não importa quão rápido/devagar o gesto foi feito ou quão rápida é a
# máquina. Só precisa bater entre coleta e treino (train_movement_
# classifier.py usa esse mesmo valor).
MOVEMENT_STEPS = 20

# Janela curta (nº de frames) usada só para MEDIR velocidade/movimento
# - não confundir com MOVEMENT_WINDOW_SIZE (que é o buffer completo do
# gesto). Um valor pequeno deixa a detecção de "começou a mover"
# responsiva.
MOTION_DETECT_FRAMES = 4

# Limiar de deslocamento médio (em unidades normalizadas - ver
# landmarks_utils.py) entre frames consecutivos para considerar que a
# mão está "em movimento" em vez de "parada". Ajuste testando ao vivo:
# se letras ESTÁTICAS estiverem sendo confundidas com movimento (por
# causa de tremor natural da mão), AUMENTE este valor. Se gestos
# genuínos não estiverem sendo detectados como movimento, DIMINUA.
MOTION_THRESHOLD = 0.06

# Confiança mínima do classificador de movimento para aceitar a letra.
MOVEMENT_CONF_THRESHOLD = 0.70

# Alfabeto estático do LIBRAS suportado (letras que não exigem
# movimento - faltam H, J, K, X, Z).
LETTERS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "I",
    8: "L", 9: "M", 10: "N", 11: "O", 12: "P", 13: "Q", 14: "R",
    15: "S", 16: "T", 17: "U", 18: "V", 19: "W", 20: "Y",
}

# ------------------------------------------------------------------
# MEDIAPIPE HANDS
# ------------------------------------------------------------------
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.6   # confiança mínima pra achar UMA mão no frame
MIN_TRACKING_CONFIDENCE = 0.5    # confiança mínima pra manter o rastreio entre frames

# Confiança mínima do CLASSIFICADOR (predict_proba da letra escolhida)
# pra considerar a letra válida - ajustável em tempo real na interface.
CONF_THRESHOLD_DEFAULT = 0.70
CONF_THRESHOLD_MIN = 0.10
CONF_THRESHOLD_MAX = 0.98
CONF_THRESHOLD_STEP = 0.05

# ------------------------------------------------------------------
# REGRAS DE ESTABILIZACAO
# ------------------------------------------------------------------
MIN_STABLE_SECONDS = 0.8
COOLDOWN_SECONDS = 1.8

# ------------------------------------------------------------------
# DESEMPENHO / CAMERA
# ------------------------------------------------------------------
CAMERA_FPS_TARGET = 30
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
DESYNC_INFERENCE = True
MAX_CAMERAS_TO_SCAN = 5

# ------------------------------------------------------------------
# TTS (Google Text-to-Speech via gTTS)
# ------------------------------------------------------------------
ENABLE_TTS = True
TTS_LANG = "pt"
TTS_CACHE_DIR = os.path.join(BASE_DIR, "tts_cache")

# ------------------------------------------------------------------
# INTERFACE
# ------------------------------------------------------------------
APP_TITLE = "Tradutor de LIBRAS - Alfabeto"
UI_RADIUS = 14
SHOW_FPS = True

# ------------------------------------------------------------------
# OVERRIDE EXTERNO
# ------------------------------------------------------------------
import json

CONFIG_OVERRIDE_PATH = os.path.join(BASE_DIR, "config_override.json")


def _apply_overrides():
    if not os.path.exists(CONFIG_OVERRIDE_PATH):
        return
    try:
        with open(CONFIG_OVERRIDE_PATH, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            globals()[key] = value
        print(f"[config] Overrides aplicados de {CONFIG_OVERRIDE_PATH}: {list(overrides.keys())}")
    except Exception as e:
        print(f"[config] Falha ao ler {CONFIG_OVERRIDE_PATH}, usando padrões: {e}")


_apply_overrides()
