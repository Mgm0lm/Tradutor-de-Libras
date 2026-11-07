"""
Normalização dos 21 landmarks da mão (MediaPipe Hands) em um vetor de
features invariante a posição na tela e a distância da câmera.

ESSENCIAL: esta mesma função é usada na hora de COLETAR o dataset
(collect_landmarks.py), de TREINAR (train_classifier.py) e de INFERIR
em tempo real (classifier.py). Se você mudar a lógica aqui, precisa
regerar o dataset e retreinar - senão o modelo passa a ver features
calculadas de um jeito diferente do que foi treinado.
"""

import numpy as np

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3  # x, y, z por ponto


def normalize_landmarks(landmarks_xyz):
    """
    landmarks_xyz: sequência de 21 pontos (x, y, z), no formato que o
    MediaPipe já entrega (coordenadas normalizadas 0.0-1.0 relativas à
    imagem, z relativo à profundidade do pulso).

    Retorna um np.ndarray de 63 floats (21 * 3), com:
    - Translação: todos os pontos ficam relativos ao pulso (landmark 0),
      que vira a origem (0, 0, 0). Isso torna a classificação
      indiferente a ONDE na tela a mão está.
    - Escala: os pontos são divididos pela maior distância entre o
      pulso e qualquer outro landmark. Isso torna a classificação
      indiferente ao TAMANHO da mão na imagem (ou seja, à distância da
      câmera) - uma mão longe ou perto da câmera gera o mesmo vetor.
    """
    pts = np.asarray(landmarks_xyz, dtype=np.float32).reshape(NUM_LANDMARKS, 3)
    wrist = pts[0].copy()
    pts = pts - wrist

    scale = float(np.linalg.norm(pts, axis=1).max())
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale

    return pts.flatten()


def resample_sequence(sequence, target_len):
    """Reamostra (via interpolação linear) uma sequência de T passos
    de tempo para exatamente target_len passos, preservando o formato
    do trajeto independente de quantos frames foram capturados.

    Isso desacopla a duração REAL do gesto (em segundos) da taxa de
    quadros da câmera/inferência - um gesto rápido feito numa máquina
    rápida e o mesmo gesto lento numa máquina fraca acabam virando
    entradas do mesmo tamanho pro modelo, só que "esticadas" ou
    "comprimidas" no tempo. sequence: array (T, F). Retorna (target_len, F).
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    t = sequence.shape[0]
    if t == target_len:
        return sequence
    if t < 2:
        return np.repeat(sequence, target_len, axis=0)[:target_len]

    old_idx = np.linspace(0.0, 1.0, t)
    new_idx = np.linspace(0.0, 1.0, target_len)
    resampled = np.empty((target_len, sequence.shape[1]), dtype=np.float32)
    for f in range(sequence.shape[1]):
        resampled[:, f] = np.interp(new_idx, old_idx, sequence[:, f])
    return resampled


def mediapipe_landmarks_to_xyz(hand_landmarks):
    """Converte o objeto de resultado do MediaPipe (hand_landmarks.landmark,
    lista de 21 NormalizedLandmark) para uma lista simples de tuplas
    (x, y, z), no formato esperado por normalize_landmarks()."""
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
