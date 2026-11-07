"""
Ferramenta de coleta de dataset para as letras de LIBRAS que exigem
MOVIMENTO (H, J, K, X, Z) - diferente de collect_landmarks.py, aqui
cada amostra é uma SEQUÊNCIA de frames (o gesto inteiro), não um único
frame.

Uso:
    python collect_movement.py

Fluxo de captura: o script mantém um buffer contínuo (rolling) com os
últimos config.MOVEMENT_WINDOW_SECONDS segundos de landmarks (não uma
quantidade fixa de frames - a duração real em segundos é o que
importa, pra não depender da velocidade da câmera/máquina). Você faz o
gesto na frente da câmera e, QUANDO TERMINAR, aperta ESPAÇO - o script
pega tudo que está no buffer (o gesto que você acabou de fazer, na
duração real que ele levou) e REAMOSTRA (via interpolação) pra
exatamente config.MOVEMENT_STEPS passos antes de salvar - assim gestos
rápidos e lentos viram entradas do mesmo tamanho pro modelo, sem perder
a forma do trajeto. Não precisa acertar o timing de quando "começar a
gravar": grava sempre, e você confirma no final.

Controles:
    A / D (ou setas)   -> troca a letra alvo (H, J, K, X, Z)
    ESPAÇO              -> salva os últimos N frames como uma amostra
                            do gesto que você acabou de terminar
    R                    -> apaga a última amostra salva (desfazer)
    ESC                  -> sai

Cada amostra vira um arquivo .npy em
dataset/movement_sequences/<LETRA>/sample_NNNN.npy, shape
(MOVEMENT_STEPS, 63).

Recomendação: faça cada gesto em velocidade natural (nem muito rápido
nem em câmera lenta) e varie um pouco o ponto de partida na tela entre
as capturas - o traço da letra é o que importa, não a posição.
"""

import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

import config
from landmarks_utils import mediapipe_landmarks_to_xyz, normalize_landmarks, resample_sequence

LETTERS_LIST = config.MOVEMENT_LETTERS

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def sample_dir(letter):
    return os.path.join(config.MOVEMENT_DATASET_DIR, letter)


def count_samples(letter):
    d = sample_dir(letter)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".npy")])


def next_sample_path(letter):
    d = sample_dir(letter)
    os.makedirs(d, exist_ok=True)
    n = count_samples(letter)
    return os.path.join(d, f"sample_{n:04d}.npy"), n


def main():
    target_idx = 0
    buffer = deque()  # lista de (timestamp, feature) - sem limite fixo de tamanho, poda por tempo
    last_saved_path = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Coleta de gestos LIBRAS - ESC pra sair. Veja os controles no topo do arquivo.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            now = time.time()

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                xyz = mediapipe_landmarks_to_xyz(hand_landmarks)
                feature = normalize_landmarks(xyz)
                buffer.append((now, feature))
            else:
                buffer.clear()  # gesto precisa de mão contínua na cena

            # poda o que passou de MOVEMENT_WINDOW_SECONDS (janela deslizante por tempo)
            while buffer and (now - buffer[0][0]) > config.MOVEMENT_WINDOW_SECONDS:
                buffer.popleft()

            span = (buffer[-1][0] - buffer[0][0]) if len(buffer) >= 2 else 0.0
            buffer_ready = span >= config.MOVEMENT_MIN_SECONDS

            target_letter = LETTERS_LIST[target_idx]

            cv2.putText(frame, f"Letra alvo: {target_letter}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Amostras salvas: {count_samples(target_letter)}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            status_color = (80, 220, 80) if buffer_ready else (60, 60, 220)
            status_text = f"Buffer: {span:.2f}s / min {config.MOVEMENT_MIN_SECONDS:.2f}s" + (" - pronto pra salvar" if buffer_ready else "")
            cv2.putText(frame, status_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, "A/D troca letra | ESPACO salva o gesto | R desfaz | ESC sai",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Coleta de gestos - LIBRAS", frame)
            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            key_ascii = key & 0xFF

            if key_ascii == 27:  # ESC
                break
            elif key in (2424832, 65361, 81, ord("a"), ord("A")):
                target_idx = (target_idx - 1) % len(LETTERS_LIST)
            elif key in (2555904, 65363, 83, ord("d"), ord("D")):
                target_idx = (target_idx + 1) % len(LETTERS_LIST)
            elif key_ascii == ord(" "):
                if buffer_ready:
                    raw_sequence = np.array([f for _t, f in buffer], dtype=np.float32)
                    resampled = resample_sequence(raw_sequence, config.MOVEMENT_STEPS)
                    path, _ = next_sample_path(target_letter)
                    np.save(path, resampled)
                    last_saved_path = path
                    print(f"Salvo gesto de '{target_letter}' ({span:.2f}s, {len(buffer)} frames brutos -> "
                          f"{config.MOVEMENT_STEPS} passos) em {path} (total agora: {count_samples(target_letter)})")
                else:
                    print(f"Buffer curto demais ({span:.2f}s < {config.MOVEMENT_MIN_SECONDS:.2f}s) - continue o gesto antes de salvar.")
            elif key_ascii in (ord("r"), ord("R")):
                if last_saved_path and os.path.exists(last_saved_path):
                    os.remove(last_saved_path)
                    print(f"Removido {last_saved_path}")
                    last_saved_path = None

    cap.release()
    cv2.destroyAllWindows()

    print("\nResumo de amostras por letra:")
    for letter in LETTERS_LIST:
        print(f"  {letter}: {count_samples(letter)}")


if __name__ == "__main__":
    main()
