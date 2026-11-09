"""
Ferramenta de coleta de dataset para o alfabeto de LIBRAS baseado em
landmarks do MediaPipe Hands.

Uso:
    python collect_landmarks.py
    python collect_landmarks.py --movement  # sequências de H/J/K/X/Z e rejeição

No modo --movement, segure Espaço durante o gesto, solte para parar e use S
para salvar. Na coleta por quadro, selecione _ para poses que não são letras.

Controles:
    SETA ESQUERDA/DIREITA (ou A / D)  -> troca a letra "alvo" atual
                                          (A/D é a opção garantida em
                                          qualquer SO - os códigos de
                                          seta variam por plataforma)
    ESPAÇO                    -> captura UMA amostra da letra atual
    C (segurar)                -> captura contínua (~8 amostras/seg)
                                  enquanto a tecla estiver pressionada
                                  - útil pra juntar variação natural
                                  de leve tremor/ângulo rápido
    S                           -> salva o dataset acumulado em disco
                                    (também salva automaticamente ao
                                    sair com ESC)
    R                           -> remove a ÚLTIMA amostra capturada
                                    (desfazer captura errada)
    ESC                         -> salva e sai

Cada amostra salva é o vetor de 63 features (normalizado - ver
landmarks_utils.py) + a letra, uma linha por amostra, em
dataset/landmarks.csv (acumula entre execuções - pode rodar o script
várias vezes ao longo dos dias pra ir completando o dataset).

Recomendação: tente variar um pouco o ângulo da mão, distância da
câmera e iluminação entre as capturas de uma mesma letra - isso deixa
o modelo final mais robusto no dia da apresentação (evita treinar só
com a mão numa posição "perfeita" única).
"""

import argparse
import csv
import os
import time

import cv2
import mediapipe as mp

import config
from landmarks_utils import mediapipe_landmarks_to_xyz, normalize_landmarks
from movement import REJECT_LABEL

LETTERS_LIST = [*sorted(config.LETTERS.values()), REJECT_LABEL]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def load_existing_counts(csv_path):
    counts = {letter: 0 for letter in LETTERS_LIST}
    if not os.path.exists(csv_path):
        return counts
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # cabeçalho
        for row in reader:
            if row:
                label = row[-1]
                counts[label] = counts.get(label, 0) + 1
    return counts


def append_rows(csv_path, rows):
    is_new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
            writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Coleta de landmarks para Libras")
    parser.add_argument("--movement", action="store_true", help="Coletar sequências de movimento para MLP ou LSTM")
    parser.add_argument("--camera", type=int, default=0, help="Índice da câmera")
    args = parser.parse_args()
    if args.movement:
        from collect_movement import collect

        collect(args.camera)
        return

    csv_path = config.DATASET_CSV_PATH
    counts = load_existing_counts(csv_path)
    pending_rows = []  # amostras da sessão atual, ainda não salvas em disco
    target_idx = 0
    last_auto_capture = 0.0
    auto_capture_interval = 1.0 / 8  # ~8 amostras/seg no modo "segurar C"

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Coleta de dataset LIBRAS - ESC pra salvar e sair. Veja os controles no topo do arquivo.")

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

            current_feature = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                xyz = mediapipe_landmarks_to_xyz(hand_landmarks)
                current_feature = normalize_landmarks(xyz)

            target_letter = LETTERS_LIST[target_idx]
            target_name = "Nula (_)" if target_letter == REJECT_LABEL else target_letter

            # --- overlay de texto ---
            hand_status = "MAO DETECTADA" if current_feature is not None else "sem mao"
            hand_color = (80, 220, 80) if current_feature is not None else (60, 60, 220)
            cv2.putText(frame, f"Classe alvo: {target_name}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Amostras salvas: {counts[target_letter]}  (+{sum(1 for r in pending_rows if r[-1] == target_letter)} nao salvas)",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, hand_status, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
            cv2.putText(frame, "<-/A e ->/D trocam letra | ESPACO captura | C segura=continuo | S salva | R desfaz | ESC sai",
                        (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Coleta de dataset - LIBRAS", frame)
            key = cv2.waitKeyEx(1)

            if key == -1:
                continue

            key_ascii = key & 0xFF  # seguro para comparar teclas ASCII normais (letras, espaço, ESC)

            # Setas: os códigos variam por SO/backend do OpenCV, então
            # aceitamos os valores mais comuns (Windows sem/with Ex,
            # Linux/GTK) - e também A/D como alternativa que SEMPRE
            # funciona, independente de plataforma.
            LEFT_CODES = (2424832, 65361, 81, ord("a"), ord("A"))
            RIGHT_CODES = (2555904, 65363, ord("d"), ord("D"))

            if key_ascii == 27:  # ESC
                break
            elif key in LEFT_CODES or key_ascii in LEFT_CODES:
                target_idx = (target_idx - 1) % len(LETTERS_LIST)
            elif key in RIGHT_CODES or key_ascii in RIGHT_CODES:
                target_idx = (target_idx + 1) % len(LETTERS_LIST)
            elif key_ascii == ord(" "):
                if current_feature is not None:
                    pending_rows.append(list(current_feature) + [target_letter])
                    counts[target_letter] += 1
                    print(f"Capturado 1 amostra de '{target_letter}' (total agora: {counts[target_letter]})")
                else:
                    print("Nenhuma mão detectada - amostra ignorada.")
            elif key_ascii in (ord("c"), ord("C")):
                now = time.time()
                if current_feature is not None and (now - last_auto_capture) >= auto_capture_interval:
                    pending_rows.append(list(current_feature) + [target_letter])
                    counts[target_letter] += 1
                    last_auto_capture = now
            elif key_ascii in (ord("s"), ord("S")):
                if pending_rows:
                    append_rows(csv_path, pending_rows)
                    print(f"Salvo {len(pending_rows)} amostras em {csv_path}")
                    pending_rows = []
            elif key_ascii in (ord("r"), ord("R")):
                if pending_rows:
                    removed = pending_rows.pop()
                    counts[removed[-1]] -= 1
                    print(f"Removida a última amostra ('{removed[-1]}').")

    if pending_rows:
        append_rows(csv_path, pending_rows)
        print(f"Salvo {len(pending_rows)} amostras em {csv_path}")

    cap.release()
    cv2.destroyAllWindows()

    print("\nResumo de amostras por letra:")
    for letter in LETTERS_LIST:
        print(f"  {letter}: {counts[letter]}")


if __name__ == "__main__":
    main()
