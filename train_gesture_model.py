"""
Coleta amostras do gesto de "finalizar compra" (landmarks das duas
mãos, via MediaPipe) e treina o classificador usado por gesture.py.

USO:
    python train_gesture_model.py

CONTROLES (janela da webcam):
    P    - captura uma amostra POSITIVA (o gesto escolhido pra
           finalizar a compra). Segure a tecla pra capturar várias
           seguidas (usa o autorepeat do teclado), variando um pouco
           a posição/ângulo das mãos entre uma captura e outra.
    O    - captura uma amostra NEGATIVA (qualquer outra coisa com as
           duas mãos: paradas, acenando, segurando produto, gestos
           aleatórios - quanto mais variado, melhor o modelo aprende
           a diferença).
    S    - salva as amostras acumuladas em modelo/gesture_dataset.csv
           (a captura P/O só guarda em memória até você salvar).
    T    - salva e treina o modelo com TODO o dataset salvo até agora
           (inclusive de sessões anteriores), gerando
           modelo/gesture_model.joblib.
    ESC  - sai (pergunta se quer salvar amostras pendentes).

Só é possível capturar quando exatamente 2 mãos estão visíveis - é a
mesma regra usada pelo app em produção.

Recomendação: grave bem mais amostras negativas que positivas (ex:
100 positivas e 300+ negativas, incluindo momentos de "mão normal"
enquanto segura produtos, gesticula, etc.) para o modelo não ficar
"gatilho fácil".
"""

import csv
import os

import cv2
import numpy as np

import config
from gesture_features import extract_features, NUM_FEATURES


def main():
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    os.makedirs(os.path.dirname(config.GESTURE_DATASET_PATH), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não consegui abrir a câmera (índice 0). Troque o índice na linha "
              "'cv2.VideoCapture(0)' se sua câmera não for a 0.")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    )

    samples = []  # [(features, label), ...] - ainda nao salvos em disco
    n_pos, n_neg = 0, 0

    print("Controles: [P] positivo  [O] negativo  [S] salvar  [T] treinar  [ESC] sair")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # espelha - mais intuitivo pro usuario
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for lm in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            features = None
            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
                features = extract_features(result.multi_hand_landmarks, result.multi_handedness)

            ready = features is not None
            status = "2 MAOS OK - pronto pra capturar" if ready else "Mostre as DUAS maos pra capturar"
            color = (0, 200, 0) if ready else (0, 0, 200)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Positivos: {n_pos}   Negativos: {n_neg}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "[P] positivo  [O] negativo  [S] salvar  [T] treinar  [ESC] sair",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Treino de gesto - Caixa Autonomo", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key in (ord("p"), ord("P")) and features is not None:
                samples.append((features, 1))
                n_pos += 1
            elif key in (ord("o"), ord("O")) and features is not None:
                samples.append((features, 0))
                n_neg += 1
            elif key in (ord("s"), ord("S")):
                _save_dataset(samples)
                samples = []
            elif key in (ord("t"), ord("T")):
                _save_dataset(samples)
                samples = []
                _train_and_save()
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    if samples:
        resp = input(f"Você tem {len(samples)} amostra(s) não salva(s). Salvar agora? [s/N] ")
        if resp.strip().lower() == "s":
            _save_dataset(samples)


def _save_dataset(samples):
    if not samples:
        return
    file_exists = os.path.exists(config.GESTURE_DATASET_PATH)
    with open(config.GESTURE_DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([f"f{i}" for i in range(NUM_FEATURES)] + ["label"])
        for features, label in samples:
            writer.writerow(list(features) + [label])
    print(f"{len(samples)} amostra(s) salva(s) em {config.GESTURE_DATASET_PATH}")


def _load_dataset():
    X, y = [], []
    with open(config.GESTURE_DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # cabecalho
        for row in reader:
            *feats, label = row
            X.append([float(v) for v in feats])
            y.append(int(label))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _train_and_save():
    if not os.path.exists(config.GESTURE_DATASET_PATH):
        print("Nenhum dataset encontrado ainda. Capture amostras com [P]/[O] antes de treinar.")
        return

    X, y = _load_dataset()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"Dataset: {len(X)} amostra(s) ({n_pos} positiva(s), {n_neg} negativa(s))")

    if n_pos == 0 or n_neg == 0:
        print("Preciso de pelo menos 1 amostra positiva E 1 negativa pra treinar.")
        return

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib

    can_split = len(X) >= 10 and n_pos >= 2 and n_neg >= 2
    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = np.empty((0, X.shape[1])), np.empty((0,))

    model_type = getattr(config, "GESTURE_MODEL_TYPE", "mlp").lower()
    if model_type == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        k = min(5, n_pos, n_neg)
        model = KNeighborsClassifier(n_neighbors=max(1, k))
    else:
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)

    model.fit(X_train, y_train)

    if len(X_test) > 0:
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Acurácia no conjunto de teste: {acc * 100:.1f}%")
    else:
        print("Poucas amostras pra separar um conjunto de teste - treinado com tudo, sem validação.")
        print("Colete mais amostras (recomendado: 20+ de cada classe) pra uma avaliação confiável.")

    os.makedirs(os.path.dirname(config.GESTURE_MODEL_PATH), exist_ok=True)
    joblib.dump(model, config.GESTURE_MODEL_PATH)
    print(f"Modelo salvo em {config.GESTURE_MODEL_PATH} (tipo: {model_type})")


if __name__ == "__main__":
    main()
