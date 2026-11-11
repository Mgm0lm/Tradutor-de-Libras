"""
Treina o classificador de letras a partir do dataset (dataset/landmarks.csv
+ dataset/landmarks_external.csv, se existir).

Usa scikit-learn (MLPClassifier) só AQUI, no seu ambiente de
desenvolvimento, pra treinar e avaliar o modelo. Mas em vez de salvar
o objeto do sklearn inteiro (via joblib/pickle - o que exigiria
scikit-learn E scipy dentro do .exe, e o scipy.stats tem um bug
conhecido de incompatibilidade com PyInstaller que não tem fix
confiável até agora), este script extrai só os PESOS BRUTOS da rede
(matrizes de peso + bias de cada camada) e salva num .npz simples.

O app (classifier.py) faz a inferência com numpy puro (multiplicação
de matriz + ReLU + softmax - uma MLP não é mais que isso), sem
precisar de scikit-learn nem scipy em tempo de execução. Resultado:
.exe mais leve, mais rápido de abrir, e livre desse bug de
empacotamento.

Uso:
    python train_classifier.py
    python train_classifier.py --edit X Z
    python train_classifier.py --movement
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import config
from movement import REJECT_LABEL, STEPS, TRAJECTORY_FEATURES, load_samples

HIDDEN_LAYER_SIZES = (64, 32)


def load_dataset(csv_paths, excluded=()):
    X, y = [], []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue
        count_before = len(X)
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # cabeçalho
            for row in reader:
                if not row:
                    continue
                *features, label = row
                if label in excluded:
                    continue
                X.append([float(v) for v in features])
                y.append(label)
        print(f"  {csv_path}: {len(X) - count_before} amostras")
    return np.array(X, dtype=np.float32), np.array(y)


def export_weights(clf, labels_sorted, path, **metadata):
    """Extrai os pesos brutos da MLPClassifier treinada e salva num
    .npz simples, sem depender de sklearn/scipy pra ler depois."""
    weights = {}
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        weights[f"W{i}"] = W.astype(np.float32)
        weights[f"b{i}"] = b.astype(np.float32)

    if len(labels_sorted) == 2:
        # sklearn uses one sigmoid logit for two classes. Softmax([0, z])
        # gives the same pair of probabilities and keeps the NPZ portable.
        output = len(clf.coefs_) - 1
        w_key, b_key = f"W{output}", f"b{output}"
        weights[w_key] = np.column_stack((np.zeros_like(weights[w_key]), weights[w_key]))
        weights[b_key] = np.concatenate((np.zeros_like(weights[b_key]), weights[b_key]))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        n_layers=len(clf.coefs_),
        classes=np.array(labels_sorted),
        **weights,
        **metadata,
    )


def train_movement_mlp(directory, model_path, excluded=()):
    """Train on ordered windows, keeping each capture whole in its partition."""
    letters = [letter for letter in config.MOVEMENT_LETTERS if letter not in excluded]
    if not letters:
        raise ValueError("Deixe pelo menos uma letra com movimento para treinar junto com a classe nula (_).")
    data = load_samples(Path(directory), [*letters, REJECT_LABEL], TRAJECTORY_FEATURES)
    if not any(letter in data for letter in letters):
        raise ValueError("Colete gestos com python collect_landmarks.py --movement.")
    insufficient = [label for label in data if len(data[label]) < 10]
    if REJECT_LABEL not in data:
        insufficient.append(REJECT_LABEL)
    if insufficient:
        raise ValueError(f"Colete pelo menos 10 sequências para: {', '.join(insufficient)}. Use _ para outros movimentos.")

    train, train_y, test, test_y = [], [], [], []
    for label, samples in data.items():
        cut = int(len(samples) * 0.8)
        train.extend(samples[:cut])
        train_y.extend([label] * cut)
        test.extend(samples[cut:])
        test_y.extend([label] * (len(samples) - cut))
        print(f"  {label}: {cut} para treino, {len(samples) - cut} para teste")
    missing = [label for label in letters if label not in data]
    if missing:
        print(f"Sem amostras para: {', '.join(missing)}. Essas letras não serão reconhecidas.")

    width = STEPS * TRAJECTORY_FEATURES
    X_train = np.asarray(train).reshape(-1, width)
    X_test = np.asarray(test).reshape(-1, width)
    clf = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=500, random_state=42)
    clf.fit(X_train, train_y)
    print("\nTeste local (últimos 20% de cada classe; sem separação por pessoa/sessão):")
    print(classification_report(test_y, clf.predict(X_test), zero_division=0))
    export_weights(clf, clf.classes_, model_path, steps=STEPS, feature_count=TRAJECTORY_FEATURES)
    print(f"MLP de movimento salva em {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Treina a MLP de Libras")
    parser.add_argument("--movement", action="store_true", help="Treinar uma MLP separada com sequências de movimento")
    parser.add_argument(
        "--edit", nargs="+", type=str.upper, choices=sorted(config.LETTERS.values()),
        metavar="LETRA", help="Excluir letras deste treino, por exemplo: --edit X Z",
    )
    args = parser.parse_args()
    excluded = set(args.edit or ())
    if excluded:
        print(f"Letras excluídas deste treino: {', '.join(sorted(excluded))}")
    if args.movement:
        try:
            train_movement_mlp(config.MOVEMENT_TRAJECTORY_DIR, config.MOVEMENT_MLP_MODEL_PATH, excluded)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return

    csv_paths = [config.DATASET_CSV_PATH, config.EXTERNAL_DATASET_CSV_PATH]
    existing = [p for p in csv_paths if os.path.exists(p)]
    if not existing:
        print(f"Nenhum dataset encontrado ({', '.join(csv_paths)}).")
        print("Rode primeiro: python collect_landmarks.py")
        print("(opcional) e/ou: python import_cnn_libras_dataset.py <pasta>")
        return

    print("Carregando dataset de:")
    X, y = load_dataset(csv_paths, excluded)
    if excluded and len(set(y)) < 2:
        parser.error("O treino precisa de pelo menos duas classes com amostras após a exclusão.")
    try:
        train_frame_mlp(X, y, config.MODEL_PATH)
    except ValueError as error:
        parser.error(str(error))


def train_frame_mlp(X, y, model_path):
    """Infer the output classes from the labels present in the CSV."""
    print(f"\nTotal: {len(X)} amostras, {len(set(y))} classes.")

    counts = {label: int((y == label).sum()) for label in sorted(set(y))}
    if REJECT_LABEL not in counts:
        print("Sem amostras da classe nula (_). O treino segue com as letras disponíveis.")
    print("Amostras por letra:")
    for label, n in counts.items():
        marker = "  <- poucas amostras, considere coletar mais" if n < 30 else ""
        print(f"  {label}: {n}{marker}")

    if len(X) < 50:
        print("\nAVISO: dataset muito pequeno ainda. Colete mais amostras "
              "(recomendado: pelo menos ~50-100 por letra) antes de confiar "
              "no resultado abaixo.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        max_iter=2000,
        random_state=42,
        early_stopping=True,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Relatório de avaliação (conjunto de teste, 20% separado) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== Matriz de confusão ===")
    labels_sorted = sorted(set(y))
    print("Linhas = letra real, colunas = letra prevista, ordem:", labels_sorted)
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

    # Retreina no dataset inteiro pra não desperdiçar dados no modelo
    # final - a divisão acima foi só pra medir a qualidade de forma honesta.
    clf_final = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        max_iter=2000,
        random_state=42,
        early_stopping=True,
    )
    clf_final.fit(X, y)
    export_weights(clf_final, clf_final.classes_, model_path)
    print(f"\nModelo salvo em {model_path} (pesos brutos, sem sklearn/scipy necessários pra usar)")


if __name__ == "__main__":
    main()
