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
"""

import csv
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import config

HIDDEN_LAYER_SIZES = (64, 32)


def load_dataset(csv_paths):
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
                X.append([float(v) for v in features])
                y.append(label)
        print(f"  {csv_path}: {len(X) - count_before} amostras")
    return np.array(X, dtype=np.float32), np.array(y)


def export_weights(clf, labels_sorted, path):
    """Extrai os pesos brutos da MLPClassifier treinada e salva num
    .npz simples, sem depender de sklearn/scipy pra ler depois."""
    weights = {}
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        weights[f"W{i}"] = W.astype(np.float32)
        weights[f"b{i}"] = b.astype(np.float32)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        n_layers=len(clf.coefs_),
        classes=np.array(labels_sorted),
        **weights,
    )


def main():
    csv_paths = [config.DATASET_CSV_PATH, config.EXTERNAL_DATASET_CSV_PATH]
    existing = [p for p in csv_paths if os.path.exists(p)]
    if not existing:
        print(f"Nenhum dataset encontrado ({', '.join(csv_paths)}).")
        print("Rode primeiro: python collect_landmarks.py")
        print("(opcional) e/ou: python import_cnn_libras_dataset.py <pasta>")
        return

    print("Carregando dataset de:")
    X, y = load_dataset(csv_paths)
    print(f"\nTotal: {len(X)} amostras, {len(set(y))} classes.")

    counts = {label: int((y == label).sum()) for label in sorted(set(y))}
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
    labels_sorted_final = sorted(clf_final.classes_.tolist())

    export_weights(clf_final, sorted(clf_final.classes_.tolist()), config.MODEL_PATH)
    print(f"\nModelo salvo em {config.MODEL_PATH} (pesos brutos, sem sklearn/scipy necessários pra usar)")


if __name__ == "__main__":
    main()
