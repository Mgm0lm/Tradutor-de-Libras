"""
Treina o classificador de GESTOS (H, J, K, X, Z) a partir das
sequências gravadas por collect_movement.py.

Usa PyTorch (nn.LSTM) só AQUI, no seu ambiente de desenvolvimento -
igual ao scikit-learn no train_classifier.py, PyTorch NÃO precisa
estar instalado no .exe final. Depois de treinar, este script extrai
os pesos brutos da LSTM (as mesmas matrizes que o PyTorch usa
internamente) e salva num .npz simples. O app (classifier.py) refaz o
forward pass manualmente com numpy - sem PyTorch em tempo de execução.

Uso:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python train_movement_classifier.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import config

HIDDEN_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.01


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # usa o estado oculto do último frame do gesto
        return self.fc(last_step)


def load_dataset():
    X, y = [], []
    for letter in config.MOVEMENT_LETTERS:
        letter_dir = os.path.join(config.MOVEMENT_DATASET_DIR, letter)
        if not os.path.isdir(letter_dir):
            continue
        files = sorted(f for f in os.listdir(letter_dir) if f.endswith(".npy"))
        for filename in files:
            arr = np.load(os.path.join(letter_dir, filename))
            if arr.shape[0] != config.MOVEMENT_STEPS:
                continue  # amostra com tamanho inconsistente - ignora
            X.append(arr)
            y.append(letter)
        print(f"  {letter}: {len(files)} amostras")
    return np.array(X, dtype=np.float32), np.array(y)


def export_weights(model, labels_sorted, path):
    lstm = model.lstm
    fc = model.fc
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        W_ih=lstm.weight_ih_l0.detach().numpy(),
        W_hh=lstm.weight_hh_l0.detach().numpy(),
        b_ih=lstm.bias_ih_l0.detach().numpy(),
        b_hh=lstm.bias_hh_l0.detach().numpy(),
        W_fc=fc.weight.detach().numpy(),
        b_fc=fc.bias.detach().numpy(),
        hidden_size=HIDDEN_SIZE,
        classes=np.array(labels_sorted),
    )


def main():
    print("Carregando dataset de gestos:")
    X, y = load_dataset()
    if len(X) == 0:
        print(f"\nNenhuma amostra encontrada em {config.MOVEMENT_DATASET_DIR}.")
        print("Rode primeiro: python collect_movement.py")
        return

    print(f"\nTotal: {len(X)} amostras, {len(set(y))} classes, janela de {X.shape[1]} frames.")
    counts = {label: int((y == label).sum()) for label in sorted(set(y))}
    for label, n in counts.items():
        marker = "  <- poucas amostras, colete mais" if n < 25 else ""
        print(f"  {label}: {n}{marker}")

    if len(X) < 40:
        print("\nAVISO: dataset pequeno pra uma LSTM. Recomendado pelo menos "
              "~25-40 amostras por letra (gestos, não frames) antes de confiar no resultado.")

    labels_sorted = sorted(set(y.tolist()))
    label_to_idx = {label: i for i, label in enumerate(labels_sorted)}
    y_idx = np.array([label_to_idx[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test).long()

    model = LSTMClassifier(input_size=X.shape[2], hidden_size=HIDDEN_SIZE, num_classes=len(labels_sorted))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 30 == 0:
            print(f"  época {epoch + 1}/{EPOCHS} - loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).argmax(dim=1).numpy()

    print("\n=== Relatório de avaliação (conjunto de teste, 20% separado) ===")
    print(classification_report(y_test, y_pred, target_names=labels_sorted, zero_division=0))
    print("=== Matriz de confusão ===")
    print("Ordem:", labels_sorted)
    print(confusion_matrix(y_test, y_pred))

    # Retreina no dataset inteiro pro modelo final
    X_all_t = torch.from_numpy(X)
    y_all_t = torch.from_numpy(y_idx).long()
    model_final = LSTMClassifier(input_size=X.shape[2], hidden_size=HIDDEN_SIZE, num_classes=len(labels_sorted))
    optimizer = torch.optim.Adam(model_final.parameters(), lr=LEARNING_RATE)
    model_final.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model_final(X_all_t), y_all_t)
        loss.backward()
        optimizer.step()

    export_weights(model_final, labels_sorted, config.MOVEMENT_MODEL_PATH)
    print(f"\nModelo salvo em {config.MOVEMENT_MODEL_PATH}")


if __name__ == "__main__":
    main()
