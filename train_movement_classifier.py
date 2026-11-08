"""Train the shared LSTM; legacy data is opt-in and supports only present labels."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report

import config
from movement import STEPS

REJECT = "_"
SEED = 42


class LSTMClassifier(nn.Module):
    def __init__(self, features, classes):
        super().__init__()
        self.lstm = nn.LSTM(features, 32, batch_first=True)
        self.fc = nn.Linear(32, classes)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.fc(sequence[:, -1])


def load_samples(directory, features):
    data = {}
    for letter in [*config.MOVEMENT_LETTERS, REJECT]:
        sequences = []
        for path in sorted((directory / letter).glob('*.npy')):
            sample = np.load(path, allow_pickle=False)
            if sample.shape != (STEPS, features) or not np.isfinite(sample).all():
                raise ValueError(f'Amostra inválida: {path}')
            sequences.append(sample.astype(np.float32))
        if sequences:
            data[letter] = sequences
    return data


def proxy_negatives():
    """Static poses and interpolated transitions, never labels for real gestures."""
    with open(config.DATASET_CSV_PATH, encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    rng = np.random.default_rng(SEED)
    features = np.asarray([[float(row[f'{axis}{i}']) for i in range(21) for axis in 'xyz']
                           for row in rows], dtype=np.float32)
    alpha = np.linspace(0, 1, STEPS, dtype=np.float32)[:, None]
    for first, last in rng.integers(0, len(features), size=(200, 2)):
        yield features[first] * (1 - alpha) + features[last] * alpha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', action='store_true', help='Usar sequências antigas sem trajetória')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    features = 63 if args.legacy else 65
    directory = Path(config.MOVEMENT_DATASET_DIR if args.legacy else config.MOVEMENT_TRAJECTORY_DIR)
    data = load_samples(directory, features)
    if args.legacy and REJECT not in data:
        data[REJECT] = list(proxy_negatives())
    missing = [label for label in [*config.MOVEMENT_LETTERS, REJECT] if len(data.get(label, [])) < 10]
    if not args.legacy and missing:
        parser.error(f'Colete pelo menos 10 amostras por classe, incluindo negativos: {missing}')
    if REJECT not in data or len(data) < 3 or any(len(samples) < 10 for samples in data.values()):
        parser.error('São necessários pelo menos 10 exemplos por classe, duas letras e negativos.')

    labels = sorted(data)
    train, train_y, test, test_y = [], [], [], []
    for index, label in enumerate(labels):
        samples = data[label]
        cut = int(len(samples) * 0.8)
        train.extend(samples[:cut]); train_y.extend([index] * cut)
        test.extend(samples[cut:]); test_y.extend([index] * (len(samples) - cut))
    model = LSTMClassifier(features, len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006)
    counts = np.bincount(train_y)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(train_y) / (len(labels) * counts), dtype=torch.float32))
    x, y = torch.from_numpy(np.array(train)), torch.tensor(train_y)
    for _ in range(args.epochs):
        for indices in torch.randperm(len(y)).split(64):
            optimizer.zero_grad()
            loss = criterion(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(np.array(test))).argmax(dim=1).numpy()
    print(classification_report(test_y, prediction, target_names=labels, zero_division=0))
    report = {'features': features, 'classes': labels, 'counts': {label: len(samples) for label, samples in data.items()},
              'test': classification_report(test_y, prediction, target_names=labels, output_dict=True, zero_division=0),
              'split': 'last 20% of each class, sorted by filename', 'proxy_negatives': args.legacy}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    path = Path(config.MOVEMENT_MODEL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W_ih=model.lstm.weight_ih_l0.detach().numpy(), W_hh=model.lstm.weight_hh_l0.detach().numpy(),
             b_ih=model.lstm.bias_ih_l0.detach().numpy(), b_hh=model.lstm.bias_hh_l0.detach().numpy(),
             W_fc=model.fc.weight.detach().numpy(), b_fc=model.fc.bias.detach().numpy(),
             hidden_size=32, classes=np.array(labels), feature_count=features, steps=STEPS)
    print(f'Modelo salvo em {path}. Classes: {labels}')


if __name__ == '__main__':
    main()
