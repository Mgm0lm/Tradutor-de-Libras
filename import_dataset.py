"""
Importa um dataset de FOTOS de terceiros (ex: a pasta dataset/ do repo
https://github.com/lucaaslb/cnn-libras) e converte cada imagem em um
vetor de landmarks (via MediaPipe Hands), no mesmo formato usado por
collect_landmarks.py - assim você pode combinar suas próprias amostras
com um dataset de terceiros bem maior, sem precisar reusar o modelo de
imagem original deles.

ATENÇÃO (licença): o repo cnn-libras não tem um arquivo de LICENSE
explícito, ou seja, por padrão os direitos são reservados ao autor.
Treinar um modelo pessoal pra uma apresentação acadêmica é um uso de
baixo risco, mas evite redistribuir o dataset em si (ex: não inclua a
pasta de fotos originais dentro do seu .exe) e credite o repositório
original nos créditos da apresentação.

Uso:
    1. Clone ou baixe o repo: git clone https://github.com/lucaaslb/cnn-libras
    2. Rode:
        python import_cnn_libras_dataset.py "caminho/para/cnn-libras/dataset"
    3. O script imprime os nomes de subpasta encontrados ANTES de
       processar tudo - confira se batem com o alfabeto (ver
       FOLDER_ALIASES abaixo, editável se os nomes não baterem).
"""

import argparse
import os
import sys

import cv2
import mediapipe as mp

import config
from landmarks_utils import mediapipe_landmarks_to_xyz, normalize_landmarks

# Se os nomes das subpastas do dataset de terceiros não baterem
# exatamente com as letras de config.LETTERS (ex: pasta "a_minusculo"
# em vez de "A"), adicione o mapeamento aqui: {"nome_da_pasta": "LETRA"}
FOLDER_ALIASES = {
    # "0": "A", "1": "B", ...  # exemplo, se as pastas forem numéricas
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def resolve_label(folder_name, valid_letters):
    if folder_name in FOLDER_ALIASES:
        return FOLDER_ALIASES[folder_name]
    upper = folder_name.strip().upper()
    if upper in valid_letters:
        return upper
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Caminho para a pasta com subpastas por letra (ex: cnn-libras/dataset)")
    parser.add_argument("--limit-per-class", type=int, default=None,
                         help="Máximo de imagens a processar por classe (padrão: todas). Útil pra um teste rápido antes de rodar tudo.")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Pasta não encontrada: {args.dataset_dir}")
        sys.exit(1)

    valid_letters = set(config.LETTERS.values())
    subfolders = sorted(
        d for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d))
    )

    print(f"Subpastas encontradas em {args.dataset_dir}:")
    for folder in subfolders:
        label = resolve_label(folder, valid_letters)
        status = f"-> letra '{label}'" if label else "-> IGNORADA (sem mapeamento em config.LETTERS/FOLDER_ALIASES)"
        print(f"  {folder} {status}")

    label_by_folder = {f: resolve_label(f, valid_letters) for f in subfolders}
    if not any(label_by_folder.values()):
        print("\nNenhuma subpasta foi reconhecida como uma letra válida.")
        print("Edite FOLDER_ALIASES no topo deste arquivo e rode de novo.")
        sys.exit(1)

    input("\nConfira o mapeamento acima. Pressione ENTER para processar todas as imagens (pode demorar alguns minutos)...")

    rows = []
    stats = {}  # label -> [processadas, com_mao_detectada]

    with mp.solutions.hands.Hands(
        static_image_mode=True,  # fotos independentes, não vídeo - sem tracking entre frames
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        for folder in subfolders:
            label = label_by_folder[folder]
            if label is None:
                continue

            folder_path = os.path.join(args.dataset_dir, folder)
            image_files = sorted(
                f for f in os.listdir(folder_path)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            )
            if args.limit_per_class:
                image_files = image_files[:args.limit_per_class]

            processed = 0
            detected = 0
            for filename in image_files:
                path = os.path.join(folder_path, filename)
                image = cv2.imread(path)
                if image is None:
                    continue
                processed += 1

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if not results.multi_hand_landmarks:
                    continue

                hand_landmarks = results.multi_hand_landmarks[0]
                xyz = mediapipe_landmarks_to_xyz(hand_landmarks)
                feature = normalize_landmarks(xyz)
                rows.append(list(feature) + [label])
                detected += 1

            stats[label] = [processed, detected]
            print(f"[{label}] {detected}/{processed} imagens com mão detectada.")

    if not rows:
        print("\nNenhum landmark extraído - nada foi salvo.")
        return

    os.makedirs(os.path.dirname(config.EXTERNAL_DATASET_CSV_PATH), exist_ok=True)
    is_new = not os.path.exists(config.EXTERNAL_DATASET_CSV_PATH)
    import csv
    with open(config.EXTERNAL_DATASET_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
            writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSalvo {len(rows)} amostras em {config.EXTERNAL_DATASET_CSV_PATH}")
    print("\nResumo (imagens com mão detectada / total processadas):")
    for label, (processed, detected) in sorted(stats.items()):
        pct = (detected / processed * 100) if processed else 0
        print(f"  {label}: {detected}/{processed} ({pct:.0f}%)")

    print("\nRode 'python train_classifier.py' - ele combina este arquivo "
          "automaticamente com o seu dataset pessoal (landmarks.csv), se existir.")


if __name__ == "__main__":
    main()
