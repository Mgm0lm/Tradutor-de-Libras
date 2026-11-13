"""Exporta uma MLP por quadro do Tradutor de Libras de NPZ para ONNX."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COUNT = 63


def build_model(data, onnx):
    helper = onnx.helper
    classes = data["classes"]
    count = int(data["n_layers"])
    if count < 1 or classes.ndim != 1 or classes.size == 0 or classes.dtype.kind != "U":
        raise ValueError("Esperados n_layers positivo e classes em um vetor de texto.")
    labels = classes.tolist()

    nodes, initializers = [], []
    previous = "landmarks"
    width = FEATURE_COUNT
    for index in range(count):
        weight_name, bias_name = f"W{index}", f"b{index}"
        weight, bias = data[weight_name], data[bias_name]
        if bias.ndim != 1 or not bias.size or weight.shape != (width, bias.size):
            raise ValueError(f"Dimensões inválidas em {weight_name} ou {bias_name}. Esperadas {width} entradas.")
        for name, values in ((weight_name, weight), (bias_name, bias)):
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"{name} deve conter valores float32 finitos.")
            initializers.append(onnx.numpy_helper.from_array(values, name=name))
        dense = f"dense_{index}"
        nodes.append(helper.make_node("Gemm", [previous, weight_name, bias_name], [dense]))
        previous = dense
        if index < count - 1:
            previous = f"relu_{index}"
            nodes.append(helper.make_node("Relu", [dense], [previous]))
        width = bias.size
    if width != len(labels):
        raise ValueError("O tamanho da saída da MLP deve coincidir com a quantidade de classes.")

    nodes.append(helper.make_node("Softmax", [previous], ["probabilities"], axis=1))
    graph = helper.make_graph(
        nodes, "libras_mlp",
        [helper.make_tensor_value_info("landmarks", onnx.TensorProto.FLOAT, ["batch", FEATURE_COUNT])],
        [helper.make_tensor_value_info("probabilities", onnx.TensorProto.FLOAT, ["batch", width])],
        initializers,
    )
    model = helper.make_model(
        graph, producer_name="Tradutor de Libras", ir_version=8,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    helper.set_model_props(model, {"classes": json.dumps(labels, ensure_ascii=False)})
    onnx.checker.check_model(model, full_check=True)
    return model, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "modelo/landmark_model.npz")
    parser.add_argument("-o", "--output", type=Path, help="Destino ONNX. Por padrão, ao lado do NPZ.")
    parser.add_argument(
        "--rust-json", type=Path,
        help="Também exporta o JSON para o Rust 0.3.4. Informe o caminho modelo/landmark_model.json do pacote.",
    )
    args = parser.parse_args()
    output = args.output or args.source.with_suffix(".onnx")
    if output.suffix.lower() != ".onnx" or (args.rust_json and args.rust_json.suffix.lower() != ".json"):
        parser.error("Use a extensão .onnx na saída e .json em --rust-json.")
    if args.source.resolve() in {p.resolve() for p in (output, args.rust_json) if p is not None}:
        parser.error("O caminho de saída deve ser diferente do arquivo de origem.")
    try:
        import onnx
    except ImportError as error:
        parser.error(f"Falha ao importar ONNX ({error}). Instale com: python -m pip install onnx==1.17.0")

    try:
        with np.load(args.source, allow_pickle=False) as data:
            model, labels = build_model(data, onnx)
        output.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output)
        print(f"ONNX: {output}")
        if args.rust_json:
            from export_models import export

            args.rust_json.parent.mkdir(parents=True, exist_ok=True)
            export(args.source, args.rust_json)
    except (OSError, KeyError, TypeError, ValueError, onnx.checker.ValidationError) as error:
        parser.error(str(error))
    print(f"Classes na ordem da saída: {', '.join(labels)}")


if __name__ == "__main__":
    main()
