# Versão Rust para Windows

O branch `Rust` mantém o código Python e adiciona `rust/`. O executável nativo
não inicia Python nem depende de um ambiente virtual na máquina de destino.

## Arquitetura

- Interface: egui/eframe, preservando os dois painéis, cores, controles e Esc.
- Câmera: nokhwa / Windows Media Foundation, espelhada, com último frame compartilhado.
- Pontos da mão: modelos MediaPipe do OpenCV Zoo, executados por ONNX Runtime CPU.
- Classificação: pesos originais da MLP convertidos de NPZ para JSON, sem retreino.
- Áudio: os mesmos áudios gerados pelo gTTS, convertidos para WAV PCM na preparação
  do pacote e reproduzidos pela API multimídia do Windows, sem codec MP3 externo.
  A distribuição inclui o cache; gerar novos áudios é uma etapa de preparação.

A extração de pontos usa uma implementação nativa de detecção de palma e recorte
orientado; não é o mesmo grafo de rastreamento da API Python `mp.solutions`.
Por isso a equivalência numérica da MLP não garante equivalência de ponta a ponta.

## Compilar

Pré-requisitos: Rust MSVC x64, Visual Studio C++ Build Tools e Windows SDK.
Python/NumPy são usados somente para exportar os modelos e preparar os arquivos.

```powershell
.\venv\Scripts\python.exe tools/export_models.py
.\venv\Scripts\python.exe tools/fetch_rust_assets.py
.\venv\Scripts\python.exe tools/prepare_audio.py
cargo build --manifest-path rust/Cargo.toml --release --locked
Copy-Item tools/downloads/onnxruntime.dll .
.\rust\target\release\tradutor-libras.exe --base .
```

O script de download usa os modelos oficiais do OpenCV Zoo e ONNX Runtime 1.23.2
oficial da Microsoft. Fontes e hashes ficam em `modelo/rust-assets.json`. Licenças
dos modelos e do runtime acompanham o pacote.

## Testar e distribuir

```powershell
cargo test --manifest-path rust/Cargo.toml --locked
.\venv\Scripts\python.exe tools/verify_native.py rust/target/release/tradutor-libras.exe
.\venv\Scripts\python.exe tools/package_release.py rust 0.2.0
```

Extraia o ZIP e abra `TradutorLibras.exe`, mantendo DLL, modelos e cache juntos.
As pastas de versões anteriores não são sobrescritas. `SHA256.json` permite
conferir os arquivos distribuídos.

Referências: [MediaPipe Hands](https://arxiv.org/abs/2006.10214),
[OpenCV Zoo](https://github.com/opencv/opencv_zoo),
[ONNX Runtime](https://github.com/microsoft/onnxruntime).
