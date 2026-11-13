# Versão Rust para Windows

O branch `Rust` mantém o código Python e adiciona `rust/`. O executável nativo
não inicia Python nem depende de um ambiente virtual na máquina de destino.

## Arquitetura

- Interface: Qt Widgets 6.11.1, com os mesmos componentes, layout e folha de
  estilos usados no Python. Uma ponte C++ liga os controles ao aplicativo Rust.
- Câmera: nokhwa / Windows Media Foundation, espelhada, com último frame compartilhado.
- Pontos da mão: modelos MediaPipe do OpenCV Zoo, executados por ONNX Runtime CPU.
- Classificação: pesos originais da MLP convertidos de NPZ para JSON, sem retreino.
- Áudio: os mesmos áudios gerados pelo gTTS, convertidos para WAV PCM na preparação
  do pacote e reproduzidos pela API multimídia do Windows, sem codec MP3 externo.
  A distribuição inclui o cache; gerar novos áudios é uma etapa de preparação.

Os tempos são configurados no mesmo `config_override.json` usado pelo Python.
No pacote Rust, copie `config_override.example.json` para `config_override.json`
ao lado do executável e reinicie após editar. `MIN_STABLE_SECONDS` regula a
estabilização das poses estáticas. `MOVEMENT_MIN_STABLE_SECONDS` regula a de
H, J, K, X e Z por quadro. `MOVEMENT_TTS_DELAY_SECONDS` regula a espera após
confirmar H, K, X e Z, inclusive no modo LSTM. J e o botão **Repetir** falam
sem essa espera. Uma nova confirmação substitui a fala pendente.

Quando a MLP exportada prevê `_`, o aplicativo descarta o quadro. O modelo
incluído nesta versão tem 24 letras e ainda não possui essa classe. Para
ativar a rejeição, colete amostras `_`, treine a MLP no Python e exporte os pesos.

A extração de pontos usa uma implementação nativa de detecção de palma e recorte
orientado; não é o mesmo grafo de rastreamento da API Python `mp.solutions`.
Por isso a equivalência numérica da MLP não garante equivalência de ponta a ponta.

## Exportar a MLP treinada

O Rust 0.3.4 carrega `modelo/landmark_model.json` para reconhecer letras.
Os arquivos ONNX que acompanham o pacote executam a detecção da palma e dos
pontos da mão. Para atualizar a MLP do executável, gere o JSON a partir do
NPZ recém-treinado e copie o JSON para a pasta `modelo` ao lado do executável,
com o aplicativo fechado. O conversor `python tools/export_models.py` gera
o JSON no projeto.

Para também gerar um ONNX da própria MLP, execute:

```powershell
python -m pip install onnx==1.17.0
python tools/export_mlp_onnx.py
```

Por padrão, o script lê `modelo/landmark_model.npz` e gera
`modelo/landmark_model.onnx`. Use `-o` para mudar a saída e
`--rust-json "caminho/do/pacote/modelo/landmark_model.json"` para gravar,
no mesmo comando, o JSON aceito pela versão atual do Rust. O ONNX recebe
lotes de 63 coordenadas já normalizadas e devolve probabilidades na ordem
das classes guardadas no metadado `classes`.

## Compilar

Pré-requisitos: Rust MSVC x64, Visual Studio C++ Build Tools, Windows SDK e
Qt Base 6.11.1 para MSVC 2022 x64. O script prepara o Qt dentro do projeto.
Também é possível usar um SDK equivalente via `LIBRAS_QT_DIR`.

```powershell
.\venv\Scripts\python.exe tools/export_models.py
.\venv\Scripts\python.exe tools/fetch_rust_assets.py
.\venv\Scripts\python.exe tools/fetch_qt_sdk.py
.\venv\Scripts\python.exe tools/prepare_audio.py
cargo build --manifest-path rust/Cargo.toml --release --locked
Copy-Item tools/downloads/onnxruntime.dll .
$env:PATH = (Resolve-Path tools/downloads/qt-6.11.1/bin).Path + ';' + $env:PATH
$env:QT_PLUGIN_PATH = (Resolve-Path tools/downloads/qt-6.11.1/plugins).Path
.\rust\target\release\tradutor-libras.exe --base .
```

O script de download usa os modelos oficiais do OpenCV Zoo e ONNX Runtime 1.23.2
oficial da Microsoft. Fontes e hashes ficam em `modelo/rust-assets.json`. Licenças
dos modelos e do runtime acompanham o pacote.

## Testar e distribuir

```powershell
cargo test --manifest-path rust/Cargo.toml --locked
.\venv\Scripts\python.exe tools/verify_native.py rust/target/release/tradutor-libras.exe
.\venv\Scripts\python.exe tools/package_release.py rust 0.3.3-beta
```

Extraia o ZIP e abra `TradutorLibras.exe`, mantendo DLL, plugins Qt, modelos
e cache juntos. O pacote requer o runtime Visual C++ x64 do Windows.
O pacote experimental inclui `Iniciar-com-movimento.cmd` para ativar a LSTM
de H e J. Consulte [MOVIMENTO.md](MOVIMENTO.md) para coleta e treinamento.
As pastas de versões anteriores não são sobrescritas. `SHA256.json` permite
conferir os arquivos distribuídos.

Referências: [MediaPipe Hands](https://arxiv.org/abs/2006.10214),
[OpenCV Zoo](https://github.com/opencv/opencv_zoo),
[ONNX Runtime](https://github.com/microsoft/onnxruntime) e
[distribuição do Qt no Windows](https://doc.qt.io/qt-6/windows-deployment.html).
