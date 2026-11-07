# Changelog

As mudanças são registradas por versão. Datas no formato AAAA-MM-DD.

## 0.2.0 — 2026-11-07

- Aplicativo Windows nativo em Rust no branch `Rust`, sem processo Python.
- Interface com câmera, letra, confiança, progresso, seleção de câmera e voz.
- MediaPipe via modelos ONNX do OpenCV Zoo e ONNX Runtime CPU.
- Inferência MLP com pesos originais, verificada numericamente contra Python.
- Reprodução nativa do cache MP3 gerado pelo Google TTS.
- Pacote portátil com modelos, DLL, licenças e manifesto SHA-256.
- Testes de normalização, estabilização e integração com imagens locais.
- Empacotamento Python exclui bibliotecas de treinamento desnecessárias.

O extrator Rust difere do grafo de tracking Python e apresentou diferenças em
imagens de teste; ver `docs/VALIDATION.md`. Movimento permanece experimental.

## 0.1.0 — 2026-11-07

- Importação da implementação Python original, desenvolvida com auxílio do Claude.
- Reconhecimento de 21 letras estáticas com MediaPipe Hands e MLP.
- Interface PySide6, captura em thread separada, estabilização e gTTS com cache.
- Modelo estático `modelo/landmark_model.npz` incluído.
- Ferramentas preliminares de coleta, treinamento e inferência LSTM preservadas.
- README corrigido: o modelo em uso é uma MLP com pesos NumPy, não RandomForest/joblib.
- Datasets, ambientes virtuais, backups, logs e builds excluídos do Git.

Esta importação não certifica o reconhecimento de movimento: não havia pesos
LSTM salvos e os dados locais disponíveis cobriam somente H e J.
