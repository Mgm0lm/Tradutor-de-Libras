# Changelog

As mudanças são registradas por versão. Datas no formato AAAA-MM-DD.

## 0.3.1 — 2026-11-08 — Rust

- A fala WAV começa sem esperar o fim do áudio anterior. Desativar o TTS
  interrompe a fala em andamento.
- O cache WAV preserva a fala e reduz o silêncio no início e no fim.
- O controle **Estabilização** ajusta a confirmação de letras estáticas de
  0,45 a 1,20 s. O padrão continua em 0,80 s.
- A próxima letra estática pode começar a estabilizar enquanto o último
  gesto de movimento ainda aparece na tela.

## 0.3.0 — 2026-11-07 — experimental

- LSTM compartilhada entre Python e Rust, com pesos experimentais para H e J.
- Segmentação por tempo, parada ou duração máxima e descarte ao perder a mão.
- Coletor que preserva a trajetória do pulso, mantendo as sequências antigas.
- Classe de rejeição para movimentos que não correspondem a uma letra.
- Atalho para iniciar o modo experimental. K, X e Z aguardam amostras reais.

## 0.2.1 — 2026-11-07

- Uma posição mantida gera uma única fala. É possível repetir a letra depois
  de interromper o gesto ou usar o botão Repetir.
- A fala mais recente substitui a anterior quando ainda está pendente.
- Desativar a voz limpa a fala pendente.

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
