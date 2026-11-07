# Changelog

As mudanças são registradas por versão. Datas no formato AAAA-MM-DD.

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
