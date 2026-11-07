# Registro de validação

## Estado inicial — 2026-11-07

- Python 3.12.0, Windows x64.
- Pesos MLP presentes: `modelo/landmark_model.npz`.
- Pesos LSTM ausentes.
- Dados locais de movimento: H (177 sequências), J (263 sequências).
- K, X e Z sem sequências locais; nenhum dataset é incluído no Git ou backup.
- O código legado normaliza cada frame em relação ao pulso, removendo sua
  trajetória absoluta. Dados já normalizados não permitem reconstruí-la.

Testes de unidade e comparação numérica não substituem avaliação com pessoas
sinalizando. Uma release só deve ser descrita como validada para reconhecimento
ao vivo após testar câmera, modelos, interface e áudio no executável distribuído.
