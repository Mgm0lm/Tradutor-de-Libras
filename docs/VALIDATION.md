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

## Rust estático — 2026-11-07

- Compilação MSVC x64 em modo release concluída.
- Testes de normalização e estabilização/cooldown: 2 aprovados.
- Paridade MLP: 400 vetores (100 sintéticos e 300 locais), mesmas classes em
  400/400; maior erro absoluto nas probabilidades: 0,0000027944.
- Integração em 63 imagens locais: Python detectou 60 mãos e acertou 59 rótulos;
  Rust detectou 58 e acertou 56. **Essas imagens pertencem ao conjunto de
  treinamento: não são uma estimativa independente de acurácia.**
- Janela nativa abriu em tela cheia, câmera exibiu vídeo e landmarks ao vivo;
  aproximadamente 15 fps de câmera e 10 fps de inferência no primeiro teste.
- O primeiro teste visual identificou disposição incorreta dos painéis; corrigida
  para layouts verticais dentro das duas colunas.
- Esc encerrou o aplicativo.
- Segundo teste visual: layout correto, W detectado e L confirmado; botão +
  alterou o limiar de 0,70 para 0,75. Aproximadamente 14 fps de inferência.
- Reprodução MP3 direta falhou nesta máquina; o pacote passou a incluir WAV PCM
  derivado dos mesmos áudios gTTS, para reprodução sem dependência de codecs MP3.

Scripts reproduzíveis: `tools/verify_native.py`, `tools/verify_vision.py`,
`cargo test --manifest-path rust/Cargo.toml --release --locked`.
Os resultados detalhados ficam em `test-output/`, que não é enviado ao GitHub.

## Regressão de áudio — 2026-11-07

Sequência A → L → pausa → L confirma exatamente essas três letras. Segurar A ou
L por mais que o intervalo de repetição não gera outra fala. Se A ainda está
pendente quando L é confirmada, a fila entrega somente L. Verificado em Python
com relógio controlado e em Rust no teste de estabilização.
