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

## LSTM experimental — 2026-11-07

- Treinamento legado H (177), J (263) e rejeição aproximada (200). O conjunto
  separado por ordem dos arquivos teve 129/129 previsões corretas, incluindo
  40 negativos gerados de poses estáticas. Sem separação por pessoa/sessão.
- Paridade Python/Rust em 80 sequências (60 locais, 20 sintéticas): mesmas
  classes em 80/80, erro máximo de probabilidade 0,000000235.
- Segmentação testada com translação do pulso, perda de mão e limite de 2 s.
- K, X e Z não foram treinadas. A precisão do movimento ao vivo ainda requer
  avaliação em sessões independentes e com exemplos negativos reais.

## Diagnóstico de generalização — 2026-11-08

- A base externa tem 72.560 quadros e 21 classes. A base local tem 1.803
  quadros e 9 classes (A, B, C, D, E, F, G, L, O). Os arquivos identificam
  a letra, sem identificar pessoa ou sessão.
- O modelo distribuído acertou 99,97% dos quadros externos e 97,84% dos
  locais. Essas bases são entradas do treinamento e os números descrevem
  ajuste aos dados conhecidos.
- Experimento separado, sem alterar os pesos distribuídos. Para cada uma
  das nove letras comuns, uma MLP de camadas (64, 32) foi treinada com 600
  quadros externos e avaliada em outros 400 quadros externos, usando a
  semente 42. Acertou 99,72% nos 3.600 externos reservados e 60,12% nos
  1.803 locais, de uma fonte que não participou do treino. L caiu de
  100% para 3,5%, e G, de 100% para 33,2%.
- A diferença indica sensibilidade à fonte. A divisão externa foi por
  quadro e pode incluir imagens correlacionadas de uma mesma pessoa. Para
  estimar a precisão ao vivo, reserve gravações por pessoa e sessão e use
  os mesmos clipes para comparar extração, classificação e estabilização.

## Latência do áudio Rust — 2026-11-08

- Antes da mudança, os 23 arquivos WAV tinham 70 a 100 ms até o início
  estimado da voz. Os arquivos novos deixam cerca de 25 ms antes desse ponto.
  A verificação confirmou que o PCM dentro da janela recortada é idêntico.
- A duração total de A caiu de 623 para 430 ms e a de L, de 863 para
  630 ms. Esses tempos descrevem os arquivos, sem medir a saída do alto-falante.
- O Windows recebe pedidos de reprodução WAV assíncronos. Uma nova letra
  pode interromper o áudio anterior. A estabilização de 800 ms continua
  como padrão e pode ser ajustada na interface.

## Interface compartilhada — 2026-11-09

- Comparação fora da tela com Qt 6.11.1, a mesma imagem RGB e os mesmos
  textos. Python e a janela nativa usada pelo Rust renderizaram 1.659 × 864
  pixels, sem diferenças no estado sem letra e com J confirmada.
- Python e Rust enviaram J para o áudio sem a espera configurada. H, K, X e Z
  continuam usando a espera. Os testes verificaram a substituição de fala pendente.
- A conversão da interface preservou as classes da MLP em 400/400 vetores
  comparados entre Python e Rust.
