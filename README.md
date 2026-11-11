# Tradutor de Libras — Alfabeto

Aplicativo Windows que reconhece configurações de mão do alfabeto de Libras
pela webcam, mostra a letra e a pronuncia em português. A versão original foi
escrita em Python com auxílio do Claude. Este projeto reconhece letras isoladas;
não traduz frases ou a gramática de Libras.

## Estado e letras suportadas

A versão Python usa **MediaPipe Hands + MLP**, com inferência NumPy. O modelo
por quadro incluído reconhece 24 letras, incluindo J, X e Z. O coletor oferece
as 26 letras no mesmo CSV. Amostras rotuladas `_` acrescentam uma classe de
rejeição para poses que não representam letras quando o modelo é treinado de novo.
Há também um modelo LSTM experimental para **H e J**, ativado sob demanda.
Na LSTM distribuída, K, X e Z ainda não têm pesos treinados. Veja [movimento](docs/MOVIMENTO.md),
[validação](docs/VALIDATION.md) e [changelog](CHANGELOG.md).

## Como funciona

1. OpenCV captura e espelha o vídeo da câmera, em uma thread própria.
2. MediaPipe localiza uma mão e extrai 21 pontos com coordenadas x, y e z.
3. `normalize_landmarks` subtrai o pulso e divide pela maior distância ao pulso,
   produzindo 63 valores. Coleta, treinamento e inferência usam a mesma função.
4. A MLP aplica camadas densas, ReLU e softmax. Os pesos são lidos de NPZ com
   `allow_pickle=False`; scikit-learn não é necessário para inferir.
5. O aplicativo aplica as regras de confiança e estabilidade antes de
   confirmar a letra e iniciar a espera de exibição ou de fala.
6. A interface exibe vídeo, esqueleto, letra e progresso. O gTTS gera o áudio
   a partir do texto da letra e o aplicativo toca o cache local.

## Executar no Windows

Use Python 3.12 x64. No PowerShell, na pasta do projeto:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.\venv\Scripts\python.exe main.py
```

O modelo `modelo/landmark_model.npz` está incluído. MediaPipe 0.10.21 é fixado
porque o código usa `mp.solutions.hands`. Instalar indiscriminadamente a versão
mais recente não garante compatibilidade. `requirements.txt` preserva as
orientações originais de desenvolvimento; use o arquivo runtime para executar.

## Interface

- Vídeo à esquerda e letra reconhecida à direita, em tela cheia.
- Seletor de câmera e opção de dessincronizar vídeo e inferência.
- Acima do vídeo, os contadores mostram separadamente os quadros por segundo
  da câmera e da IA. Use `"SHOW_FPS": false` no JSON para ocultá-los.
- Botões +/− alteram o limiar de confiança (padrão 0,70).
- Azul indica detecção; verde indica confirmação.
- A opção de voz habilita/desabilita a fala; **Repetir** toca a última letra.
- **Esc** ou **Sair** fecha o aplicativo.

Boa iluminação, mão inteira visível e distância adequada ajudam a detecção.
A confiança é uma saída do modelo, não uma garantia de acerto.

## Regras de detecção, exibição e fala

| Regra | Padrão e comportamento |
|---|---|
| Limiar de confiança | `CONF_THRESHOLD_DEFAULT` começa em 0,70 e pode ser ajustado pelos botões +/−. Sem mão, com confiança abaixo do limiar ou com previsão da classe nula `_`, o quadro fica sem candidato. |
| Confirmação das demais letras | `MIN_STABLE_SECONDS` exige 0,8 s de previsão contínua da mesma letra. A contagem recomeça quando a previsão muda ou some. |
| Confirmação de H, J, K, X e Z | `MOVEMENT_MIN_STABLE_SECONDS` exige 0,1 s por padrão para cada letra na MLP por quadro. Os tempos são individuais. |
| Repetição automática | Uma previsão contínua confirma uma única vez. Para confirmar a mesma letra em sequência, interrompa a previsão e aguarde pelo menos 1,8 s desde a confirmação anterior. Em Python, esse intervalo fica em `COOLDOWN_SECONDS`. |
| Exibição de X e Z | `XZ_DISPLAY_HOLD_SECONDS` mantém a letra confirmada na tela por 0,6 s. Durante esse intervalo, novas previsões são descartadas e o vídeo continua. O tempo de fala segue em paralelo. Use 0 para desativar. |
| Fala automática | `MOVEMENT_TTS_DELAY_SECONDS` conta a partir da confirmação, inclusive nos modelos de sequência. O padrão é 0,5 s para H, K, X e Z e 0 s para J. As demais letras falam sem espera adicional. Uma nova confirmação substitui a fala pendente. |
| Voz e repetição manual | Desligar a voz cancela a fala pendente. Com a voz ligada, **Repetir** toca a última letra confirmada imediatamente. |

Nos modos de sequência, um gesto válido precisa de pelo menos 0,25 s de
movimento e quatro quadros. A janela encerra após 0,18 s sem movimento ou ao
chegar a 2 s.
Ela pode incluir até 0,12 s de quadros anteriores ao início, e um intervalo
acima de 0,35 s entre quadros reinicia a captura. O classificador usa
confiança mínima de 0,70 ou o limiar da interface, se for maior. A confirmação
acontece ao reconhecer a sequência, sem uma espera extra de estabilidade.
H, J e K reconhecidas por sequência podem ficar visíveis por até 0,65 s
antes de voltar à indicação por quadro. X e Z seguem a regra de exibição da tabela.

### Como ajustar

No Python, os tempos das letras com movimento ficam em `main.py`, e os
parâmetros gerais em `config.py`. Nas duas versões, copie
[`config_override.example.json`](config_override.example.json) para
`config_override.json` ao lado de `main.py` ou do executável. Edite o arquivo
e reinicie o aplicativo. O JSON tem prioridade e pode conter apenas as chaves
e letras que você quer ajustar. Arquivos antigos com um único número para o
tempo de confirmação ou fala das letras com movimento continuam aceitos e
mantêm a fala de J imediata.

A primeira geração de cada áudio requer internet. Os MP3 ficam em `tts_cache/`
e podem ser reutilizados offline. A pasta deve permitir gravação. Não há envio
de vídeo na implementação original. Logs locais ficam em `app.log`.

## Coleta e treinamento por quadro

```powershell
.\venv\Scripts\python.exe -m pip install scikit-learn
.\venv\Scripts\python.exe collect_landmarks.py
.\venv\Scripts\python.exe train_classifier.py
```

Para excluir letras de um treino, liste-as após `--edit`, separadas por
espaços. Por exemplo, `python train_classifier.py --edit X Z` ignora X e Z
nos dois CSVs e gera pesos sem essas classes. As amostras dos CSVs e a classe
nula `_` continuam disponíveis. Devem sobrar pelo menos duas classes com
amostras. Para usar os pesos por quadro no Rust, exporte-os depois do treino
com `python tools/export_models.py`.

O coletor usa setas para escolher a letra, Espaço para capturar, C para captura
contínua, S para salvar e R para desfazer. O treinamento exporta pesos MLP para
`modelo/landmark_model.npz`. Faça backup dos pesos antes de retreinar. Variações
de pessoas, sessões, iluminação e câmera são essenciais; separar frames muito
parecidos entre treino e teste pode inflar a avaliação.

## Movimento experimental

No Python, o coletor e o treinador também aceitam `--movement` para preparar
uma MLP temporal de H, J, K, X e Z. Ative esse modelo com
`python main.py --movement-mlp`. Os pesos são separados da MLP estática e
da LSTM. Veja os comandos e o formato das amostras em
[docs/MOVIMENTO.md](docs/MOVIMENTO.md#alternativa-experimental-em-python-mlp-temporal).
Use `python train_classifier.py --movement --edit X Z` para excluir X e Z
apenas da MLP temporal. Cada modelo pode reconhecer as letras que constam
nos próprios pesos.

Execute `python main.py --movement` ou abra `Iniciar-com-movimento.cmd` no
pacote que inclui a LSTM. O modelo distribuído trabalha com H e J. Uma posição
estática mantida produz uma confirmação; o botão Repetir toca a última letra.
Consulte [docs/MOVIMENTO.md](docs/MOVIMENTO.md) para coletar novos gestos,
treinar as cinco classes e gerar os mesmos pesos para Python e Rust.

## Gerar executável Python

```powershell
.\venv\Scripts\python.exe -m pip install pyinstaller==6.21.0
.\venv\Scripts\python.exe -m PyInstaller --noconfirm TradutorLibra.spec
Copy-Item modelo dist/TradutorLibra/modelo -Recurse -Force
# Opcional, se o cache de voz já foi gerado:
Copy-Item tts_cache dist/TradutorLibra/tts_cache -Recurse -Force
```

Distribua **a pasta inteira** `dist/TradutorLibra`, não apenas o EXE. Os pesos
ficam junto do executável em `modelo/`. A especificação coleta os recursos do
MediaPipe. Não exige Python instalado na máquina de destino. Teste a pasta
extraída em outro local antes de publicar uma release.

## Estrutura

| Arquivo | Responsabilidade |
|---|---|
| `main.py`, `ui_styles.py` | Interface e coordenação |
| `camera_manager.py` | Captura e compartilhamento do último frame |
| `classifier.py` | MediaPipe, MLP e LSTM opcional |
| `landmarks_utils.py` | Normalização e reamostragem |
| `stabilizer.py` | Confirmação temporal e repetição |
| `tts.py` | Geração, cache e reprodução de voz |
| `collect_*.py`, `train_*.py` | Ferramentas de dados e treinamento |
| `modelo/` | Pesos utilizados pelo aplicativo |
| `TradutorLibra.spec` | Empacotamento PyInstaller |

Datasets, imagens de treinamento, backups, ambientes virtuais e builds não são
versionados. O repositório é privado. As licenças das bibliotecas e modelos de
terceiros continuam aplicáveis; esta importação não atribui uma nova licença ao
código original nem aos datasets.

## Versão Rust

O branch `Rust` contém a implementação nativa Windows, instruções de compilação,
testes e empacotamento em [docs/RUST.md](docs/RUST.md). O aplicativo usa os mesmos
pesos da MLP e áudios, com a interface em Qt Widgets. Os resultados e as
diferenças do extrator de landmarks estão documentados em `docs/VALIDATION.md`.
