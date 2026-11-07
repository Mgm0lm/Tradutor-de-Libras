# Tradutor de Libras — Alfabeto

Aplicativo Windows que reconhece configurações de mão do alfabeto de Libras
pela webcam, mostra a letra e a pronuncia em português. A versão original foi
escrita em Python com auxílio do Claude. Este projeto reconhece letras isoladas;
não traduz frases ou a gramática de Libras.

## Estado e letras suportadas

A versão Python usa **MediaPipe Hands + MLP**, com inferência NumPy. O modelo
incluído reconhece 21 classes: **A B C D E F G I L M N O P Q R S T U V W Y**.
Há código preliminar de LSTM para H, J, K, X e Z, mas a importação original não
contém pesos treinados de movimento. A ausência desses pesos mantém somente
o reconhecimento estático. Não interprete a presença do código como validação
das 26 letras. Veja [validação](docs/VALIDATION.md) e [changelog](CHANGELOG.md).

## Como funciona

1. OpenCV captura e espelha o vídeo da câmera, em uma thread própria.
2. MediaPipe localiza uma mão e extrai 21 pontos com coordenadas x, y e z.
3. `normalize_landmarks` subtrai o pulso e divide pela maior distância ao pulso,
   produzindo 63 valores. Coleta, treinamento e inferência usam a mesma função.
4. A MLP aplica camadas densas, ReLU e softmax. Os pesos são lidos de NPZ com
   `allow_pickle=False`; scikit-learn não é necessário para inferir.
5. Uma classe acima do limiar deve permanecer estável por 0,8 s. O intervalo
   entre confirmações repetidas da mesma letra é de 1,8 s.
6. PySide6 exibe vídeo, esqueleto, letra e progresso. gTTS gera áudio e pygame
   reproduz o cache local. A câmera e os frames não são enviados ao Google;
   a geração de voz envia o texto da letra ao serviço.

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
- Botões +/− alteram o limiar de confiança (padrão 0,70).
- Azul indica detecção; verde indica confirmação.
- A opção de voz habilita/desabilita a fala; **Repetir** toca a última letra.
- **Esc** ou **Sair** fecha o aplicativo.

Boa iluminação, mão inteira visível e distância adequada ajudam a detecção.
A confiança é uma saída do modelo, não uma garantia de acerto.

## Configuração e áudio

Os parâmetros estão em `config.py`. Um `config_override.json` local pode
sobrescrever valores, por exemplo:

```json
{"ENABLE_TTS": false, "CONF_THRESHOLD_DEFAULT": 0.8}
```

A primeira geração de cada áudio requer internet. Os MP3 ficam em `tts_cache/`
e podem ser reutilizados offline. A pasta deve permitir gravação. Não há envio
de vídeo na implementação original. Logs locais ficam em `app.log`.

## Coleta e treinamento estático

```powershell
.\venv\Scripts\python.exe -m pip install scikit-learn
.\venv\Scripts\python.exe collect_landmarks.py
.\venv\Scripts\python.exe train_classifier.py
```

O coletor usa setas para escolher a letra, Espaço para capturar, C para captura
contínua, S para salvar e R para desfazer. O treinamento exporta pesos MLP para
`modelo/landmark_model.npz`. Faça backup dos pesos antes de retreinar. Variações
de pessoas, sessões, iluminação e câmera são essenciais; separar frames muito
parecidos entre treino e teste pode inflar a avaliação.

## Movimento (implementação original, experimental)

```powershell
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\venv\Scripts\python.exe collect_movement.py
.\venv\Scripts\python.exe train_movement_classifier.py
```

O coletor registra sequências, reamostradas em 20 passos. A LSTM de uma camada
é treinada com PyTorch e exportada como matrizes NPZ; o aplicativo executa o
forward com NumPy. O conjunto local inicial contém somente H e J. Os dados
legados removem o deslocamento do pulso frame a frame: isso limita a descrição
da trajetória e exige revisão antes de considerar movimento validado.

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
pesos da MLP e áudios, com interface equivalente em egui. Os resultados e as
diferenças do extrator de landmarks estão documentados em `docs/VALIDATION.md`.
