# Reconhecimento de letras com movimento

## Captura e treino por quadro

O coletor padrão oferece as 26 letras, incluindo H, J, K, X e Z. Para essas
cinco, pause numa pose característica do gesto e capture com Espaço. Pressione
S para gravar as novas linhas no mesmo `dataset/landmarks.csv`. Cada linha
continua contendo 63 valores e o rótulo. Use `python train_classifier.py` para
treinar com as classes que têm amostras e `python main.py` para executar.
Um dataset com as 21 classes originais continua válido.

A MLP por quadro pode distinguir poses características. Para H, J, K, X e Z,
ela pede 0,10 s de estabilidade na configuração atual. Ajuste
`MOVEMENT_MIN_STABLE_SECONDS` em `config_override.json`. As outras letras usam
`MIN_STABLE_SECONDS` (0,80 s por padrão). A voz de H, K, X e Z espera mais
`MOVEMENT_TTS_DELAY_SECONDS` após a confirmação (0,50 s por padrão), inclusive
nos modos de sequência. J fala na confirmação. Se dois gestos usam a mesma pose e diferem pela ordem ou direção
do movimento, use a classificação de sequências para compará-los.

No coletor por quadro, use a classe **Nula (_)** para salvar poses de mãos
visíveis que não formam letras, como as transições que mais causam falsos
positivos. As amostras entram no mesmo CSV. Depois de treinar de novo, quando
`_` for a previsão mais provável, o aplicativo descarta o quadro sem mostrar
nem falar uma letra. Os pesos e os CSVs sem essa classe continuam válidos.

## Alternativa experimental em Python: MLP temporal

```powershell
.\venv\Scripts\python.exe collect_landmarks.py --movement
.\venv\Scripts\python.exe train_classifier.py --movement
.\venv\Scripts\python.exe main.py --movement-mlp
```

Use A/D ou as setas para escolher H, J, K, X, Z ou `_`. Segure Espaço
enquanto faz o gesto, solte para parar e pressione S para salvar. Use `_` para gravar
movimentos que não formam letras, como transições entre poses. O treino exige
pelo menos dez sequências por classe usada, incluindo `_`. Você pode começar
com uma letra e a rejeição. Somente letras coletadas entram no modelo.

A MLP recebe os 20 quadros em ordem, com 65 valores por quadro, e confirma o
resultado ao fim do gesto. Ela usa `dataset/movement_trajectory/`, o mesmo
formato do coletor de trajetória usado pela LSTM. O treino salva apenas
`modelo/movement_mlp.npz`. O CSV e o modelo estático continuam nos caminhos
originais. As sequências antigas de 63 valores continuam no fluxo LSTM com
`--legacy`. Use o coletor de trajetória para preparar dados para a MLP temporal.

Para comparar as duas alternativas, use a mesma coleta e sessões de teste
separadas. A MLP de movimento roda sem PyTorch. `main.py --movement` continua
ativando a LSTM. Se os pesos selecionados estiverem ausentes, o aplicativo
continua reconhecendo letras estáticas e registra o motivo no `app.log`.

## LSTM

Uma LSTM pequena (32 unidades) aproveita o código já existente e pode ser
executada com as mesmas matrizes em Python e Rust. Uma GRU teria menos portões
e parâmetros, e uma rede convolucional temporal também modela a ordem dos
quadros com operações paralelas. Mantivemos a LSTM como referência até haver
dados de mais sessões e pessoas para comparar os modelos. Consulte a
[implementação LSTM do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
e o [estudo de convoluções temporais](https://arxiv.org/abs/1803.01271).

Durante o reconhecimento, o segmentador mede mudanças na pose e no pulso por segundo. Ele preserva alguns
quadros anteriores ao início, termina após 0,18 s de quietude ou 2 s de captura
e descarta a janela quando perde a mão. O gesto completo vira 20 passos com
63 valores de pose e 2 de deslocamento do pulso relativo ao primeiro quadro.
A classe `_` representa outros movimentos. Ela e resultados abaixo da
confiança mínima são descartados. O resultado aceito é confirmado no fim do
gesto e usa o mesmo intervalo de repetição da voz estática.

## Modelo experimental distribuído

Há pesos para H e J, treinados com 177 e 263 sequências legadas e 200 exemplos
negativos aproximados por interpolação de poses estáticas. As sequências
antigas têm 63 valores por quadro e perderam a trajetória do pulso. O modelo
legado continua compatível com esse formato. A coleta nova usa 65 valores e
fica em outra pasta. O teste local separou os últimos 20% de cada classe por
nome de arquivo. A base não registra a identidade ou a sessão do sinalizante.
Os negativos aproximados não substituem transições capturadas da webcam.

Para experimentar, extraia o pacote e abra `Iniciar-com-movimento.cmd`. No
código, use `python main.py --movement` ou acrescente `--movement` ao executável
Rust. A interface Rust informa quais letras o modelo carregado contém.

## Coletar e treinar H, J, K, X e Z com trajetória

```powershell
.\venv\Scripts\python.exe collect_movement.py
.\venv\Scripts\python.exe train_movement_classifier.py --report test-output/movement-validation.json
```

No coletor, use A/D ou as setas para mudar a classe. Segure **Espaço** durante
o gesto, solte para encerrar e pressione **S** para guardar os landmarks
em `dataset/movement_trajectory/`. Use `_` para movimentos comuns que não
correspondem a uma letra. A coleta guarda coordenadas, sem imagens da câmera.
Treine com pelo menos 10 exemplos por classe e por negativos. Para avaliar a
qualidade ao vivo, reserve sessões com outras pessoas, velocidades, posições,
câmeras e iluminação. O mínimo aceito pelo script serve para iniciar o treino.

O treinamento usa PyTorch CPU e scikit-learn. O executável Python lê
`modelo/movement_model.npz`; para Rust, rode `tools/export_models.py` com
Python/NumPy para gerar `modelo/movement_model.json`. Gere os áudios ausentes
com `tools/prepare_audio.py --generate HJKXZ` antes de empacotar o Rust. Use
`--legacy` no treino somente para reaproveitar o conjunto inicial de 63 valores.
