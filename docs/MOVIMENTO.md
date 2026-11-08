# Reconhecimento de letras com movimento

Uma MLP por quadro recebe só uma posição da mão. Para distinguir sinais com
movimento, o aplicativo precisa observar a ordem das posições e o caminho do
pulso. A MLP continua responsável pelas 21 classes estáticas.

Uma LSTM pequena (32 unidades) aproveita o código já existente e pode ser
executada com as mesmas matrizes em Python e Rust. Uma GRU teria menos portões
e parâmetros, e uma rede convolucional temporal também modela a ordem dos
quadros com operações paralelas. Mantivemos a LSTM como referência até haver
dados de mais sessões e pessoas para comparar os modelos. Consulte a
[implementação LSTM do PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
e o [estudo de convoluções temporais](https://arxiv.org/abs/1803.01271).

O segmentador mede mudanças na pose e no pulso por segundo. Ele preserva alguns
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

No coletor, use A/D para mudar a classe. Faça um gesto completo e espere a
indicação **ESPAÇO para salvar**. Pressione Espaço para guardar os landmarks
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
