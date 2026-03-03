# Previsão de Séries Temporais com LSTM

## Objetivo do Projeto
O objetivo deste código é construir e avaliar um modelo de rede neural Long Short-Term Memory (LSTM) para prever os preços de fechamento futuros de diversas ações de tecnologia a partir `tickers` ma biblioteca yfinance. Utilizando dados financeiros históricos, nessa aplicação, com uma janela (`start_date`) em 2020-01-01

## Aviso 

yfinance utiliza webscraping (pela dependência beautifulsoup4), a execução pelo notebook do google colab é mais simples e provavelmente funcionará com mais tranquilidade. Além dos kernels, iPython e afins também facilitarem toda a questão de dependency hell, versões de Python, etc. Utilize o repositório como uma forma mais simples de revisar o código, por favor.

E claro, é processamento (e nesse caso, especialmente de GPU) fornecido pela Google, em agradecimento, o ticker dela está na lista.

## Dependências
As seguintes bibliotecas são necessárias para executar o notebook com sucesso:
- `yfinance`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `numpy`
- `tensorflow`

Forma recomendada de teste:

- Download do notebook (no diretório `notebook`)
- Upload via Google Colab 
- Clicar em `Run All Cells`

Forma alternativa (não-recomendada por questão de dependency hell, pelo fato do tensorflow estar lockado ainda em python 3.11/3.10 ): 

- `uv python install 3.11`
- `uv venv --python 3.11`
- `uv python pin 3.11`
- `uv sync`
- `uv run main.py`


## Ativos Analisados
O notebook analisa os preços históricos de fechamento a partir de 1º de janeiro de 2020, baixando os dados pela biblioteca `yfinance`, do **yahoo**. Os ativos processados pelo modelo de previsão estão detalhados abaixo:

| Ticker | Tipo de Ativo | Descrição |
|---|---|---|
| AAPL | Ação | Apple Inc. |
| MSFT | Ação | Microsoft Corporation |
| GOOGL | Ação | Alphabet Inc. (Google) |
| BTC-USD | Criptomoeda | Bitcoin para Dólar Americano |
| IBM | Ação | International Business Machines |

## Estrutura do Código e Funções
O notebook está organizado em funções modulares para lidar com a ingestão de dados, pré-processamento, modelagem e visualização.

| Nome da Função | Descrição |
|---|---|
| setup_tickers | Baixa os dados históricos de mercado para a lista especificada de tickers utilizando a biblioteca yfinance. |
| test_tickers_data | Exibe as primeiras e últimas linhas dos conjuntos de dados baixados para verificação estrutural. |
| create_windows | Estrutura os dados de série temporal em janelas sobrepostas de 30 dias (features) e o 31º dia (alvo). |
| normalize_data | Escala o conjunto de dados para um intervalo entre 0 e 1 utilizando MinMaxScaler para otimizar a convergência do treinamento da rede neural. |
| train_test | Divide os dados estruturados cronologicamente em conjuntos de treinamento (80%) e teste (20%). |
| build_model | Inicializa um modelo Keras Sequential contendo uma camada LSTM e uma camada de saída Dense. |
| train_evaluate | Treina o modelo LSTM utilizando EarlyStopping para evitar overfitting, avalia-o no conjunto de teste e calcula a Raiz do Erro Quadrático Médio (RMSE). |
| tomorrow_pred | Utiliza o modelo treinado e a janela mais recente de 30 dias para prever o preço de fechamento do dia seguinte. |
| plot_graph_closing | Gera um gráfico de linha exibindo toda a série histórica de preços de fechamento do ativo. |
| plot_graph_closing_prediction| Plota um gráfico comparativo mostrando os valores reais versus os valores previstos para o conjunto de dados de teste. |

## Arquitetura do Modelo
A rede neural baseia-se em uma arquitetura sequencial projetada especificamente para previsão de séries temporais.
- Camada de Entrada: Aceita uma sequência de 30 preços de fechamento consecutivos.
- Camada Oculta: Uma camada LSTM configurada com 50 unidades e uma função de ativação tangente hiperbólica (tanh).
- Camada de Saída: Uma camada Dense com uma única unidade representando o preço numérico previsto.
- Compilação: O modelo é compilado utilizando o otimizador Adam e minimiza a função de perda de Erro Quadrático Médio (MSE).

## Fluxo de Execução
1. Os tickers alvo e o horizonte temporal são definidos.
2. Os preços históricos de fechamento são baixados para cada ativo.
3. Um gráfico de preço histórico base é gerado para contexto visual.
4. Os dados são normalizados, segmentados em janelas de 30 dias e divididos em subconjuntos de treinamento e teste.
5. O modelo LSTM é construído, treinado e avaliado, fornecendo uma métrica RMSE para avaliar a precisão do modelo.
6. Os valores reais de teste versus os valores previstos de teste são plotados para inspeção visual.
7. O modelo processa a última janela de dados disponível para gerar uma previsão do preço de fechamento do próximo dia útil.
