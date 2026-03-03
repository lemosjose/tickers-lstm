import numpy as np
import yfinance as yf

# 'test' para testar download, não do teste-treino


def setup_tickers(tickers: list[str], start_date, end_date):
    series = []
    for ticker in tickers:
        # novas versões do yfinance retornam uma coluna multiindex. passei
        # muita raiva até descobrir que o mesmo código do notebook
        # quebrava aqui por isso
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if not df:
            raise ValueError("Não foi possível baixar o dataframe para os tickers")

        df.columns = df.columns.get_level_values(0)
        series.append(df[["Close"]].dropna())
    return series


def create_windows(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        Y.append(data[i + window_size])

    return np.array(X), np.array(Y)


def test_tickers_data(dataframe):
    print(dataframe.head())
    print(dataframe.tail())
