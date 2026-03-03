import matplotlib.pyplot as plt


def plot_graph_closing(ticker: str, series, label):
    plt.figure()
    plt.plot(series)
    plt.title(f"{label} - {ticker}")
    plt.xlabel("Data")
    plt.ylabel("Preço de fechamento")
    plt.show()


def plot_graph_closing_prediction(ticker: str, series_original, y_test_inv, y_pred_inv):
    plt.figure()
    plt.plot(series_original.index[-len(y_test_inv) :], y_test_inv, label="Valor Real")
    plt.plot(
        series_original.index[-len(y_pred_inv) :], y_pred_inv, label="Valor previsto"
    )
    plt.title(f"Comparativo de previsão e real da ação {ticker} utilizando LSTM")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.legend()
    plt.show()
