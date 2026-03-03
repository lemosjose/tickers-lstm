from utils.graphs import plot_graph_closing, plot_graph_closing_prediction
from utils.lstm import (
    build_model,
    normalize_data,
    tomorrow_pred,
    train_evaluate,
    train_test,
)
from utils.setup_test import create_windows, setup_tickers, test_tickers_data

# achei interessante inserir o BTC_USD, já que o ciclo temporal basicamente troca de ATH em ATH
tickers: list[str] = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "IBM"]

# Visualização como no 5Y do Google
start_date: str = "2020-01-01"
end_date = None

window_size: int = 30


all_series = setup_tickers(tickers, start_date, end_date)

for serie in all_series:
    test_tickers_data(serie)

for i, serie in enumerate(all_series):
    ticker = tickers[i]

    plot_graph_closing(ticker, serie, "Historico")

    scaled_data, scaler = normalize_data(serie)
    X, Y = create_windows(scaled_data, window_size)
    X_train, X_test, Y_train, Y_test = train_test(X, Y)

    model = build_model(window_size)
    model, y_test_inv, y_pred_inv, rmse = train_evaluate(
        model, X_train, Y_train, X_test, Y_test, scaler
    )

    print(f"RMSE {ticker}: {rmse}")

    plot_graph_closing_prediction(ticker, serie, y_test_inv, y_pred_inv)

    next_price = tomorrow_pred(model, scaled_data, window_size, scaler)
    print(f"Previsao proximo dia {ticker}: {next_price:.2f}")
