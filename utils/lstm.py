import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def build_model(window_size):
    model = Sequential()
    model.add(LSTM(50, activation="tanh", input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def train_evaluate(model, X_train, Y_Train, X_test, Y_test, scaler):
    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        Y_Train,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    Y_pred = model.predict(X_test)
    Y_test_inv = scaler.inverse_transform(Y_test)
    Y_pred_inv = scaler.inverse_transform(Y_pred)

    rmse = np.sqrt(mean_squared_error(Y_test_inv, Y_pred_inv))

    return model, Y_test_inv, Y_pred_inv, rmse


def train_test(X, Y):
    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    return X_train, X_test, Y_train, Y_test


def tomorrow_pred(model, series_scaled, window_size, scaler):
    last_window = series_scaled[-window_size:]
    last_window = last_window.reshape(1, window_size, 1)

    next_day_pred = model.predict(last_window)
    next_day_pred = scaler.inverse_transform(next_day_pred)

    return next_day_pred[0][0]
