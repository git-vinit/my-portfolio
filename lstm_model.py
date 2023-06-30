import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(1, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, X_train, y_train, epochs=100, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

def predict_stock_price(model, X):
    return model.predict(X)

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    history = stock.history(period="1y")
    return history['Close'].values.reshape(-1, 1)

def get_latest_price(symbol):
    stock = yf.Ticker(symbol)
    latest_price = stock.history(period="1d")['Close'].iloc[-1]
    return latest_price
