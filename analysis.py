"""
Voorspelling van close-prijs met behulp van LSTM.

Door: David Moerdijk
Open source
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Set plotting style
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Override yfinance with pandas_datareader's data
yf.pdr_override()

def download_stock_data(stocks, start, end):
    """Download stock data for given stocks between start and end dates."""
    stock_data = {}
    for stock in stocks:
        stock_data[stock] = yf.download(stock, start, end)
        stock_data[stock]["company_name"] = stock
    return stock_data

def plot_closing_prices(stock_data):
    """Plot closing prices for stocks."""
    plt.figure(figsize=(15, 10))
    for i, (stock, data) in enumerate(stock_data.items(), 1):
        plt.subplot(2, 2, i)
        data['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.title(f"Closing Price of {stock}")
    plt.tight_layout()
    plt.savefig('figures/ClosingPrices.png')
    plt.clf()

def add_moving_averages(stock_data, days=[10, 20, 50]):
    """Add moving averages for specified days to stock data."""
    for stock, data in stock_data.items():
        for day in days:
            column_name = f"MA for {day} days"
            data[column_name] = data['Adj Close'].rolling(day).mean()

def plot_daily_returns(stock_data):
    """Plot daily return percentage for stocks."""
    plt.figure(figsize=(15, 10))
    for i, (stock, data) in enumerate(stock_data.items(), 1):
        data['Daily Return'] = data['Adj Close'].pct_change()
        plt.subplot(2, 2, i)
        data['Daily Return'].plot(legend=True, linestyle='--', marker='o')
        plt.title(stock)
    plt.tight_layout()
    plt.savefig('figures/DailyReturns.png')
    plt.clf()

def get_scaled_data(data, scaler):
    """Scale data using MinMaxScaler."""
    return scaler.fit_transform(data)

def prepare_data(scaled_data, look_back_window_size=60):
    """Prepare training data from scaled data."""
    X, Y = [], []
    for i in range(len(scaled_data) - look_back_window_size):
        X.append(scaled_data[i:(i + look_back_window_size), 0])
        Y.append(scaled_data[i + look_back_window_size, 0])
    return np.array(X), np.array(Y)

def build_and_train_model(x_train, y_train):
    """Build and train LSTM model."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model

def predict_closing_price(model, x_valid, scaler):
    """Predict closing price using the model."""
    predictions = model.predict(x_valid)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def plot_predictions(x_train, y_train, x_valid, predictions, Y, scaler, stock_ticker=None):
    """
    Plot the training, validation, and predicted prices to visualize the model's performance.
    
    Parameters:
    - x_train: Training data features.
    - y_train: Training data labels.
    - x_valid: Validation data features.
    - predictions: Predicted values for the validation set.
    """
    plt.figure(figsize=(16,6))
    title = 'Predictions of Closing Prices'
    if stock_ticker:
        title += f' for {stock_ticker}'
    plt.title('Model Performance')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Normalized Price', fontsize=18)
    # Assuming x_train, y_train, and x_valid are sequences of normalized prices
    train_len = len(x_train)
    valid_len = train_len + len(x_valid)
    
    # scale back the data to original form (apply inverse transform to x_valid and y_train)
    x_valid = scaler.inverse_transform(x_valid)
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
    
    plt.plot(range(train_len), y_train, label='Train')
    # plt.plot(range(train_len, valid_len), x_valid[:, 0], label='Val')
    Y = scaler.inverse_transform(Y.reshape(-1, 1))
    plt.plot(range(train_len, valid_len), Y[train_len:valid_len], label='Validation (true value)')
    plt.plot(range(train_len, valid_len), predictions, label='Predictions')
    plt.legend(loc='lower right')
    plt.savefig(f'figures/model_performance_{stock_ticker}.png')
    plt.clf()

def main():
#     1. Alibaba
# 2. Alphabet Inc. (Google)
# 3. Berkshire Hathaway (Zowel A als B)
# 4. Ahold Delhaize

    stock_list = ["BABA", "GOOGL", "BRK-A", "BRK-B", "AD.AS"]
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    stock_data = download_stock_data(stock_list, start, end)
    for stock_ticker in stock_list:
        if stock_data[stock_ticker].empty:
            raise ValueError(f"No data found for {stock_ticker}")
    
    for stock_ticker in stock_list:
        look_back_window_size = 100
        test_size = 0.2
        
        if len(stock_data) == 4:
            plot_closing_prices(stock_data)
            add_moving_averages(stock_data)
            plot_daily_returns(stock_data)

        # Example for AAPL stock
        stock_data_current_stock = stock_data[stock_ticker]['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = get_scaled_data(stock_data_current_stock, scaler)
        X, Y = prepare_data(scaled_data, look_back_window_size)
        
        # Manually splicing the data for train and validation sets
        split_index = int(len(X) * (1 - test_size))
        x_train, x_valid = X[:split_index], X[split_index:]
        y_train, y_valid = Y[:split_index], Y[split_index:]
        
        model = build_and_train_model(x_train, y_train)
        predictions = predict_closing_price(model, x_valid, scaler)
        
        
        # Assuming 'train' and 'valid' DataFrames are already defined as per the previous code block
        plot_predictions(x_train, y_train, x_valid, predictions, Y, scaler, stock_ticker)

if __name__ == "__main__":
    main()  