from datetime import datetime


import yfinance as yf
from pandas_datareader import data as pdr
# Override yfinance with pandas_datareader's data
yf.pdr_override()

def download_stock_data(stocks, start, end):
    """Download stock data for given stocks between start and end dates."""
    stock_data = {}
    for stock in stocks:
        stock_data[stock] = yf.download(stock, start, end)
        stock_data[stock]["company_name"] = stock
    return stock_data

stock_list = ["BABA", "GOOGL", "BRK-A", "BRK-B", "AD.AS"]
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

stock_data = download_stock_data(stock_list, start, end)
for stock_ticker in stock_list:
    if stock_data[stock_ticker].empty:
        raise ValueError(f"No data found for {stock_ticker}")
    else:
        print(stock_data[stock_ticker].head())