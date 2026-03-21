import yfinance as yf
import pandas as pd

#Fetches the list of S&P 500 ticker symbols from Wikipedia.
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    data = pd.read_html(url, storage_options=headers)
    stocks = data[0]['Symbol']
    stocks = stocks.str.replace('.', '-', regex=False)
    return stocks.tolist()

#Fetched historical OHLCV price data for a single stock ticker
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]
    stock_data = stock_data.reset_index()
    stock_data['Ticker'] = ticker
    stock_data = stock_data.dropna()
    return stock_data

#Fetches historical OHLCV price data for multiple stock tickers.
def fetch_multiple_stocks(tickers, start_date, end_date):
    all_stocks = []
    for ticker in tickers:
        df = fetch_stock_data(ticker, start_date, end_date)
        all_stocks.append(df)
    combined = pd.concat(all_stocks, ignore_index=True)
    return combined    