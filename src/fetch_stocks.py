import yfinance as yf
import pandas as pd

#Fetches the list of S&P 500 ticker symbols from Wikipedia.
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data = pd.read_html(url)
    stocks = data[0]['Symbol']
    stocks = stocks.str.replace('.', '-', regex=False)
    return stocks.tolist()



