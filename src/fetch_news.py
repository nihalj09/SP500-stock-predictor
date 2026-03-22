from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('NEWS_API_KEY')
newsapi = NewsApiClient(api_key=api_key)

def fetch_news_for_ticker(ticker, days_back=30):
    # Calculate the end date as today and start date as days_back days ago
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    # Fetch articles from NewsAPI using the ticker as the search query
    response = newsapi.get_everything(
        q=ticker,
        from_param=start_date,
        to=end_date,
        language='en',
        sort_by='relevancy'
    )

    # Loop over the articles and extract the fields we need
    articles = []
    for article in response['articles']:
        articles.append({
            'Ticker': ticker,
            'Headline': article['title'],
            'Description': article['description'],
            'Source': article['source']['name'],
            'Published': article['publishedAt']
        })
    
    return pd.DataFrame(articles)

def fetch_news_for_multiple_tickers(tickers, days_back=30):
    # Create an empty list to hold the DataFrame for each ticker
    all_news = []
    
    # Loop over each ticker and fetch news for it
    for ticker in tickers:
        df = fetch_news_for_ticker(ticker, days_back)
        all_news.append(df)
    
    # Combine all individual DataFrames into one and return it
    combined = pd.concat(all_news, ignore_index=True)
    return combined

