import sys
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from fetch_stocks import fetch_stock_data
from fetch_news import fetch_news_for_ticker
from features import prepare_features
from sentiment import process_news_sentiment, load_model as load_finbert

# loads the trained random forest
def load_model(model_path='models/random_forest.pkl', scaler_path='models/scaler.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model not found. Run train.py first.")

    forest = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return forest, scaler


def predict_tree(tree, x):
    # Reuse the same tree traversal logic used during training.
    if "value" in tree:
        return tree["value"]

    if x[tree["feature"]] <= tree["threshold"]:
        return predict_tree(tree["left"], x)
    return predict_tree(tree["right"], x)


def prepare_input(ticker, start_date='2020-01-01', end_date=None, news_days_back=20):
    """Fetch fresh stock and news data, compute features, and merge sentiment."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    stock_data = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    if stock_data.empty:
        raise ValueError(f"No stock data returned for ticker {ticker}.")

    feature_df, _ = prepare_features(stock_data)
    if feature_df.empty:
        raise ValueError(f"Not enough stock history to prepare features for {ticker}.")

    feature_df['Ticker'] = ticker
    feature_df['Date'] = stock_data['Date'].iloc[:len(feature_df)].values

    # Perform news sentiment processing for the same ticker.
    news_df = fetch_news_for_ticker(ticker, days_back=news_days_back)
    tokenizer, model, device = load_finbert()
    sentiment_df = process_news_sentiment(news_df, tokenizer, model, device)

    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    feature_df['Date'] = pd.to_datetime(feature_df['Date'])

    merged = feature_df.merge(sentiment_df, on=['Ticker', 'Date'], how='left')

    # If no sentiment row exists for the chosen date, default the scores to neutral.
    for col in ['positive', 'negative', 'neutral']:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)
    if 'sentiment' not in merged.columns:
        merged['sentiment'] = 'neutral'
    merged['sentiment'] = merged['sentiment'].fillna('neutral')

    # Use the most recent valid feature row from the same pipeline used in training.
    input_row = merged.iloc[[-1]].copy()
    input_row = input_row.drop(columns=['Ticker', 'Date', 'sentiment'])

    return input_row


def predict(ticker):
    forest, scaler = load_model()
    input_row = prepare_input(ticker)

    if input_row.shape[0] != 1:
        raise ValueError("prepare_input must return exactly one feature row.")

    X_scaled = scaler.transform(input_row)

    votes = [predict_tree(tree, X_scaled[0]) for tree in forest]
    positive_votes = sum(votes)
    confidence = positive_votes / len(votes)
    prediction = 1 if positive_votes > len(votes) / 2 else 0
    label = "UP" if prediction == 1 else "DOWN"

    return {
        "ticker": ticker,
        "prediction": int(prediction),
        "label": label,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    try:
        result = predict(ticker)
        print(f"Ticker: {result['ticker']}")
        print(f"Prediction: {result['prediction']} ({result['label']})")
        print(f"Confidence: {result['confidence']:.2f}")
    except Exception as exc:
        print(f"Error: {exc}")

