import numpy as np
import joblib
import os

from fetch_stocks import fetch_stock_data
from fetch_news import fetch_news_for_ticker
from features import prepare_features
from sentiment import process_news_sentiment

# loads the trained random forest
def load_model(model_path='models/random_forest.pkl', scaler_path='models/scaler.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model not found. Run train.py first.")
    
    forest = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return forest, scaler