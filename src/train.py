import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

from features import prepare_features
from sentiment import process_news_sentiment
from fetch_stocks import fetch_multiple_stocks
from fetch_news import fetch_news_for_multiple_tickers

def gini_impurity(left_y, right_y):
    def gini(y):
        if len(y) == 0:
            return 0
        p = np.sum(y) / len(y)
        return 1 - p**2 - (1 - p)**2
    
    n = len(left_y) + len(right_y)
    return (len(left_y) / n) * gini(left_y) + (len(right_y) / n) * gini(right_y)

