import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

from features import prepare_features
from sentiment import process_news_sentiment, load_model
from fetch_stocks import fetch_multiple_stocks
from fetch_news import fetch_news_for_multiple_tickers

def gini_impurity(left_y, right_y):
    # Compute gini for a single node: 1 - sum(p_i^2) for binary labels
    def gini(y):
        if len(y) == 0:
            return 0
        p = np.sum(y) / len(y)  # fraction of positive class
        return 1 - p**2 - (1 - p)**2

    # Weighted average of child impurities by their share of total samples
    n = len(left_y) + len(right_y)
    return (len(left_y) / n) * gini(left_y) + (len(right_y) / n) * gini(right_y)

def build_decision_tree(X, y, max_depth=10, min_samples_split=2, depth=0):
    # Leaf node: majority-class vote
    def make_leaf():
        return {"value": int(np.sum(y) > len(y) / 2)}

    # Stop recursing if max depth reached, too few samples, or node is already pure
    if depth == max_depth or len(y) < min_samples_split or len(np.unique(y)) == 1:
        return make_leaf()

    best_gini, best_feature, best_threshold, best_mask = float('inf'), None, None, None

    # Random feature subset (sqrt of total features) — standard random forest trick
    for feature in np.random.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False):
        for threshold in np.unique(X[:, feature]):
            left_mask = X[:, feature] <= threshold
            # Skip splits that don't actually divide the data
            if left_mask.sum() == 0 or left_mask.sum() == len(y):
                continue
            gini = gini_impurity(y[left_mask], y[~left_mask])
            if gini < best_gini:
                best_gini, best_feature, best_threshold, best_mask = gini, feature, threshold, left_mask

    # No valid split found (all features constant); fall back to leaf
    if best_feature is None:
        return make_leaf()

    # Recurse on left (<=threshold) and right (>threshold) partitions
    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": build_decision_tree(X[best_mask], y[best_mask], max_depth, min_samples_split, depth + 1),
        "right": build_decision_tree(X[~best_mask], y[~best_mask], max_depth, min_samples_split, depth + 1),
    }

def predict_tree(tree, x):
    # Base case — if we've reached a leaf node, return its value
    if "value" in tree:
        return tree["value"]
    
    # Walk down the correct branch based on the feature threshold
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_tree(tree["left"], x)
    else:
        return predict_tree(tree["right"], x)

def build_random_forest(X, y, n_trees=100, max_depth=10, min_samples_split=2):
    forest = []
    
    for _ in range(n_trees):
        # Randomly sample rows with replacement (bootstrapping)
        indices = np.random.choice(len(y), len(y), replace=True)
        X_sample, y_sample = X[indices], y[indices]
        
        # Build a tree on the sampled data and add it to the forest
        tree = build_decision_tree(X_sample, y_sample, max_depth, min_samples_split)
        forest.append(tree)
    
    return forest

def train(tickers):
    # Fetch stock and news data for all tickers
    stock_data = fetch_multiple_stocks(tickers, start_date='2020-01-01', end_date='2024-12-31')    
    news_data = fetch_news_for_multiple_tickers(tickers)

    # Build technical indicator features for each ticker
    all_features = []
    for ticker in tickers:
        ticker_df = stock_data[stock_data['Ticker'] == ticker].copy()
        X, y = prepare_features(ticker_df)
        X['Ticker'] = ticker
        X['Date'] = ticker_df['Date'].iloc[:len(X)].values
        X['target'] = y.values
        all_features.append(X)
    
    features_df = pd.concat(all_features, ignore_index=True)

    # Step 3 — Load FinBERT and build sentiment scores from news data
    tokenizer, model, device = load_model()
    sentiment_df = process_news_sentiment(news_data, tokenizer, model, device)

    # Merge technical features with sentiment scores on Ticker and Date
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    merged_df = features_df.merge(sentiment_df, on=['Ticker', 'Date'], how='left')
    for col in ['positive', 'negative', 'neutral']:
        merged_df[col] = merged_df[col].fillna(0)
    merged_df['sentiment'] = merged_df['sentiment'].fillna('neutral')

    # Split into features and target, then train/test split
    feature_cols = [col for col in merged_df.columns if col not in ['Ticker', 'Date', 'target', 'sentiment']]
    X = merged_df[feature_cols].values
    y = merged_df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the random forest
    print("Training random forest...")
    forest = build_random_forest(X_train, y_train)

    # Majority vote across all trees for each data point
    final_predictions = []
    for i in range(len(X_test)):
        tree_predictions = [predict_tree(tree, X_test[i]) for tree in forest]
        final_predictions.append(1 if sum(tree_predictions) > len(forest) / 2 else 0)

    print(classification_report(y_test, final_predictions))
    print(confusion_matrix(y_test, final_predictions))

    # Save the forest and scaler to the models/ folder
    os.makedirs('models', exist_ok=True)
    joblib.dump(forest, 'models/random_forest.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model saved to models/")

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    train(tickers)