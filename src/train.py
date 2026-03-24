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