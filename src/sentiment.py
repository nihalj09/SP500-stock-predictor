import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_name="ProsusAI/finbert"):
    # tokenizer converts raw text into token IDs that the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # load the FinBERT model with its classification head
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # move model to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # set model to evaluation mode — disables dropout and gradient tracking
    model.eval()
    
    return tokenizer, model, device

def aggregate_sentiment(articles, tokenizer, model, device):
    # return empty result if no articles are provided for a ticker
    if not articles:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "sentiment": "neutral"}
    
    # run sentiment analysis on each article and collect the scores
    all_scores = []
    for article in articles:
        # combine title and description for richer context
        text = f"{article.get('title', '')} {article.get('description', '')}"
        
        # skip articles that have no usable text
        if not text.strip():
            continue
        
        _, scores = analyze_sentiment(text, tokenizer, model, device)
        all_scores.append(scores)
    
    # average the scores across all articles
    avg_scores = {
        "positive": float(np.mean([s["positive"] for s in all_scores])),
        "negative": float(np.mean([s["negative"] for s in all_scores])),
        "neutral":  float(np.mean([s["neutral"]  for s in all_scores]))
    }
    
    # pick the dominant sentiment label from the averaged scores
    avg_scores["sentiment"] = max(["positive", "negative", "neutral"], key=lambda l: avg_scores[l])
    
    return avg_scores

