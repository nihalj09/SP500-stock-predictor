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

def analyze_sentiment(text, tokenizer, model, device):
    # tokenize the input text and prepare it as a pytorch tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    # move the tokenized inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # disable gradient calculation since we are only doing inference not training
    with torch.no_grad():
        outputs = model(**inputs)
    
    # convert raw output logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # move probabilities to cpu and convert to a numpy array
    probabilities = probabilities.cpu().numpy()[0]
    
    # map probabilities to their corresponding labels
    labels = ["positive", "negative", "neutral"]
    scores = {label: float(prob) for label, prob in zip(labels, probabilities)}
    
    # return the label with the highest probability and all scores
    sentiment = max(scores, key=scores.get)
    return sentiment, scores

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

def process_news_sentiment(news_df, tokenizer, model, device):
    # store results as a list of dictionaries, one per ticker per day
    results = []
    
    # group the news dataframe by ticker and date
    grouped = news_df.groupby(["ticker", "date"])
    
    # loop through each ticker and date combination
    for (ticker, date), group in grouped:
        # convert the group of articles to a list of dictionaries
        articles = group.to_dict("records")
        
        # aggregate sentiment scores across all articles for this ticker and date
        scores = aggregate_sentiment(articles, tokenizer, model, device)
        
        # store the ticker, date and scores as a single row
        results.append({
            "ticker": ticker,
            "date": date,
            **scores
        })
    
    # convert the list of results into a dataframe
    sentiment_df = pd.DataFrame(results)
    
    # sort by ticker and date so it aligns cleanly with the features dataframe
    sentiment_df = sentiment_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    return sentiment_df