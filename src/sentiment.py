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