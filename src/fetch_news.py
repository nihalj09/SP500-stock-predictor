from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('NEWS_API_KEY')
newsapi = NewsApiClient(api_key=api_key)
