import pandas as pd
import pandas_ta as ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Trend indicators
    df['SMA_10'] = df.ta.sma(length=10)
    df['SMA_20'] = df.ta.sma(length=20)
    df['SMA_50'] = df.ta.sma(length=50)
    df['EMA_10'] = df.ta.ema(length=10)
    df['EMA_20'] = df.ta.ema(length=20)
    df['EMA_50'] = df.ta.ema(length=50)
    
    # Momentum indicators
    df['RSI'] = df.ta.rsi(length=14)
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    stoch = df.ta.stoch(k=14, d=3)
    df['STOCH_K'] = stoch['STOCHk_14_3_3']
    df['STOCH_D'] = stoch['STOCHd_14_3_3']
    df['CCI'] = df.ta.cci(length=20)
    
    # Volatility indicators
    bb = df.ta.bbands(length=20)
    df['BB_upper'] = bb['BBU_20_2.0']
    df['BB_mid'] = bb['BBM_20_2.0']
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_width'] = bb['BBB_20_2.0']
    df['ATR'] = df.ta.atr(length=14)
    df['daily_return'] = df['Close'].pct_change()
    
    # Volume indicators
    df['OBV'] = df.ta.obv()
    df['VWAP'] = df.ta.vwap()
    df['CMF'] = df.ta.cmf(length=20)
    df['MFI'] = df.ta.mfi(length=14)
    
    # Lag features
    df['return_lag1'] = df['daily_return'].shift(1)
    df['return_lag2'] = df['daily_return'].shift(2)
    df['return_lag3'] = df['daily_return'].shift(3)
    
    df.dropna(inplace=True)
    
    return df

def prepare_features(df: pd.DataFrame):
    # Add all technical indicators to the DataFrame
    df = add_technical_indicators(df)
    
    # Target: 1 if price is higher 10 days from now, 0 if lower
    df['target'] = (df['Close'].shift(-10) > df['Close']).astype(int)
    
    # Drop last 10 rows since they have no valid future price to compare against
    df = df.dropna()
    
    # Explicitly define which columns are features
    feature_cols = [
        'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_10', 'EMA_20', 'EMA_50',
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
        'STOCH_K', 'STOCH_D', 'CCI',
        'BB_upper', 'BB_mid', 'BB_lower', 'BB_width',
        'ATR', 'daily_return',
        'OBV', 'VWAP', 'CMF', 'MFI',
        'return_lag1', 'return_lag2', 'return_lag3'
    ]
    
    # X = input features, y = target label
    X = df[feature_cols]
    y = df['target']
    
    return X, y