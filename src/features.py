import pandas as pd
import ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Trend indicators
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)

    # Momentum indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['STOCH_K'] = stoch.stoch()
    df['STOCH_D'] = stoch.stoch_signal()
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)

    # Volatility indicators
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_mid'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = bb.bollinger_wband()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['daily_return'] = df['Close'].pct_change()

    # Volume indicators
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)

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
