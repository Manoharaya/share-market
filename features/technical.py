"""
Technical Indicator Features
Computes technical indicators using TA-Lib or Pandas-TA.
"""
import talib
import pandas as pd
from loguru import logger

class TechnicalIndicators:
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates indicators and returns a copy of the dataframe with features added."""
        if df.empty:
            logger.warning("Empty dataframe provided to calculate_indicators")
            return df

        df = df.copy()
        # Clean column names to be standardized
        df.columns = [col.title() for col in df.columns]

        if len(df) < 50:
            logger.warning(f"Dataframe length {len(df)} is too short for technical indicator calculations")
            return df

        try:
            close = df['Close'].astype(float).values
            high = df['High'].astype(float).values
            low = df['Low'].astype(float).values

            # RSI
            df['RSI'] = talib.RSI(close, timeperiod=14)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'] = macd
            df['MACD_Signal'] = macd_signal
            df['MACD_Hist'] = macd_hist

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower

            # ATR
            df['ATR'] = talib.ATR(high, low, close, timeperiod=14)

            # EMAs (only compute 200 EMA if we have enough points, else fallback)
            df['EMA_20'] = talib.EMA(close, timeperiod=20)
            df['EMA_50'] = talib.EMA(close, timeperiod=50)
            if len(df) >= 200:
                df['EMA_200'] = talib.EMA(close, timeperiod=200)
            else:
                df['EMA_200'] = df['EMA_50']  # fallback

            # Crossover and trend indicators
            df['EMA_20_gt_50'] = (df['EMA_20'] > df['EMA_50']).astype(int)
            df['EMA_50_gt_200'] = (df['EMA_50'] > df['EMA_200']).astype(int)
            
            # Simple daily momentum (Close vs yesterday's Close)
            df['Daily_Return'] = df['Close'].pct_change() * 100

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")

        return df
