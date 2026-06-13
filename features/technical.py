"""
Technical Indicator Features
Computes technical indicators using TA-Lib.
"""
import talib
import pandas as pd
import numpy as np
from loguru import logger

class TechnicalFeatures:
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takes OHLCV DataFrame, returns feature DataFrame with 40+ columns"""
        if df.empty:
            logger.warning("Empty dataframe provided to TechnicalFeatures.compute")
            return df

        df = df.copy()
        df.columns = [col.title() for col in df.columns]

        if len(df) < 201:
            logger.warning(f"Dataframe length {len(df)} is too short. Need at least 201 rows for 200 EMA.")
            return df

        try:
            close = df['Close'].astype(float).values
            high = df['High'].astype(float).values
            low = df['Low'].astype(float).values
            volume = df['Volume'].astype(float).values

            # 1. Trend Features
            df['EMA_9'] = talib.EMA(close, timeperiod=9)
            df['EMA_20'] = talib.EMA(close, timeperiod=20)
            df['EMA_50'] = talib.EMA(close, timeperiod=50)
            df['EMA_100'] = talib.EMA(close, timeperiod=100)
            df['EMA_200'] = talib.EMA(close, timeperiod=200)

            # EMA Slopes
            df['EMA_9_slope'] = df['EMA_9'].diff()
            df['EMA_20_slope'] = df['EMA_20'].diff()
            df['EMA_50_slope'] = df['EMA_50'].diff()
            df['EMA_100_slope'] = df['EMA_100'].diff()
            df['EMA_200_slope'] = df['EMA_200'].diff()

            # VWAP Approximation
            tp = (df['High'] + df['Low'] + df['Close']) / 3.0
            pv = tp * df['Volume']
            rolling_pv = pv.rolling(20).sum()
            rolling_vol = df['Volume'].rolling(20).sum()
            df['VWAP'] = rolling_pv / rolling_vol
            df['VWAP'] = df['VWAP'].fillna(df['Close'])

            # 2. Momentum Features
            df['RSI_7'] = talib.RSI(close, timeperiod=7)
            df['RSI_14'] = talib.RSI(close, timeperiod=14)
            df['Rsi'] = df['RSI_14']  # System legacy alias
            df['RSI_21'] = talib.RSI(close, timeperiod=21)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'] = macd
            df['MACD_Signal'] = macd_signal
            df['MACD_Hist'] = macd_hist
            df['Macd'] = macd  # System legacy alias
            df['Macd_Signal'] = macd_signal  # System legacy alias
            df['Macd_Hist'] = macd_hist  # System legacy alias

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            df['Stoch_K'] = slowk
            df['Stoch_D'] = slowd

            # CCI
            df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

            # Williams %R
            df['Will_R'] = talib.WILLR(high, low, close, timeperiod=14)

            # 3. Volatility Features
            df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            df['Atr'] = df['ATR']  # System legacy alias

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['BB_Upper'] = upper
            df['BB_Middle'] = middle
            df['BB_Lower'] = lower
            df['BB_Width'] = (upper - lower) / middle
            df['BB_PercentB'] = (df['Close'] - lower) / (upper - lower)

            # Historical Volatility (5d, 10d, 20d)
            log_ret = np.log(df['Close'] / df['Close'].shift(1))
            df['HV_5'] = log_ret.rolling(5).std() * np.sqrt(252) * 100.0
            df['HV_10'] = log_ret.rolling(10).std() * np.sqrt(252) * 100.0
            df['HV_20'] = log_ret.rolling(20).std() * np.sqrt(252) * 100.0

            # 4. Volume Features
            df['OBV'] = talib.OBV(close, volume)
            df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            
            # Volume Ratio
            rolling_vol_mean = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = np.where(rolling_vol_mean > 0, df['Volume'] / rolling_vol_mean, 1.0)

            # 5. Return Features
            df['Return_1d'] = df['Close'].pct_change(1) * 100.0
            df['Return_3d'] = df['Close'].pct_change(3) * 100.0
            df['Return_5d'] = df['Close'].pct_change(5) * 100.0
            df['Return_10d'] = df['Close'].pct_change(10) * 100.0
            df['Return_20d'] = df['Close'].pct_change(20) * 100.0

            # Compatibility crossover flags
            df['EMA_20_gt_50'] = (df['EMA_20'] > df['EMA_50']).astype(int)
            df['EMA_50_gt_200'] = (df['EMA_50'] > df['EMA_200']).astype(int)
            df['Daily_Return'] = df['Return_1d']

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")

        return df

class TechnicalIndicators:
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """System legacy entry point calling the new TechnicalFeatures compute engine"""
        return TechnicalFeatures().compute(df)
