"""
Historical Data Collector
Fetches historical OHLCV data for training and analysis.
"""
import yfinance as yf
import pandas as pd
from loguru import logger

YF_MAPPING = {
    'NIFTY': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'FINNIFTY': 'NIFTY_FIN_SERVICE.NS'
}

class HistoricalData:
    def get_historical_daily(self, instrument: str, period: str = "1y") -> pd.DataFrame:
        """Fetch daily historical data from Yahoo Finance"""
        ticker_symbol = YF_MAPPING.get(instrument, instrument)
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=period)
            if df.empty:
                logger.warning(f"No daily historical data returned for {instrument} ({ticker_symbol})")
            return df
        except Exception as e:
            logger.error(f"Error fetching daily historical data for {instrument}: {e}")
            return pd.DataFrame()

    def get_historical_intraday(self, instrument: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
        """Fetch intraday historical data from Yahoo Finance"""
        ticker_symbol = YF_MAPPING.get(instrument, instrument)
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"No intraday data returned for {instrument} ({ticker_symbol})")
            return df
        except Exception as e:
            logger.error(f"Error fetching intraday data for {instrument}: {e}")
            return pd.DataFrame()

    def get_vix_data(self, period: str = "1y") -> pd.DataFrame:
        """Fetch historical India VIX data from Yahoo Finance"""
        try:
            ticker = yf.Ticker('^INDIAVIX')
            df = ticker.history(period=period)
            if df.empty:
                logger.warning("No India VIX data returned")
            return df
        except Exception as e:
            logger.error(f"Error fetching India VIX data: {e}")
            return pd.DataFrame()
