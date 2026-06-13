"""
Historical Data Collector
Fetches historical OHLCV data for training and analysis.
"""
import os
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from loguru import logger

YF_MAPPING = {
    'NIFTY': '^NSEI',
    'BANKNIFTY': '^NSEBANK',
    'FINNIFTY': 'NIFTY_FIN_SERVICE.NS'
}

class HistoricalData:
    def get_ohlcv(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch daily historical data using NSEpy with fallback to yfinance"""
        cleaned_symbol = symbol.upper().strip()
        
        # 1. Try to fetch using nsepy
        try:
            from nsepy import get_history
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            logger.info(f"Attempting to fetch {cleaned_symbol} using nsepy from {start_date} to {end_date}...")
            is_index = cleaned_symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'NIFTY 50', 'NIFTY BANK']
            df = get_history(symbol=cleaned_symbol, start=start_date, end=end_date, index=is_index)
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched {cleaned_symbol} using nsepy")
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df = df.ffill()
                return df
        except Exception as e:
            logger.warning(f"nsepy failed to fetch {cleaned_symbol}: {e}")

        # 2. Fallback to yfinance
        logger.info(f"Falling back to yfinance to fetch {cleaned_symbol}...")
        ticker_symbol = YF_MAPPING.get(cleaned_symbol, cleaned_symbol)
        
        if days <= 5:
            period = '5d'
        elif days <= 30:
            period = '1mo'
        elif days <= 180:
            period = '6mo'
        elif days <= 365:
            period = '1y'
        elif days <= 730:
            period = '2y'
        else:
            period = '5y'
            
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=period)
            if df.empty:
                logger.warning(f"No daily historical data returned for {cleaned_symbol} ({ticker_symbol})")
                return pd.DataFrame()
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.ffill()
            df = df.dropna(subset=['Close'])
            return df
        except Exception as e:
            logger.error(f"Error fetching daily historical data for {cleaned_symbol}: {e}")
            return pd.DataFrame()

    def get_historical_daily(self, instrument: str, period: str = "1y") -> pd.DataFrame:
        """Standard daily fetch mapped to yfinance for internal system modules"""
        days = 365
        if period == "2y":
            days = 730
        elif period == "5y":
            days = 1825
        return self.get_ohlcv(instrument, days=days)

    def get_historical_intraday(self, instrument: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
        """Fetch intraday historical data from Yahoo Finance"""
        cleaned_instrument = instrument.upper().strip()
        ticker_symbol = YF_MAPPING.get(cleaned_instrument, cleaned_instrument)
        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"No intraday data returned for {cleaned_instrument} ({ticker_symbol})")
                return pd.DataFrame()
            df = df.ffill()
            return df
        except Exception as e:
            logger.error(f"Error fetching intraday data for {cleaned_instrument}: {e}")
            return pd.DataFrame()

    def get_52week_iv_history(self) -> pd.DataFrame:
        """Fetch 52 weeks (1 year) of India VIX daily historical data"""
        try:
            ticker = yf.Ticker('^INDIAVIX')
            df = ticker.history(period='1y')
            if df.empty:
                logger.warning("No India VIX data returned for 52-week IV history")
                return pd.DataFrame()
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df = df.ffill()
            df = df.dropna(subset=['Close'])
            return df
        except Exception as e:
            logger.error(f"Error fetching India VIX 52-week history: {e}")
            return pd.DataFrame()

    def get_vix_data(self, period: str = "1y") -> pd.DataFrame:
        """Alias for VIX retrieval used by main pipeline"""
        return self.get_52week_iv_history()

    def cache_historical_data(self):
        """Pre-fetch and save NIFTY (2y), BANKNIFTY (1y), and FINNIFTY (1y) data to cache CSVs"""
        os.makedirs('data/cache', exist_ok=True)
        
        # NIFTY 2 years (730 days)
        logger.info("Caching NIFTY 2y daily data to CSV...")
        nifty_df = self.get_ohlcv('NIFTY', days=730)
        nifty_df.to_csv('data/cache/NIFTY_daily.csv')
        logger.info(f"NIFTY daily data saved. Shape: {nifty_df.shape}, NaNs: {nifty_df.isna().sum().sum()}")
        
        # BANKNIFTY 1 year (365 days)
        logger.info("Caching BANKNIFTY 1y daily data to CSV...")
        bank_df = self.get_ohlcv('BANKNIFTY', days=365)
        bank_df.to_csv('data/cache/BANKNIFTY_daily.csv')
        logger.info(f"BANKNIFTY daily data saved. Shape: {bank_df.shape}, NaNs: {bank_df.isna().sum().sum()}")
        
        # FINNIFTY 1 year (365 days)
        logger.info("Caching FINNIFTY 1y daily data to CSV...")
        fin_df = self.get_ohlcv('FINNIFTY', days=365)
        fin_df.to_csv('data/cache/FINNIFTY_daily.csv')
        logger.info(f"FINNIFTY daily data saved. Shape: {fin_df.shape}, NaNs: {fin_df.isna().sum().sum()}")
        
        logger.info("Historical caching completed successfully.")
