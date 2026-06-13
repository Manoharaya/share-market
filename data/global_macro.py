"""
Global Macro Signals
Fetches SGX Nifty, Crude Brent, USD/INR, FOMC/RBI dates, and FII/DII data.
"""
import urllib.request
import yfinance as yf
import pandas as pd
from loguru import logger
from config import config

class GlobalMacro:
    def __init__(self):
        self.fred = None
        if config.FRED_API_KEY:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=config.FRED_API_KEY)
            except Exception as e:
                logger.error(f"Failed to initialize FRED API client: {e}")

    def get_gift_nifty(self) -> float:
        """Fetch GIFT Nifty futures price or fallback to spot Nifty"""
        try:
            df = yf.Ticker('NIFTY50-SGX.SI').history(period='1d')
            if not df.empty:
                return float(df['Close'].iloc[-1])
        except Exception:
            pass
        
        # Fallback to Nifty 50 spot
        try:
            df = yf.Ticker('^NSEI').history(period='1d')
            if not df.empty:
                return float(df['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Failed to fetch GIFT Nifty and Nifty 50 spot: {e}")
        return 0.0

    def get_us_markets(self) -> dict:
        """Fetch US market close prices"""
        markets = {'dow': 0.0, 'sp500': 0.0, 'nasdaq': 0.0}
        tickers = {
            'dow': '^DJI',
            'sp500': '^GSPC',
            'nasdaq': '^IXIC'
        }
        for name, ticker in tickers.items():
            try:
                df = yf.Ticker(ticker).history(period='1d')
                if not df.empty:
                    markets[name] = float(df['Close'].iloc[-1])
            except Exception as e:
                logger.error(f"Error fetching US market {name}: {e}")
        return markets

    def get_crude_oil(self) -> float:
        """Fetch Brent crude price"""
        try:
            df = yf.Ticker('BZ=F').history(period='1d')
            if not df.empty:
                return float(df['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error fetching Crude Brent: {e}")
        return 0.0

    def get_usdinr(self) -> float:
        """Fetch USD/INR rate"""
        for ticker in ['USDINR=X', 'INR=X']:
            try:
                df = yf.Ticker(ticker).history(period='1d')
                if not df.empty:
                    return float(df['Close'].iloc[-1])
            except Exception:
                pass
        logger.error("Error fetching USD/INR exchange rate")
        return 0.0

    def get_fii_dii(self) -> dict:
        """Scrape FII/DII daily data or return fallback values"""
        data = {'fii_net': 0.0, 'dii_net': 0.0}
        url = 'https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Failed to scrape FII/DII daily data: {e}. Returning fallback values.")
        return data

    def get_signals(self) -> dict:
        """Assembles all global macro signals into a single dictionary"""
        signals = {
            'gift_nifty': self.get_gift_nifty(),
            'us_markets': self.get_us_markets(),
            'crude_oil': self.get_crude_oil(),
            'usdinr': self.get_usdinr(),
            'fii_dii': self.get_fii_dii()
        }
        return signals

    def get_macro_indicators(self) -> dict:
        """Fetch daily changes of Crude Oil and USD/INR, and interest rates (used by system pipeline)"""
        indicators = {
            'crude_oil_change': 0.0,
            'usdinr_change': 0.0,
            'fed_rate': 5.25,
            'rbi_rate': 6.50
        }

        # Crude Oil Change
        try:
            crude = yf.Ticker('BZ=F').history(period='5d')
            if not crude.empty and len(crude) >= 2:
                indicators['crude_oil_change'] = ((crude['Close'].iloc[-1] - crude['Close'].iloc[-2]) / crude['Close'].iloc[-2]) * 100
        except Exception as e:
            logger.error(f"Error calculating Crude Brent percent change: {e}")

        # USDINR Change
        try:
            usdinr = yf.Ticker('USDINR=X').history(period='5d')
            if not usdinr.empty and len(usdinr) >= 2:
                indicators['usdinr_change'] = ((usdinr['Close'].iloc[-1] - usdinr['Close'].iloc[-2]) / usdinr['Close'].iloc[-2]) * 100
        except Exception as e:
            logger.error(f"Error calculating USDINR percent change: {e}")

        # FRED Fed Rate
        if self.fred:
            try:
                fed_series = self.fred.get_series('FEDFUNDS')
                if not fed_series.empty:
                    indicators['fed_rate'] = float(fed_series.iloc[-1])
            except Exception as e:
                logger.error(f"Error fetching FRED interest rates: {e}")

        return indicators
