"""
Global Macro Signals
Fetches SGX Nifty, Crude Brent, USD/INR, FOMC/RBI dates, and FII/DII data.
"""
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

    def get_macro_indicators(self) -> dict:
        """Fetch crude, USDINR daily percent changes, and interest rates"""
        indicators = {
            'crude_oil_change': 0.0,
            'usdinr_change': 0.0,
            'fed_rate': 5.25,  # Fallback standard rate
            'rbi_rate': 6.50   # Fallback standard rate
        }

        # 1. Crude Oil (Brent)
        try:
            crude = yf.Ticker('BZ=F').history(period='5d')
            if not crude.empty and len(crude) >= 2:
                indicators['crude_oil_change'] = ((crude['Close'].iloc[-1] - crude['Close'].iloc[-2]) / crude['Close'].iloc[-2]) * 100
        except Exception as e:
            logger.error(f"Error fetching Crude Brent price: {e}")

        # 2. USDINR exchange rate
        try:
            usdinr = yf.Ticker('USDINR=X').history(period='5d')
            if not usdinr.empty and len(usdinr) >= 2:
                indicators['usdinr_change'] = ((usdinr['Close'].iloc[-1] - usdinr['Close'].iloc[-2]) / usdinr['Close'].iloc[-2]) * 100
        except Exception as e:
            logger.error(f"Error fetching USDINR rate: {e}")

        # 3. FRED Interest Rates (Federal Funds Rate)
        if self.fred:
            try:
                fed_series = self.fred.get_series('FEDFUNDS')
                if not fed_series.empty:
                    indicators['fed_rate'] = float(fed_series.iloc[-1])
            except Exception as e:
                logger.error(f"Error fetching FRED interest rates: {e}")

        return indicators
