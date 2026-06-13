"""
Options Features
Computes IV Rank, ATM strikes, and options-specific metrics.
"""
import pandas as pd
from loguru import logger

STRIKE_DIFF_MAP = {
    'NIFTY': 50.0,
    'BANKNIFTY': 100.0,
    'FINNIFTY': 50.0
}

class OptionsFeatures:
    def get_atm_strike(self, ltp: float, instrument: str) -> float:
        """Find the ATM strike closest to the LTP based on instrument type"""
        diff = STRIKE_DIFF_MAP.get(instrument, 50.0)
        if ltp <= 0:
            return 0.0
        return round(ltp / diff) * diff

    def calculate_iv_rank(self, current_vix: float, vix_history: pd.DataFrame) -> float:
        """Calculate IV Rank using India VIX history over the historical period (usually 252 days)"""
        if vix_history.empty or len(vix_history) < 20:
            logger.warning("VIX history is empty or too short. Returning default IV Rank of 50.0")
            return 50.0
        
        try:
            # Standardize columns to capital letters
            vix_history.columns = [col.title() for col in vix_history.columns]
            close_prices = vix_history['Close'].astype(float)
            
            min_vix = close_prices.min()
            max_vix = close_prices.max()
            
            if max_vix == min_vix:
                return 0.0
            
            iv_rank = ((current_vix - min_vix) / (max_vix - min_vix)) * 100.0
            return float(iv_rank)
        except Exception as e:
            logger.error(f"Error calculating IV Rank: {e}")
            return 50.0
