"""
Options Features
Computes IV Rank, ATM strikes, Black-Scholes IV, and GEX metrics.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger

STRIKE_DIFF_MAP = {
    'NIFTY': 50.0,
    'NIFTY 50': 50.0,
    'BANKNIFTY': 100.0,
    'NIFTY BANK': 100.0,
    'FINNIFTY': 50.0,
    'NIFTY FIN SERVICE': 50.0
}

def bs_price(S, K, T, r, sigma, option_type='CE'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'CE' or option_type == 'C':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    except Exception:
        return 0.0

def compute_iv(price, spot, strike, expiry_days, rate, option_type='CE') -> float:
    """Computes Black-Scholes Implied Volatility using the bisection search method"""
    T = expiry_days / 365.0
    if T <= 0 or price <= 0 or spot <= 0 or strike <= 0:
        return 0.0
        
    low = 0.0001
    high = 5.0
    for _ in range(50):
        mid = (low + high) / 2.0
        mid_price = bs_price(spot, strike, T, rate, mid, option_type)
        if abs(mid_price - price) < 1e-4:
            return mid
        if mid_price < price:
            low = mid
        else:
            high = mid
    return mid

def calculate_gamma(S, K, T, r, sigma) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except Exception:
        return 0.0

def compute_iv_rank(current_iv: float, historical_ivs: list) -> float:
    """Calculates IV Rank (0 to 100)"""
    if not historical_ivs:
        return 50.0
    min_iv = min(historical_ivs)
    max_iv = max(historical_ivs)
    if max_iv == min_iv:
        return 0.0
    rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100.0
    return max(0.0, min(100.0, rank))

def iv_rank_for_vix(vix_value: float, vix_history_series: pd.Series) -> float:
    """Convenience wrapper for calculating IV Rank using VIX series data"""
    if vix_history_series is None or vix_history_series.empty:
        return 50.0
    history_list = vix_history_series.dropna().tolist()
    return compute_iv_rank(vix_value, history_list)

def compute_gex(df: pd.DataFrame, spot: float, expiry_days: float, rate: float = 0.065) -> float:
    """Calculates aggregate Gamma Exposure (GEX) from the options chain DataFrame"""
    if df.empty or spot <= 0 or expiry_days <= 0:
        return 0.0
        
    total_gex = 0.0
    for _, row in df.iterrows():
        try:
            strike = float(row['strike'])
            opt_type = str(row['type']).upper().strip()
            price = float(row['ltp'])
            oi = float(row['oi'])
            
            if price <= 0 or oi <= 0:
                continue
                
            # Compute Implied Volatility
            iv = compute_iv(price, spot, strike, expiry_days, rate, opt_type)
            if iv <= 0:
                continue
                
            # Compute Gamma
            gamma = calculate_gamma(spot, spot, expiry_days / 365.0, rate, iv)
            
            # GEX Call: Spot * Gamma * OI * 100 (standard lot sizing/multiplier)
            # GEX Put: -Spot * Gamma * OI * 100
            sign = 1.0 if (opt_type == 'CE' or opt_type == 'C') else -1.0
            gex = sign * oi * gamma * spot * 100.0
            total_gex += gex
        except Exception:
            pass
            
    return total_gex

class OptionsFeatures:
    def get_atm_strike(self, ltp: float, instrument: str) -> float:
        """Find the ATM strike closest to the LTP based on instrument type"""
        cleaned = instrument.upper().strip()
        diff = STRIKE_DIFF_MAP.get(cleaned, 50.0)
        if ltp <= 0:
            return 0.0
        return round(ltp / diff) * diff

    def calculate_iv_rank(self, current_vix: float, vix_history: pd.DataFrame) -> float:
        """Standard IV Rank wrapper for index VIX daily history dataframes"""
        if vix_history.empty:
            return 50.0
        vix_history.columns = [col.title() for col in vix_history.columns]
        if 'Close' not in vix_history.columns:
            return 50.0
        return iv_rank_for_vix(current_vix, vix_history['Close'])
