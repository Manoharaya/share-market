"""
Market Data Collector (Upstox version)
Fetches live prices, options chains, PCR, and Max Pain from Upstox API.
"""
import requests
import pandas as pd
from datetime import date
from loguru import logger
from config import config

INDEX_MAPPING = {
    'NIFTY': 'NSE_INDEX|Nifty 50',
    'NIFTY 50': 'NSE_INDEX|Nifty 50',
    'BANKNIFTY': 'NSE_INDEX|Nifty Bank',
    'NIFTY BANK': 'NSE_INDEX|Nifty Bank',
    'FINNIFTY': 'NSE_INDEX|Nifty Fin Service',
    'NIFTY FIN SERVICE': 'NSE_INDEX|Nifty Fin Service',
    'INDIA VIX': 'NSE_INDEX|India VIX',
    'INDIAVIX': 'NSE_INDEX|India VIX'
}

class MarketData:
    def __init__(self):
        self.access_token = config.UPSTOX_ACCESS_TOKEN
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }

    def get_ltp(self, instrument: str) -> float:
        """Get live last traded price using Upstox API"""
        # Clean the input string and resolve mapping
        cleaned_instrument = instrument.upper().strip()
        inst_key = INDEX_MAPPING.get(cleaned_instrument, cleaned_instrument)
        
        url = f"https://api.upstox.com/v3/market-quote/ltp"
        params = {'instrument_key': inst_key}
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # The returned key will have a colon instead of a pipe
            lookup_key = inst_key.replace('|', ':')
            if 'data' in data and lookup_key in data['data']:
                return float(data['data'][lookup_key]['last_price'])
            else:
                # Look for matching substring key in response dict
                for k, v in data.get('data', {}).items():
                    if cleaned_instrument in k or lookup_key in k:
                        return float(v['last_price'])
                logger.error(f"LTP not found in response for {instrument}. Response: {data}")
                return 0.0
        except Exception as e:
            logger.error(f"Error fetching LTP for {instrument}: {e}")
            return 0.0

    def get_india_vix(self) -> float:
        """Returns the current India VIX value using Upstox API"""
        return self.get_ltp('INDIA VIX')

    def get_expiry_dates(self, instrument: str) -> list:
        """Fetch all unique expiry dates for the instrument"""
        cleaned_instrument = instrument.upper().strip()
        inst_key = INDEX_MAPPING.get(cleaned_instrument, cleaned_instrument)
        
        url = "https://api.upstox.com/v2/option/contract"
        params = {'instrument_key': inst_key}
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            if 'data' in data:
                expiries = sorted(list(set(x['expiry'] for x in data['data'] if x.get('expiry'))))
                return expiries
            return []
        except Exception as e:
            logger.error(f"Error fetching expiry dates for {instrument}: {e}")
            return []

    def get_option_chain(self, instrument: str, expiry: str) -> list:
        """Fetch full option chain for a symbol and expiry using Upstox Option Chain API"""
        cleaned_instrument = instrument.upper().strip()
        inst_key = INDEX_MAPPING.get(cleaned_instrument, cleaned_instrument)
        
        url = "https://api.upstox.com/v2/option/chain"
        params = {
            'instrument_key': inst_key,
            'expiry_date': expiry
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('data', [])
        except Exception as e:
            logger.error(f"Error fetching option chain for {instrument} at expiry {expiry}: {e}")
            return []

    def calculate_pcr(self, option_chain: list) -> float:
        """Calculate Put-Call Ratio (PCR) from option chain"""
        total_call_oi = 0.0
        total_put_oi = 0.0
        for strike in option_chain:
            if strike.get('call_options') and strike['call_options'].get('market_data'):
                total_call_oi += float(strike['call_options']['market_data'].get('oi', 0.0))
            if strike.get('put_options') and strike['put_options'].get('market_data'):
                total_put_oi += float(strike['put_options']['market_data'].get('oi', 0.0))
        
        if total_call_oi == 0.0:
            return 0.0
        return total_put_oi / total_call_oi

    def calculate_max_pain(self, option_chain: list) -> float:
        """Calculate Max Pain strike price from option chain"""
        if not option_chain:
            return 0.0
            
        strikes = []
        calls_oi = []
        puts_oi = []
        
        for strike in option_chain:
            strike_val = float(strike['strike_price'])
            strikes.append(strike_val)
            
            c_oi = 0.0
            if strike.get('call_options') and strike['call_options'].get('market_data'):
                c_oi = float(strike['call_options']['market_data'].get('oi', 0.0))
            calls_oi.append(c_oi)
            
            p_oi = 0.0
            if strike.get('put_options') and strike['put_options'].get('market_data'):
                p_oi = float(strike['put_options']['market_data'].get('oi', 0.0))
            puts_oi.append(p_oi)
            
        if not strikes:
            return 0.0
            
        min_pain = float('inf')
        max_pain_strike = strikes[0]
        
        for k in strikes:
            pain = 0.0
            for s, c_oi, p_oi in zip(strikes, calls_oi, puts_oi):
                if k > s:
                    pain += c_oi * (k - s)
                elif k < s:
                    pain += p_oi * (s - k)
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = k
                
        return max_pain_strike
