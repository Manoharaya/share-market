"""
Market Data Collector (Upstox version)
Fetches live prices, options chains, PCR, and Max Pain from Upstox API.
"""
import requests
import pandas as pd
import time
from datetime import date, datetime, timedelta
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
        self._cache = {}  # key: (instrument_key, expiry), value: (timestamp, dataframe)

    def get_ltp(self, instrument: str) -> float:
        """Get live last traded price using Upstox API"""
        cleaned_instrument = instrument.upper().strip()
        inst_key = INDEX_MAPPING.get(cleaned_instrument, cleaned_instrument)
        
        url = f"https://api.upstox.com/v3/market-quote/ltp"
        params = {'instrument_key': inst_key}
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            lookup_key = inst_key.replace('|', ':')
            if 'data' in data and lookup_key in data['data']:
                return float(data['data'][lookup_key]['last_price'])
            else:
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

    def get_next_weekly_expiry(self) -> str:
        """Returns next Thursday's date in YYYY-MM-DD format"""
        today = date.today()
        # Thursday is weekday 3 (Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6)
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_thursday = today + timedelta(days=days_ahead)
        return next_thursday.strftime('%Y-%m-%d')

    def get_option_chain(self, instrument: str, expiry: str) -> pd.DataFrame:
        """Fetch option chain and return as a structured pandas DataFrame"""
        cleaned_instrument = instrument.upper().strip()
        inst_key = INDEX_MAPPING.get(cleaned_instrument, cleaned_instrument)
        
        # Check cache
        cache_key = (inst_key, expiry)
        now = time.time()
        if cache_key in self._cache:
            cached_time, cached_df = self._cache[cache_key]
            if now - cached_time < 300:  # 5 minutes refresh
                logger.info(f"Returning cached option chain for {cleaned_instrument} ({expiry})")
                return cached_df

        url = "https://api.upstox.com/v2/option/chain"
        params = {
            'instrument_key': inst_key,
            'expiry_date': expiry
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            res_json = response.json()
            raw_data = res_json.get('data', [])
            
            rows = []
            for strike in raw_data:
                strike_val = float(strike['strike_price'])
                
                # Parse Call (CE) option
                if strike.get('call_options'):
                    co = strike['call_options']
                    md = co.get('market_data', {})
                    rows.append({
                        'strike': strike_val,
                        'type': 'CE',
                        'ltp': float(md.get('ltp', 0.0)),
                        'oi': float(md.get('oi', 0.0)),
                        'volume': float(md.get('volume', 0.0)),
                        'bid': float(md.get('bid_price', 0.0)),
                        'ask': float(md.get('ask_price', 0.0))
                    })
                    
                # Parse Put (PE) option
                if strike.get('put_options'):
                    po = strike['put_options']
                    md = po.get('market_data', {})
                    rows.append({
                        'strike': strike_val,
                        'type': 'PE',
                        'ltp': float(md.get('ltp', 0.0)),
                        'oi': float(md.get('oi', 0.0)),
                        'volume': float(md.get('volume', 0.0)),
                        'bid': float(md.get('bid_price', 0.0)),
                        'ask': float(md.get('ask_price', 0.0))
                    })
            
            df = pd.DataFrame(rows)
            # Update cache
            self._cache[cache_key] = (now, df)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {instrument} at expiry {expiry}: {e}")
            return pd.DataFrame()

    def compute_pcr(self, df: pd.DataFrame) -> float:
        """Calculates Put-Call Ratio (PCR) from option chain DataFrame"""
        if df.empty:
            return 0.0
        call_oi = df[df['type'] == 'CE']['oi'].sum()
        put_oi = df[df['type'] == 'PE']['oi'].sum()
        if call_oi == 0.0:
            return 0.0
        return float(put_oi / call_oi)

    def compute_max_pain(self, df: pd.DataFrame) -> float:
        """Finds the strike price that minimizes option buyer payoff (Minimizes writer pain)"""
        if df.empty:
            return 0.0
            
        unique_strikes = sorted(df['strike'].unique())
        if not unique_strikes:
            return 0.0
            
        oi_dict = df.set_index(['strike', 'type'])['oi'].to_dict()
        
        min_pain = float('inf')
        max_pain_strike = unique_strikes[0]
        
        for k in unique_strikes:
            pain = 0.0
            for s in unique_strikes:
                # Call pain: Call OI * (K - S) if K > S
                call_oi = oi_dict.get((s, 'CE'), 0.0)
                if k > s and call_oi > 0:
                    pain += call_oi * (k - s)
                    
                # Put pain: Put OI * (S - K) if K < S
                put_oi = oi_dict.get((s, 'PE'), 0.0)
                if k < s and put_oi > 0:
                    pain += put_oi * (s - k)
                    
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = k
                
        return float(max_pain_strike)

    # Aliases to support main.py's function naming conventions
    def calculate_pcr(self, df: pd.DataFrame) -> float:
        return self.compute_pcr(df)

    def calculate_max_pain(self, df: pd.DataFrame) -> float:
        return self.compute_max_pain(df)
