"""
Strike Selector
Selects target strike prices for options strategies.
"""
from loguru import logger
from features.options_features import STRIKE_DIFF_MAP

class StrikeSelector:
    def select_strikes(self, instrument: str, atm_strike: float, strategy: str, atr: float = 0.0) -> dict:
        """Selects the target strikes for the legs of the given options strategy"""
        diff = STRIKE_DIFF_MAP.get(instrument, 50.0)
        
        # Determine strike offsets based on ATR if available, else 1 strike difference
        offset = diff
        if atr > 0:
            offset = max(diff, round(atr / diff) * diff)

        legs = {}
        
        if strategy == "Bull Call Spread":
            legs['buy_call'] = atm_strike
            legs['sell_call'] = atm_strike + offset
        elif strategy == "Bear Put Spread":
            legs['buy_put'] = atm_strike
            legs['sell_put'] = atm_strike - offset
        elif strategy == "Bull Put Spread":
            legs['sell_put'] = atm_strike - offset
            legs['buy_put'] = atm_strike - (offset + diff)
        elif strategy == "Bear Call Spread":
            legs['sell_call'] = atm_strike + offset
            legs['buy_call'] = atm_strike + (offset + diff)
        elif strategy == "Iron Condor":
            legs['sell_put'] = atm_strike - offset
            legs['buy_put'] = atm_strike - (offset + diff)
            legs['sell_call'] = atm_strike + offset
            legs['buy_call'] = atm_strike + (offset + diff)
        elif strategy == "Iron Fly":
            legs['sell_call'] = atm_strike
            legs['sell_put'] = atm_strike
            legs['buy_call'] = atm_strike + offset
            legs['buy_put'] = atm_strike - offset
        elif strategy == "Calendar Spread":
            legs['sell_near_call'] = atm_strike
            legs['buy_far_call'] = atm_strike
        elif strategy == "Long Call":
            legs['buy_call'] = atm_strike
        elif strategy == "Long Put":
            legs['buy_put'] = atm_strike
        else:
            logger.warning(f"Unknown strategy: {strategy}. Defaulting to ATM Call/Put.")
            legs['buy_call'] = atm_strike
            legs['buy_put'] = atm_strike

        return legs
