"""
Strategy Selector
Recommends optimal options strategy based on market direction and volatility environment.
"""
from loguru import logger

class StrategySelector:
    def determine_strategy(self, direction: str, iv_rank: float) -> str:
        """Determines options strategy based on direction and IV Rank"""
        if direction == "Bullish":
            if iv_rank > 50.0:
                return "Bull Put Spread"
            else:
                return "Bull Call Spread"
        elif direction == "Bearish":
            if iv_rank > 50.0:
                return "Bear Call Spread"
            else:
                return "Bear Put Spread"
        elif direction == "Neutral":
            if iv_rank > 50.0:
                return "Iron Condor"
            else:
                return "Calendar Spread"
        else:
            logger.warning(f"Unknown direction '{direction}', defaulting to Hold")
            return "Hold"
