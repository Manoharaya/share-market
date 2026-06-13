"""
Strategy Selector
Recommends optimal options strategy based on market direction and volatility environment.
"""
from loguru import logger

class StrategySelector:
    def determine_strategy(self, direction: str, iv_rank: float, confidence: float = 0.8) -> str:
        """Determines options strategy based on direction, IV Rank, and confidence"""
        if confidence < 0.4:
            return "No Trade"
            
        direction = direction.strip().title()
        
        if direction == "Bullish":
            if iv_rank <= 35.0:
                return "Long Call"
            elif iv_rank <= 60.0:
                return "Bull Call Spread"
            else:
                return "Bull Put Spread"
                
        elif direction == "Bearish":
            if iv_rank <= 35.0:
                return "Long Put"
            elif iv_rank <= 60.0:
                return "Bear Put Spread"
            else:
                return "Bear Call Spread"
                
        elif direction == "Neutral":
            if iv_rank > 90.0:
                return "Strangle"
            elif iv_rank > 60.0:
                return "Iron Condor"
            elif iv_rank > 35.0:
                return "Iron Butterfly"
            else:
                return "Calendar Spread"
                
        else:
            return "No Trade"
