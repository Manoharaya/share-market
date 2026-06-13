"""
Signal Scorer
Combines technical, options, macro, and sentiment indicators to score direction.
"""
from loguru import logger

class SignalScorer:
    def calculate_score(
        self,
        technical_row: dict,
        pcr: float,
        iv_rank: float,
        vix: float,
        macro_data: dict,
        sentiment_score: float
    ) -> tuple:
        """
        Combines multiple features into direction, score (0-100), and confidence (0.0-1.0).
        
        Returns:
            (direction: str, conviction_score: int, confidence: float)
        """
        try:
            # 1. Technical Indicators (Max points: 45)
            tech_points = 0
            
            # MACD Crossover
            macd = technical_row.get('Macd', 0.0)
            macd_sig = technical_row.get('Macd_Signal', 0.0)
            if macd > macd_sig:
                tech_points += 15
            else:
                tech_points -= 15
                
            # RSI
            rsi = technical_row.get('Rsi', 50.0)
            if rsi < 30.0:  # Oversold rebound
                tech_points += 15
            elif rsi > 70.0:  # Overbought pullback
                tech_points -= 15
            elif 50.0 <= rsi <= 65.0:  # Bullish momentum
                tech_points += 10
            elif 35.0 <= rsi < 50.0:  # Bearish momentum
                tech_points -= 10
                
            # Moving Averages Trend
            close_price = technical_row.get('Close', 0.0)
            ema_20 = technical_row.get('Ema_20', 0.0)
            ema_50 = technical_row.get('Ema_50', 0.0)
            
            if close_price > ema_20 > ema_50:
                tech_points += 15
            elif close_price < ema_20 < ema_50:
                tech_points -= 15

            # 2. Options Metrics (Max points: 25)
            opt_points = 0
            if pcr > 1.15:
                opt_points += 20
            elif pcr < 0.75:
                opt_points -= 20
            elif pcr > 1.0:
                opt_points += 10
            elif pcr < 0.85:
                opt_points -= 10

            # 3. Macro Factors (Max points: 15)
            macro_points = 0
            # Brent Crude Falling is positive for Indian economy
            crude_change = macro_data.get('crude_oil_change', 0.0)
            if crude_change < 0.0:
                macro_points += 7.5
            else:
                macro_points -= 7.5
                
            # USDINR Falling (INR strengthening) is bullish for Indian equities
            usdinr_change = macro_data.get('usdinr_change', 0.0)
            if usdinr_change < 0.0:
                macro_points += 7.5
            else:
                macro_points -= 7.5

            # 4. News Sentiment (Max points: 15)
            sentiment_points = sentiment_score * 15.0  # -15 to +15

            # Calculate total points
            total_points = tech_points + opt_points + macro_points + sentiment_points
            
            # Map total points (-100 to +100) to raw score (0 to 100)
            # Center is 50.
            raw_score = 50.0 + (total_points / 2.0)
            raw_score = max(0.0, min(100.0, raw_score))
            
            # Determine direction and conviction
            if raw_score >= 60.0:
                direction = "Bullish"
                conviction_score = int(raw_score)
            elif raw_score <= 40.0:
                direction = "Bearish"
                conviction_score = int(100.0 - raw_score)
            else:
                direction = "Neutral"
                conviction_score = int(50.0 + abs(raw_score - 50.0))

            # Consensus-based confidence calculation
            factors = [tech_points, opt_points, macro_points, sentiment_points]
            active_factors = [f for f in factors if f != 0]
            
            if not active_factors:
                confidence = 0.5
            else:
                # Count how many agree with the final direction
                if direction == "Bullish":
                    agree_count = sum(1 for f in active_factors if f > 0)
                elif direction == "Bearish":
                    agree_count = sum(1 for f in active_factors if f < 0)
                else:
                    # For neutral, agreement is having scores close to zero
                    agree_count = sum(1 for f in active_factors if abs(f) <= 10.0)
                
                confidence = agree_count / len(active_factors)
                # Keep confidence in reasonable bound [0.3 to 1.0]
                confidence = max(0.3, min(1.0, confidence))

            return direction, conviction_score, confidence

        except Exception as e:
            logger.error(f"Error in signal scoring: {e}")
            return "Neutral", 50, 0.5
