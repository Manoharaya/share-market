"""
Signal Scorer Module
Combines technical, options, macro, sentiment, and fundamental indicators into a composite score.
"""
from dataclasses import dataclass
from loguru import logger

@dataclass
class SignalScore:
    technical: float
    ml: float
    options: float
    sentiment: float
    macro: float
    fundamental: float
    total: float
    direction: str
    confidence: float

class SignalScorer:
    def score_technical(self, features_dict: dict) -> float:
        """Returns score between 0 and 25 based on technical indicators"""
        score = 0.0
        try:
            # Trend support (max 10)
            close = float(features_dict.get('Close', 0.0))
            ema_20 = float(features_dict.get('Ema_20', 0.0))
            ema_50 = float(features_dict.get('Ema_50', 0.0))
            if close > ema_20 > ema_50:
                score += 10.0
            elif close > ema_20:
                score += 5.0

            # Momentum support (max 8)
            macd = float(features_dict.get('Macd', 0.0))
            macd_sig = float(features_dict.get('Macd_Signal', 0.0))
            if macd > macd_sig:
                score += 8.0

            # RSI support (max 7)
            rsi = float(features_dict.get('Rsi', 50.0))
            if 50.0 <= rsi <= 65.0:
                score += 7.0
            elif rsi < 30.0:  # Oversold rebound
                score += 7.0
            elif 30.0 <= rsi < 50.0:
                score += 3.0
        except Exception as e:
            logger.error(f"Error scoring technicals: {e}")
            score = 12.5  # Neutral fallback
            
        return float(max(0.0, min(25.0, score)))

    def score_ml(self, prediction_dict: dict) -> float:
        """Scales ML ensemble probability/confidence to a score between 0 and 25"""
        prob = prediction_dict.get('probability', 0.5)
        return float(max(0.0, min(25.0, prob * 25.0)))

    def score_options(self, iv_rank: float, pcr: float, gex: float) -> float:
        """Returns score between 0 and 20 based on options metrics"""
        score = 0.0
        
        # PCR scoring (max 8)
        if pcr > 1.15:
            score += 8.0
        elif pcr > 0.95:
            score += 5.0
        elif pcr > 0.8:
            score += 3.0
            
        # IV Rank scoring (max 4)
        if iv_rank > 50.0:
            score += 4.0
        else:
            score += 2.0
            
        # GEX scoring (max 8)
        if gex >= 0:
            score += 8.0
        else:
            score += 2.0
            
        return float(max(0.0, min(20.0, score)))

    def score_sentiment(self, sentiment_score: float) -> float:
        """Returns score between 0 and 15 based on sentiment analysis"""
        # Map sentiment [-1, 1] to [0, 15]
        score = (sentiment_score + 1.0) * 7.5
        return float(max(0.0, min(15.0, score)))

    def score_macro(self, sgx_change: float, fii_flow: float, usdinr_change: float) -> float:
        """Returns score between 0 and 10 based on macro indicators"""
        score = 0.0
        if sgx_change > 0:
            score += 4.0
        if fii_flow >= 0:
            score += 3.0
        if usdinr_change < 0:  # USD/INR falling means strong Rupee (bullish)
            score += 3.0
        return float(max(0.0, min(10.0, score)))

    def score_fundamental(self, symbol: str) -> float:
        """Returns score between 0 and 5 based on fundamentals (5.0 for indices)"""
        return 5.0

    def score_all(
        self,
        features_dict: dict,
        prediction_dict: dict,
        iv_rank: float,
        pcr: float,
        gex: float,
        sentiment_score: float,
        macro_inputs: dict,  # keys: sgx_change, fii_flow, usdinr_change
        symbol: str
    ) -> SignalScore:
        """Calls all sub-scorers and returns a composite SignalScore"""
        tech = self.score_technical(features_dict)
        ml = self.score_ml(prediction_dict)
        opt = self.score_options(iv_rank, pcr, gex)
        sent = self.score_sentiment(sentiment_score)
        
        sgx = macro_inputs.get('sgx_change', 0.0)
        fii = macro_inputs.get('fii_flow', 0.0)
        usdinr = macro_inputs.get('usdinr_change', 0.0)
        mac = self.score_macro(sgx, fii, usdinr)
        
        fund = self.score_fundamental(symbol)
        
        total = tech + ml + opt + sent + mac + fund
        total = max(0.0, min(100.0, total))
        
        if total >= 60.0:
            direction = "Bullish"
        elif total <= 40.0:
            direction = "Bearish"
        else:
            direction = "Neutral"
            
        confidence = float(abs(total - 50.0) / 50.0)
        
        return SignalScore(
            technical=tech,
            ml=ml,
            options=opt,
            sentiment=sent,
            macro=mac,
            fundamental=fund,
            total=total,
            direction=direction,
            confidence=confidence
        )

    # Legacy calculate_score compatibility for main.py scheduling pipeline
    def calculate_score(
        self,
        technical_row: dict,
        pcr: float,
        iv_rank: float,
        vix: float,
        macro_data: dict,
        sentiment_score: float
    ) -> tuple:
        """System compatibility wrapper converting inputs and executing score_all"""
        # Create a default prediction dictionary to satisfy ML score block
        prob = 0.5
        # If trend is strong, assign higher default probability
        close = technical_row.get('Close', 0.0)
        ema_20 = technical_row.get('Ema_20', 0.0)
        if close > ema_20:
            prob = 0.65
        elif close < ema_20:
            prob = 0.35
            
        prediction = {'probability': prob, 'direction': 'Bullish' if prob >= 0.5 else 'Bearish'}
        
        # Assemble macro inputs
        # BZ=F change serves as crude change, USDINR change as currency change.
        crude_change = macro_data.get('crude_oil_change', 0.0)
        usdinr_change = macro_data.get('usdinr_change', 0.0)
        
        macro_inputs = {
            'sgx_change': -crude_change,  # Negative crude change is positive/bullish
            'fii_flow': 0.0,
            'usdinr_change': usdinr_change
        }
        
        # Assume positive GEX if PCR is high/bullish, else negative
        gex = 1.0 if pcr > 1.0 else -1.0
        
        # Run scorer
        sig_score = self.score_all(
            features_dict=technical_row,
            prediction_dict=prediction,
            iv_rank=iv_rank,
            pcr=pcr,
            gex=gex,
            sentiment_score=sentiment_score,
            macro_inputs=macro_inputs,
            symbol='NIFTY'
        )
        
        return sig_score.direction, int(sig_score.total), sig_score.confidence
