"""
Sentiment Analyzer
Calculates sentiment score from financial news.
"""
import re
from loguru import logger

BULLISH_WORDS = {
    'bullish', 'rally', 'surge', 'gain', 'rise', 'growth', 'positive', 'up', 'buy', 
    'rebound', 'jump', 'record', 'high', 'strong', 'optimistic', 'boost', 'recover', 
    'beat', 'gains', 'rallies', 'surges', 'gaining', 'rising', 'higher', 
    'profit', 'profits', 'bull', 'bulls', 'upgrade', 'optimism', 'green'
}

BEARISH_WORDS = {
    'bearish', 'crash', 'fall', 'drop', 'loss', 'slump', 'negative', 'down', 'sell', 
    'plunge', 'correction', 'slide', 'slip', 'fear', 'concern', 'low', 'weak', 
    'pessimistic', 'lower', 'panic', 'losses', 'falling', 'dropping', 'plunging', 
    'bear', 'bears', 'downgrade', 'pessimism', 'red', 'worry', 'worries', 'selloff'
}

class SentimentAnalyzer:
    def calculate_sentiment(self, headlines: list) -> float:
        """Calculates average sentiment score (-1.0 to 1.0) from news headlines"""
        if not headlines:
            return 0.0
            
        scores = []
        for text in headlines:
            if not text:
                continue
            # Clean and lowercase
            cleaned = re.sub(r'[^a-zA-Z\s]', '', text).lower()
            words = cleaned.split()
            
            pos_count = sum(1 for w in words if w in BULLISH_WORDS)
            neg_count = sum(1 for w in words if w in BEARISH_WORDS)
            
            total = pos_count + neg_count
            if total > 0:
                score = (pos_count - neg_count) / total
            else:
                score = 0.0
            scores.append(score)
            
        return sum(scores) / len(scores) if scores else 0.0
