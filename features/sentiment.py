"""
Sentiment Analyzer
Calculates sentiment score from financial news using FinBERT.
"""
from loguru import logger
from data.news_collector import SentimentAnalyzer as FinBERTSentiment

class SentimentAnalyzer:
    def __init__(self):
        self.finbert = FinBERTSentiment()

    def calculate_sentiment(self, headlines: list) -> float:
        """Calculates average sentiment score (-1.0 to 1.0) using FinBERT classification"""
        if not headlines:
            return 0.0
            
        try:
            results = self.finbert._nlp(headlines)
            
            total_score = 0.0
            for res in results:
                label = res['label'].lower()
                score = res['score']
                
                if label == 'positive':
                    total_score += score
                elif label == 'negative':
                    total_score -= score
                # neutral is 0.0
                
            return total_score / len(headlines)
        except Exception as e:
            logger.error(f"Error executing FinBERT sentiment on headlines list: {e}")
            return 0.0
