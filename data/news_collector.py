"""
News Collector & FinBERT Sentiment
Polls news from multiple RSS feeds + NewsAPI, and runs sentiment analysis using ProsusAI/finbert.
"""
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import feedparser
from transformers import pipeline
from loguru import logger
from config import config

class SentimentAnalyzer:
    _nlp = None  # Class-level model pipeline cache

    def __init__(self):
        if SentimentAnalyzer._nlp is None:
            logger.info("Initializing FinBERT text classification pipeline (downloads ~400MB on first run)...")
            # Set caching and load model
            SentimentAnalyzer._nlp = pipeline('text-classification', model='ProsusAI/finbert')
            logger.info("FinBERT model loaded and cached in memory.")

    def fetch_headlines(self, query: str, hours: int = 6) -> list:
        """Fetch news headlines from NewsAPI and all 3 RSS feeds within given time cutoff"""
        headlines = []
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=hours)

        rss_urls = [
            'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'https://www.moneycontrol.com/rss/marketoutlook.xml',
            'https://www.business-standard.com/rss/markets-106.rss',
            'https://news.google.com/rss/search?q=Nifty+OR+Sensex+OR+Indian+stock+market&hl=en-IN&gl=IN&ceid=IN:en'
        ]

        # 1. Fetch RSS Feeds
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    title = entry.get('title', '')
                    if not title:
                        continue

                    # Filter by date if published time structure is present
                    time_struct = entry.get('published_parsed') or entry.get('updated_parsed')
                    if time_struct and hours > 0:
                        pub_dt = datetime.fromtimestamp(time.mktime(time_struct))
                        if pub_dt < cutoff:
                            continue

                    # Filter by index relevance keywords
                    keywords = [query.lower(), 'market', 'stock', 'nifty', 'banknifty', 'finnifty', 'sensex', 'rupee', 'nifty 50']
                    title_lower = title.lower()
                    if any(kw in title_lower for kw in keywords):
                        # Clean source title suffix
                        if " - " in title:
                            title = title.rsplit(" - ", 1)[0]
                        headlines.append(title.strip())
            except Exception as e:
                logger.warning(f"Error parsing RSS feed {url}: {e}")

        # 2. Fetch NewsAPI (if key provided)
        if config.NEWSAPI_KEY:
            try:
                import json
                api_url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote(query)}&sortBy=publishedAt&pageSize=30&apiKey={config.NEWSAPI_KEY}"
                req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    if data.get('status') == 'ok':
                        for article in data.get('articles', []):
                            title = article.get('title')
                            published_at = article.get('publishedAt')
                            if published_at and hours > 0:
                                try:
                                    pub_dt = datetime.strptime(published_at[:19], "%Y-%m-%dT%H:%M:%S")
                                    if pub_dt < cutoff:
                                        continue
                                except Exception:
                                    pass
                            if title:
                                headlines.append(title.strip())
            except Exception as e:
                logger.warning(f"NewsAPI fetch failed: {e}")

        # De-duplicate
        headlines = list(set(headlines))
        
        # Weekend fallback: if 0 headlines are fetched due to tight hours cutoff, fetch without date restriction
        if not headlines and hours > 0:
            logger.info(f"No headlines found for query '{query}' in last {hours} hours. Retrying with no time cutoff...")
            return self.fetch_headlines(query, hours=0)

        return headlines[:20]  # Limit to top 20 latest headlines for inference performance

    def get_sentiment(self, symbol: str) -> float:
        """Fetches latest headlines and calculates average FinBERT sentiment score (-1 to +1)"""
        headlines = self.fetch_headlines(symbol, hours=6)
        if not headlines:
            # Fallback query
            headlines = self.fetch_headlines("Indian stock market", hours=12)

        if not headlines:
            logger.warning(f"No news headlines retrieved for {symbol}. Returning neutral sentiment (0.0).")
            return 0.0

        logger.info(f"Running FinBERT on {len(headlines)} headlines for {symbol}...")
        
        try:
            results = self._nlp(headlines)
            
            total_score = 0.0
            for res in results:
                label = res['label'].lower()
                score = res['score']
                
                if label == 'positive':
                    total_score += score
                elif label == 'negative':
                    total_score -= score
                # neutral counts as 0.0
                
            avg_score = total_score / len(headlines)
            avg_score = max(-1.0, min(1.0, avg_score))
            logger.info(f"FinBERT average sentiment score for {symbol}: {avg_score:+.4f}")
            return avg_score
            
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment calculation: {e}")
            return 0.0

# Decoupled alias class to support original news_collector logic
class NewsCollector:
    def get_latest_headlines(self, query: str = "Nifty 50 OR Indian Stock Market", max_results: int = 15) -> list:
        sa = SentimentAnalyzer()
        return sa.fetch_headlines(query, hours=0)[:max_results]
