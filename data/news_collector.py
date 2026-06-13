"""
News Collector
Fetches business/financial news for sentiment analysis.
"""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from loguru import logger
from config import config

class NewsCollector:
    def __init__(self):
        self.news_api_key = config.NEWSAPI_KEY

    def get_latest_headlines(self, query: str = "Nifty 50 OR Indian Stock Market", max_results: int = 15) -> list:
        """Fetch news headlines from NewsAPI or fallback to Google News RSS feed"""
        headlines = []
        if self.news_api_key:
            try:
                url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote(query)}&sortBy=publishedAt&pageSize={max_results}&apiKey={self.news_api_key}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    import json
                    data = json.loads(response.read().decode('utf-8'))
                    if data.get('status') == 'ok':
                        for article in data.get('articles', []):
                            title = article.get('title')
                            if title:
                                headlines.append(title)
                        if headlines:
                            return headlines
            except Exception as e:
                logger.warning(f"Failed to fetch from NewsAPI, falling back to Google News RSS: {e}")

        # Google News RSS Fallback (Free & Key-less)
        try:
            rss_query = "Nifty+OR+Sensex+OR+Indian+stock+market"
            rss_url = f"https://news.google.com/rss/search?q={rss_query}&hl=en-IN&gl=IN&ceid=IN:en"
            req = urllib.request.Request(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                for item in root.findall('.//item')[:max_results]:
                    title = item.find('title')
                    if title is not None and title.text:
                        title_text = title.text
                        if " - " in title_text:
                            title_text = title_text.rsplit(" - ", 1)[0]
                        headlines.append(title_text)
        except Exception as e:
            logger.error(f"Failed to fetch from Google News RSS: {e}")

        return headlines
