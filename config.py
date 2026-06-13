from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Config(BaseSettings):
    # Load configuration from .env file
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # ── Broker (READ-ONLY scope — no trading permissions needed) ──
    ZERODHA_API_KEY: str = ""
    ZERODHA_API_SECRET: str = ""
    ZERODHA_ACCESS_TOKEN: str = ""

    # ── News & Macro Data ──
    NEWSAPI_KEY: str = ""
    FRED_API_KEY: str = ""

    # ── Telegram ──
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # ── Instruments to Scan ──
    INSTRUMENTS: List[str] = ['NIFTY', 'BANKNIFTY', 'FINNIFTY']

    # ── Signal Thresholds ──
    MIN_SIGNAL_SCORE: int = 65
    MIN_CONFIDENCE: float = 0.60
    MAX_INDIA_VIX: float = 30.0

    # ── Scan Schedule ──
    SCAN_INTERVAL_MINUTES: int = 5
    MARKET_OPEN: str = '09:15'
    MARKET_CLOSE: str = '14:30'

    # ── Database ──
    DATABASE_URL: str = 'sqlite:///signals.db'

config = Config()
