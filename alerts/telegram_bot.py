"""
Telegram Bot Alerter
Sends formatted signal messages and exit alerts to the Telegram channel.
"""
import requests
from loguru import logger
from config import config

class TelegramBot:
    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID

    def send_message(self, text: str) -> bool:
        """Sends a message to the configured Telegram chat ID"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram token or Chat ID is not configured. Skipping alert.")
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'Markdown'
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully.")
                return True
            else:
                logger.warning(f"Telegram Markdown parse failed, attempting plaintext fallback. Response: {response.text}")
                payload.pop('parse_mode', None)
                response_fallback = requests.post(url, json=payload)
                if response_fallback.status_code == 200:
                    logger.info("Telegram alert sent successfully as plaintext.")
                    return True
                else:
                    logger.error(f"Failed to send Telegram alert: {response_fallback.text}")
                    return False
        except Exception as e:
            logger.error(f"Exception while sending Telegram alert: {e}")
            return False
