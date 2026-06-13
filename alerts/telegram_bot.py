"""
Telegram Bot Alerter
Sends formatted signal messages and exit alerts to the Telegram channel.
"""
import requests
from datetime import datetime
from loguru import logger
from config import config

class TelegramBot:
    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID

    def send_message(self, text: str) -> bool:
        """Sends a message to the configured Telegram chat ID using HTML parse mode"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram token or Chat ID is not configured. Skipping alert.")
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': 'HTML'
        }
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                logger.info("Telegram message sent successfully using HTML parse mode.")
                return True
            else:
                logger.warning(f"Telegram HTML parse failed, attempting plaintext fallback. Response: {response.text}")
                payload.pop('parse_mode', None)
                response_fallback = requests.post(url, json=payload)
                if response_fallback.status_code == 200:
                    logger.info("Telegram message sent successfully as plaintext fallback.")
                    return True
                else:
                    logger.error(f"Failed to send Telegram message: {response_fallback.text}")
                    return False
        except Exception as e:
            logger.error(f"Exception while sending Telegram message: {e}")
            return False

    def send(self, message: str) -> bool:
        """Alias for send_message to match the signature of Section 11.8"""
        return self.send_message(message)

    def format_signal_alert(self, signal_data: dict) -> str:
        """Formats the trade signal alert into structured HTML text matching Section 6.1 and 6.2"""
        score = signal_data.get('score', 0)
        emoji = '[HOT]' if score >= 80 else '[SIGNAL]'
        
        instrument = signal_data.get('instrument', '')
        strategy = signal_data.get('strategy', '')
        spot = signal_data.get('spot', 0.0)
        
        # Underlying representation
        underlying = signal_data.get('underlying', instrument)
        if underlying == 'NIFTY':
            underlying = 'NIFTY 50'
        elif underlying == 'BANKNIFTY':
            underlying = 'NIFTY BANK'
        elif underlying == 'FINNIFTY':
            underlying = 'NIFTY FIN SERVICE'
            
        expiry = signal_data.get('expiry', '')
        direction = signal_data.get('direction', 'NEUTRAL')
        
        confidence = signal_data.get('confidence', 0.0)
        # Convert decimal confidence to percentage string
        if isinstance(confidence, float) and confidence <= 1.0:
            conf_str = f"{confidence * 100:.0f}%"
        else:
            conf_str = f"{confidence:.0f}%"
            
        lines = [
            f"{emoji}  <b>NEW SIGNAL — {instrument}</b>  [{score}/100]",
            "-----------------------------",
            f"[CHART]  Strategy   : <b>{strategy}</b>",
            f"[PIN]  Instrument : {underlying}  |  Expiry: {expiry}",
            f"[UP]  Direction  : {direction.upper()}  |  Confidence: {conf_str}",
            "-----------------------------",
        ]
        
        # Format legs
        legs = signal_data.get('legs', [])
        if isinstance(legs, list):
            for i, leg in enumerate(legs, 1):
                action = leg.get('action', '').upper()
                symbol = leg.get('symbol', '')
                premium = leg.get('premium', 0.0)
                delta = leg.get('delta', None)
                delta_str = f"  (Delta: {delta:.2f})" if delta is not None else ""
                lines.append(f"[LEG]  LEG {i}  :  {action:<4}  {symbol}  @  {premium:.2f}{delta_str}₹")
        elif isinstance(legs, dict):
            # Fallback if dictionary is provided
            for i, (name, details) in enumerate(legs.items(), 1):
                if isinstance(details, dict):
                    strike = details.get('strike', 0.0)
                    premium = details.get('premium', 0.0)
                    action = details.get('action', 'BUY').upper()
                    delta = details.get('delta', None)
                    delta_str = f"  (Delta: {delta:.2f})" if delta is not None else ""
                    lines.append(f"[LEG]  LEG {i}  :  {action:<4}  {name.replace('_', ' ').upper()} {int(strike)}  @  {premium:.2f}{delta_str}₹")
                else:
                    lines.append(f"[LEG]  LEG {i}  :  {name.replace('_', ' ').upper()}  @  {details}₹")
                    
        lines.append("-----------------------------")
        
        # Payoff Profile
        net_premium = signal_data.get('net_premium', 0.0)
        max_profit = signal_data.get('max_profit', 0.0)
        max_loss = signal_data.get('max_loss', 0.0)
        breakevens = signal_data.get('breakevens', 'None')
        
        # Determine lot size
        lot_size = 50 if instrument == 'NIFTY' else 15 if instrument == 'BANKNIFTY' else 40
        lot_credit = abs(net_premium) * lot_size
        credit_debit = "Net Credit" if net_premium >= 0 else "Net Debit"
        
        lines.append(f"[PROFIT]  {credit_debit:<13}: {abs(net_premium):.2f} per lot  ( ₹{lot_credit:,.0f} for 1 lot of {lot_size})₹ ₹")
        
        profit_val = f"₹{max_profit:,.0f}" if isinstance(max_profit, (int, float)) else str(max_profit)
        loss_val = f"₹{max_loss:,.0f}" if isinstance(max_loss, (int, float)) else str(max_loss)
        
        max_profit_desc = signal_data.get('max_profit_desc', '')
        max_loss_desc = signal_data.get('max_loss_desc', '')
        
        lines.append(f"[PROFIT]  Max Profit   : {profit_val} per lot{max_profit_desc}₹")
        lines.append(f"[STOP]  Max Loss     : {loss_val} per lot{max_loss_desc}₹ ₹")
        lines.append(f"[CALC]  Breakevens   : {breakevens}")
        lines.append("-----------------------------")
        
        # IV, VIX, PCR
        vix = signal_data.get('vix', 0.0)
        iv_rank = signal_data.get('iv_rank', 0.0)
        pcr = signal_data.get('pcr', 0.0)
        pcr_bias = " (Bullish bias)" if pcr > 1.0 else " (Bearish bias)" if pcr < 0.8 else " (Neutral)"
        lines.append(f"[DOWN]  IV Rank: {iv_rank:.0f}%  |  India VIX: {vix:.1f}  |  PCR OI: {pcr:.2f}{pcr_bias}")
        
        # Global macro
        sgx_change = signal_data.get('sgx_change', 0.0)
        sgx_sign = "+" if sgx_change > 0 else ""
        usdinr = signal_data.get('usdinr', 0.0)
        usdinr_status = "Stable" if 83.0 <= usdinr <= 84.5 else "Weak" if usdinr > 84.5 else "Strong"
        lines.append(f"[GLOBAL]  Global: SGX Nifty {sgx_sign}{sgx_change:.2f}%  |  USD/INR: {usdinr:.2f} ({usdinr_status})")
        
        # News Sentiment
        sentiment_score = signal_data.get('sentiment_score', 0.0)
        sentiment_label = signal_data.get('sentiment_label', '')
        if not sentiment_label:
            sentiment_label = "BULLISH" if sentiment_score > 0.15 else "BEARISH" if sentiment_score < -0.15 else "NEUTRAL"
        lines.append(f"[NEWS]  Sentiment: {sentiment_label} (score: {sentiment_score:+.2f})")
        lines.append("-----------------------------")
        
        # Rationale
        rationale = signal_data.get('rationale', '')
        lines.append("[TIP]  WHY THIS TRADE:")
        for rat_line in rationale.split('\n'):
            if rat_line.strip():
                lines.append(f"    {rat_line.strip()}")
        lines.append("-----------------------------")
        
        # Targets and DTE
        exit_targets = signal_data.get('exit_targets', {})
        target = exit_targets.get('target', 0.0)
        stop_loss = exit_targets.get('stop_loss', 0.0)
        lines.append(f"[TARGET]  TARGET EXIT: {target:.2f}  |    STOP: {stop_loss:.2f}⛔")
        
        exit_time = signal_data.get('exit_time', '15:30 IST')
        dte = signal_data.get('dte', 0)
        lines.append(f"[TIME]  Exit by: {exit_time}  |  DTE: {dte} days")
        lines.append("-----------------------------")
        
        # Timing info
        timestamp = signal_data.get('timestamp', datetime.now().strftime('%H:%M:%S'))
        lines.append(f"[FAST] Signal generated at {timestamp} IST")
        lines.append("[>>] <i>Place order manually on Zerodha / Upstox</i>")
        
        return "\n".join(lines)

    def format_exit_alert(self, signal_data: dict, exit_reason: str) -> str:
        """Formats the exit alert into structured HTML text"""
        instrument = signal_data.get('instrument', '')
        strategy = signal_data.get('strategy', '')
        net_premium = signal_data.get('net_premium', 0.0)
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        lines = [
            "🚪  <b>OPTIONS EXIT ALERT</b>  🚪",
            "-----------------------------",
            f"[PIN]  Instrument : {instrument}",
            f"[CHART]  Strategy   : {strategy}",
            f"[EXIT]  Exit Reason: <b>{exit_reason}</b>",
            "-----------------------------",
            f"[PROFIT]  Entry Premium : {net_premium:+.2f} per lot",
            f"[TIME]  Exit Time     : {timestamp} IST",
            "-----------------------------",
            "[>>] <i>Close position manually on Zerodha / Upstox</i>"
        ]
        return "\n".join(lines)

    def send_signal_alert(self, signal_data: dict) -> bool:
        """Helper to format and send a trade signal alert"""
        msg = self.format_signal_alert(signal_data)
        return self.send_message(msg)

    def send_exit_alert(self, signal_data: dict, exit_reason: str) -> bool:
        """Helper to format and send an exit alert"""
        msg = self.format_exit_alert(signal_data, exit_reason)
        return self.send_message(msg)

# Alias to support imports as TelegramAlerter
TelegramAlerter = TelegramBot
