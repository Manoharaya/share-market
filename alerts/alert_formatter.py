"""
Alert Formatter
Builds the signal_data dict used by telegram_bot.
"""

class AlertFormatter:
    def build(self,
              instrument: str,
              spot: float,
              strategy: str,
              legs: dict,
              payoff: dict,
              rationale: str,
              exit_targets: dict,
              **kwargs) -> dict:
        """Assembles all signal fields: instrument, strategy, legs with premiums, max profit/loss, breakevens, rationale, exit targets"""
        # Extract max profit, max loss, breakevens from payoff if they exist
        max_profit = payoff.get('max_profit', 0.0) if isinstance(payoff, dict) else kwargs.get('max_profit', 0.0)
        max_loss = payoff.get('max_loss', 0.0) if isinstance(payoff, dict) else kwargs.get('max_loss', 0.0)
        breakevens_list = payoff.get('breakevens', []) if isinstance(payoff, dict) else kwargs.get('breakevens', [])
        net_premium = payoff.get('net_premium', 0.0) if isinstance(payoff, dict) else kwargs.get('net_premium', 0.0)
        
        # Convert breakevens list to string
        if isinstance(breakevens_list, list):
            breakevens_str = " — ".join([f"{int(x)}" for x in breakevens_list]) if breakevens_list else "None"
        else:
            breakevens_str = str(breakevens_list)

        signal_data = {
            'instrument': instrument,
            'spot': spot,
            'strategy': strategy,
            'legs': legs,
            'payoff': payoff,
            'net_premium': net_premium,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakevens': breakevens_str,
            'rationale': rationale,
            'exit_targets': exit_targets
        }
        signal_data.update(kwargs)
        return signal_data

    # Legacy calculate/format fallback if called from main.py
    def format_signal_message(
        self,
        instrument: str,
        ltp: float,
        direction: str,
        score: int,
        confidence: float,
        strategy: str,
        legs: dict,
        indicators: dict
    ) -> str:
        """Formats the trade signal advisory into a beautiful Markdown message."""
        dir_emoji = "🟢" if direction == "Bullish" else "🔴" if direction == "Bearish" else "🟡"
        
        legs_desc = ""
        for leg_name, strike in legs.items():
            legs_desc += f"• *{leg_name.replace('_', ' ').title()}*: Strike {int(strike)}\n"
            
        pcr = indicators.get('pcr', 0.0)
        vix = indicators.get('vix', 0.0)
        iv_rank = indicators.get('iv_rank', 0.0)
        rsi = indicators.get('rsi', 0.0)
        sentiment = indicators.get('sentiment', 0.0)
        
        msg = (
            f"🚀 *ADVISORY OPTION SIGNAL* 🚀\n\n"
            f"🎯 *Instrument*: {instrument}\n"
            f"📈 *Spot Price*: {ltp:.2f}\n"
            f"📊 *Direction*: {dir_emoji} *{direction}*\n"
            f"🔥 *Score (Conviction)*: {score}/100\n"
            f"💡 *Confidence*: {confidence * 100:.1f}%\n\n"
            f"🛡️ *Recommended Strategy*: *{strategy}*\n"
            f"{legs_desc}\n"
            f"🔍 *Supporting Metrics*:\n"
            f"• RSI (14): {rsi:.1f}\n"
            f"• Put-Call Ratio (PCR): {pcr:.2f}\n"
            f"• India VIX: {vix:.2f} (IV Rank: {iv_rank:.1f}%)\n"
            f"• Market Sentiment: {sentiment:+.2f}\n\n"
            f"⚠️ *Advisory Note*: For educational paper trading only. Maintain strict position sizing."
        )
        return msg
