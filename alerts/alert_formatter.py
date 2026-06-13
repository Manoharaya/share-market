"""
Alert Formatter
Builds human-readable alert formatting and structured rationale.
"""

class AlertFormatter:
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
        
        # Build option legs text
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
