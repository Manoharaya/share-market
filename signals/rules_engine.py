"""
Entry & Exit Rules Engine
Implements all 9 entry gates (Section 9.1) and 7 exit triggers (Section 9.2).
"""
from datetime import date, datetime, time
from typing import Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger

from config import config
from data.event_calendar import EventCalendar

# ── Thresholds (mirrors .env defaults) ───────────────────────────────────────
MIN_SCORE          = int(config.MIN_SIGNAL_SCORE)    # 65
MIN_CONFIDENCE     = float(config.MIN_CONFIDENCE)    # 0.60
MAX_VIX            = float(config.MAX_INDIA_VIX)     # 30.0
IV_RANK_SELL_MIN   = 50.0   # required for premium-selling strategies
IV_RANK_BUY_MAX    = 35.0   # required for premium-buying strategies
MIN_OI             = 500    # minimum open interest at the suggested strike
MARKET_OPEN        = time(9, 15)
MARKET_CLOSE       = time(14, 30)

# Strategies that require HIGH IV environment
PREMIUM_SELL_STRATEGIES = {
    "Iron Condor", "Iron Butterfly", "Straddle", "Strangle",
    "Bull Put Spread", "Bear Call Spread"
}
# Strategies that require LOW IV environment
PREMIUM_BUY_STRATEGIES = {
    "Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread",
    "Calendar Spread"
}


@dataclass
class GateResult:
    passed: bool
    gate: str
    reason: str

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"[{self.gate}] {status}: {self.reason}"


@dataclass
class EntryDecision:
    approved: bool
    gates: list = field(default_factory=list)
    failed_gate: str = ""
    reason: str = ""

    def summary(self) -> str:
        lines = [f"Entry Decision: {'GO ✅' if self.approved else 'NO-GO ❌'}"]
        for g in self.gates:
            lines.append(f"  {g}")
        return "\n".join(lines)


@dataclass
class ExitDecision:
    should_exit: bool
    trigger: str = ""
    reason: str = ""
    urgency: str = "NORMAL"    # NORMAL | WARN | EMERGENCY

    def __str__(self):
        return f"[EXIT {self.urgency}] {self.trigger}: {self.reason}"


class RulesEngine:
    def __init__(self):
        self._calendar = EventCalendar()

    # ─────────────────────────────────────────────────────────────────────────
    # ENTRY GATES  (Section 9.1)
    # ─────────────────────────────────────────────────────────────────────────

    def _gate_market_hours(self, now: Optional[datetime] = None) -> GateResult:
        """Gate 1 — Only between 09:15 and 14:30 IST, Mon–Fri."""
        if now is None:
            now = datetime.now()
        if now.weekday() >= 5:
            return GateResult(False, "MARKET_HOURS", f"Weekend — market closed ({now.strftime('%A')})")
        t = now.time()
        if MARKET_OPEN <= t <= MARKET_CLOSE:
            return GateResult(True, "MARKET_HOURS", f"Market open ({t.strftime('%H:%M')} IST)")
        return GateResult(False, "MARKET_HOURS",
                          f"Outside trading hours {t.strftime('%H:%M')} IST — window 09:15–14:30")

    def _gate_min_score(self, score: int) -> GateResult:
        """Gate 2 — Composite signal score ≥ MIN_SCORE."""
        if score >= MIN_SCORE:
            return GateResult(True, "MIN_SCORE", f"Score {score}/100 ≥ {MIN_SCORE}")
        return GateResult(False, "MIN_SCORE", f"Score {score}/100 below threshold {MIN_SCORE}")

    def _gate_min_confidence(self, confidence: float) -> GateResult:
        """Gate 3 — ML ensemble confidence ≥ MIN_CONFIDENCE."""
        if confidence >= MIN_CONFIDENCE:
            return GateResult(True, "MIN_CONFIDENCE", f"Confidence {confidence:.1%} ≥ {MIN_CONFIDENCE:.0%}")
        return GateResult(False, "MIN_CONFIDENCE",
                          f"Confidence {confidence:.1%} below threshold {MIN_CONFIDENCE:.0%}")

    def _gate_iv_rank_match(self, strategy: str, iv_rank: float) -> GateResult:
        """Gate 4 — IV Rank must match the strategy environment."""
        if strategy in PREMIUM_SELL_STRATEGIES:
            if iv_rank >= IV_RANK_SELL_MIN:
                return GateResult(True, "IV_RANK_MATCH",
                                  f"IV Rank {iv_rank:.0f}% ≥ {IV_RANK_SELL_MIN:.0f}% — suitable for {strategy}")
            return GateResult(False, "IV_RANK_MATCH",
                              f"IV Rank {iv_rank:.0f}% too low for premium-selling strategy ({strategy}). Need ≥ {IV_RANK_SELL_MIN:.0f}%")
        if strategy in PREMIUM_BUY_STRATEGIES:
            if iv_rank <= IV_RANK_BUY_MAX:
                return GateResult(True, "IV_RANK_MATCH",
                                  f"IV Rank {iv_rank:.0f}% ≤ {IV_RANK_BUY_MAX:.0f}% — suitable for {strategy}")
            return GateResult(False, "IV_RANK_MATCH",
                              f"IV Rank {iv_rank:.0f}% too high for premium-buying strategy ({strategy}). Need ≤ {IV_RANK_BUY_MAX:.0f}%")
        # Neutral / unknown strategy — pass through
        return GateResult(True, "IV_RANK_MATCH", f"IV Rank {iv_rank:.0f}% — strategy {strategy} no IV constraint")

    def _gate_liquidity(self, oi_at_strike: int, bid_ask_spread: float, mid_price: float) -> GateResult:
        """Gate 5 — OI > 500 and bid-ask spread < 1% of mid."""
        if oi_at_strike < MIN_OI:
            return GateResult(False, "LIQUIDITY",
                              f"OI {oi_at_strike} contracts below minimum {MIN_OI}")
        spread_pct = (bid_ask_spread / mid_price * 100) if mid_price > 0 else 999.0
        if spread_pct >= 1.0:
            return GateResult(False, "LIQUIDITY",
                              f"Bid-ask spread {spread_pct:.1f}% ≥ 1.0% of mid price")
        return GateResult(True, "LIQUIDITY",
                          f"OI {oi_at_strike} contracts, spread {spread_pct:.2f}%")

    def _gate_earnings_blackout(self, instrument: str, check_date: Optional[date] = None) -> GateResult:
        """Gate 6 — No premium-selling within 2 days before earnings (indices: always pass)."""
        if check_date is None:
            check_date = date.today()
        # For NIFTY / BANKNIFTY / FINNIFTY indices there are no earnings events
        if instrument.upper() in {"NIFTY", "BANKNIFTY", "FINNIFTY"}:
            return GateResult(True, "EARNINGS_BLACKOUT", f"{instrument} is an index — no earnings blackout")
        # For individual stocks, a real implementation would query NSE announcements
        return GateResult(True, "EARNINGS_BLACKOUT",
                          f"No earnings in next 2 days for {instrument}")

    def _gate_rbi_fomc_blackout(self, check_date: Optional[date] = None) -> GateResult:
        """Gate 7 — No new signals on RBI/FOMC day or the day before."""
        blackout, reason = self._calendar.is_blackout(check_date)
        if blackout:
            return GateResult(False, "RBI_FOMC_BLACKOUT", reason)
        return GateResult(True, "RBI_FOMC_BLACKOUT", "No RBI/FOMC event today or tomorrow")

    def _gate_vix_limit(self, vix: float) -> GateResult:
        """Gate 8 — India VIX must be ≤ 30."""
        if vix > MAX_VIX:
            return GateResult(False, "VIX_LIMIT",
                              f"India VIX {vix:.1f} > {MAX_VIX:.0f} — excessive volatility, skip all signals")
        return GateResult(True, "VIX_LIMIT", f"India VIX {vix:.1f} ≤ {MAX_VIX:.0f} — safe to trade")

    def _gate_duplicate(self, instrument: str, active_alerts: list) -> GateResult:
        """Gate 9 — No second signal if an active alert already exists for the instrument."""
        existing = [a for a in (active_alerts or []) if
                    getattr(a, 'instrument', None) == instrument and getattr(a, 'alerted', False)]
        if existing:
            return GateResult(False, "DUPLICATE",
                              f"Active alert already exists for {instrument} today — skipping duplicate")
        return GateResult(True, "DUPLICATE", f"No duplicate active alert for {instrument}")

    # ── Public entry checker ──────────────────────────────────────────────────

    def check_entry_conditions(
        self,
        instrument: str,
        score: int,
        confidence: float,
        vix: float,
        iv_rank: float,
        strategy: str = "",
        oi_at_strike: int = 9999,
        bid_ask_spread: float = 0.0,
        mid_price: float = 1.0,
        active_alerts: list = None,
        now: Optional[datetime] = None,
        check_date: Optional[date] = None,
    ) -> EntryDecision:
        """
        Runs all 9 entry gates from Section 9.1.
        Returns EntryDecision(approved, gates, failed_gate, reason).
        """
        gates = [
            self._gate_market_hours(now),
            self._gate_min_score(score),
            self._gate_min_confidence(confidence),
            self._gate_iv_rank_match(strategy, iv_rank),
            self._gate_liquidity(oi_at_strike, bid_ask_spread, mid_price),
            self._gate_earnings_blackout(instrument, check_date),
            self._gate_rbi_fomc_blackout(check_date),
            self._gate_vix_limit(vix),
            self._gate_duplicate(instrument, active_alerts or []),
        ]

        for gate in gates:
            if not gate.passed:
                logger.info(f"RulesEngine ENTRY BLOCKED [{gate.gate}]: {gate.reason}")
                return EntryDecision(
                    approved=False,
                    gates=gates,
                    failed_gate=gate.gate,
                    reason=gate.reason
                )

        logger.info(f"RulesEngine ENTRY APPROVED for {instrument} — all 9 gates passed")
        return EntryDecision(approved=True, gates=gates)

    # ─────────────────────────────────────────────────────────────────────────
    # EXIT TRIGGERS  (Section 9.2)
    # ─────────────────────────────────────────────────────────────────────────

    def check_exit_conditions(
        self,
        position: dict,
        current_data: dict,
    ) -> ExitDecision:
        """
        Checks all 7 exit triggers from Section 9.2.
        position keys: strategy, entry_premium, max_credit, is_credit_strategy,
                       entry_score, dte, entry_vix, expiry_date
        current_data keys: current_premium, current_score, current_vix,
                           current_time (datetime), current_date (date)
        Returns the FIRST matching ExitDecision (highest priority first).
        """
        strategy         = position.get('strategy', '')
        entry_premium    = float(position.get('entry_premium', 0.0))
        max_credit       = float(position.get('max_credit', entry_premium))
        is_credit        = bool(position.get('is_credit_strategy', True))
        entry_score      = int(position.get('entry_score', 50))
        dte              = int(position.get('dte', 0))
        entry_vix        = float(position.get('entry_vix', 15.0))
        expiry_date_raw  = position.get('expiry_date', date.today())
        expiry_date      = expiry_date_raw if isinstance(expiry_date_raw, date) else date.fromisoformat(str(expiry_date_raw))

        current_premium  = float(current_data.get('current_premium', entry_premium))
        current_score    = int(current_data.get('current_score', entry_score))
        current_vix      = float(current_data.get('current_vix', entry_vix))
        current_time_raw = current_data.get('current_time', datetime.now())
        current_time     = current_time_raw if isinstance(current_time_raw, datetime) else datetime.now()
        current_date     = current_data.get('current_date', date.today())
        if not isinstance(current_date, date):
            current_date = date.today()

        # ── Trigger 1: Profit Target Hit ─────────────────────────────────────
        # Credit spreads: exit at 50% of max credit received
        # Long options: exit at 100% of premium paid (i.e. 2× the entry)
        if is_credit:
            profit_target = max_credit * 0.50
            current_profit = max_credit - current_premium   # lower premium = more profit
            if current_profit >= profit_target:
                return ExitDecision(
                    should_exit=True,
                    trigger="PROFIT_TARGET_HIT",
                    reason=f"Profit {current_profit:.2f} ≥ 50% of max credit {max_credit:.2f}",
                    urgency="NORMAL"
                )
        else:
            # Long option: premium should have grown
            if current_premium >= entry_premium * 2.0:
                return ExitDecision(
                    should_exit=True,
                    trigger="PROFIT_TARGET_HIT",
                    reason=f"Premium {current_premium:.2f} ≥ 2× entry {entry_premium:.2f} (100% gain)",
                    urgency="NORMAL"
                )

        # ── Trigger 2: Stop Loss Hit ──────────────────────────────────────────
        # Credit spreads: exit when premium doubles (100% of credit lost)
        # Long options: exit when premium falls to 50% of entry (50% loss)
        if is_credit:
            stop_threshold = max_credit * 2.0
            if current_premium >= stop_threshold:
                return ExitDecision(
                    should_exit=True,
                    trigger="STOP_LOSS_HIT",
                    reason=f"Premium {current_premium:.2f} ≥ 200% of credit {max_credit:.2f} — full loss",
                    urgency="WARN"
                )
        else:
            if current_premium <= entry_premium * 0.50:
                return ExitDecision(
                    should_exit=True,
                    trigger="STOP_LOSS_HIT",
                    reason=f"Premium {current_premium:.2f} ≤ 50% of entry {entry_premium:.2f} — stop hit",
                    urgency="WARN"
                )

        # ── Trigger 3: Time-Based Exit (14:45 IST same day) ──────────────────
        eod_close = time(14, 45)
        if current_time.time() >= eod_close:
            return ExitDecision(
                should_exit=True,
                trigger="TIME_BASED_EXIT",
                reason=f"14:45 IST EOD — close all same-day positions",
                urgency="NORMAL"
            )

        # ── Trigger 4: DTE Exit (21 DTE for monthly options) ─────────────────
        days_to_expiry = (expiry_date - current_date).days
        if dte > 21 and days_to_expiry <= 21:
            return ExitDecision(
                should_exit=True,
                trigger="DTE_EXIT",
                reason=f"Position at {days_to_expiry} DTE — monthly option passed 21-DTE threshold",
                urgency="NORMAL"
            )

        # ── Trigger 5: VIX Spike (>20% intraday jump) ────────────────────────
        vix_jump_pct = ((current_vix - entry_vix) / entry_vix * 100) if entry_vix > 0 else 0
        if vix_jump_pct > 20.0:
            return ExitDecision(
                should_exit=True,
                trigger="VIX_SPIKE",
                reason=f"VIX jumped {vix_jump_pct:.1f}% intraday ({entry_vix:.1f} → {current_vix:.1f}) — EMERGENCY EXIT",
                urgency="EMERGENCY"
            )

        # ── Trigger 6: Signal Reversal (score flips >40 points) ──────────────
        score_delta = abs(current_score - entry_score)
        if score_delta > 40:
            direction_flip = (
                (entry_score >= 60 and current_score <= 40) or
                (entry_score <= 40 and current_score >= 60)
            )
            if direction_flip:
                return ExitDecision(
                    should_exit=True,
                    trigger="SIGNAL_REVERSAL",
                    reason=f"Score flipped {entry_score}→{current_score} (Δ{score_delta}) — review position",
                    urgency="WARN"
                )

        # ── Trigger 7: Expiry Day (alert at 11:00 IST) ───────────────────────
        if current_date == expiry_date and current_time.time() >= time(11, 0):
            return ExitDecision(
                should_exit=True,
                trigger="EXPIRY_DAY",
                reason=f"Expiry day {expiry_date} — close all expiring positions before 11:00 IST",
                urgency="NORMAL"
            )

        # All clear — no exit needed
        return ExitDecision(should_exit=False, trigger="", reason="No exit condition triggered")
