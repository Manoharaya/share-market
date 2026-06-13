"""
Backtesting Engine — Day 24
Replays historical signals using technical features + ML ensemble.
Simulates entry/exit using rules-engine thresholds.
Produces a signal quality report with win rate, avg P&L, Sharpe, max DD, etc.

Usage:
    python backtesting/backtest.py --instrument NIFTY --period 1y
"""
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

# ── Config ────────────────────────────────────────────────────────────────────
MIN_SCORE      = 65
MIN_CONFIDENCE = 0.60
MAX_VIX        = 30.0
HOLD_DAYS      = 5          # default holding period for signal replay
PROFIT_TARGET  = 0.50       # 50 % credit captured
STOP_LOSS      = 1.00       # 100 % credit lost
PREMIUM_PCT    = 0.015      # simulate premium as 1.5 % of spot (ATM rough proxy)
LOT_SIZE       = 50         # NIFTY lot size

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(REPORT_DIR, exist_ok=True)


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_date:   date
    exit_date:    date
    direction:    str          # BULLISH / BEARISH
    strategy:     str
    score:        int
    confidence:   float
    entry_spot:   float
    exit_spot:    float
    entry_prem:   float
    exit_prem:    float
    pnl_pts:      float        # in index points
    pnl_pct:      float        # % of entry premium
    pnl_inr:      float        # ₹ per lot
    outcome:      str          # WIN / LOSS / SCRATCH
    exit_reason:  str


@dataclass
class BacktestResult:
    instrument:    str
    period:        str
    total_trades:  int
    wins:          int
    losses:        int
    scratches:     int
    win_rate:      float
    avg_pnl_pct:   float
    avg_win_pct:   float
    avg_loss_pct:  float
    profit_factor: float
    sharpe:        float
    max_drawdown:  float       # % of rolling equity
    total_pnl_inr: float
    avg_score:     float
    avg_conf:      float
    trades:        List[Trade] = field(default_factory=list)


# ── Backtester ────────────────────────────────────────────────────────────────
class Backtester:

    def __init__(self, instrument: str = "NIFTY", period: str = "1y"):
        self.instrument  = instrument
        self.period      = period
        self._predictor  = None   # cached — loaded once on first use
        self._sym_map    = {
            "NIFTY":     "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "FINNIFTY":  "^CNXFIN",
        }
        self._vix_sym   = "^INDIAVIX"

    # ── Data loading ──────────────────────────────────────────────────────────

    @staticmethod
    def _period_to_days(period: str) -> int:
        """Convert yfinance period string (e.g. '1y', '6mo', '2y') to calendar days."""
        period = period.strip().lower()
        if period.endswith("y"):   return int(period[:-1]) * 365
        if period.endswith("mo"): return int(period[:-2]) * 30
        if period.endswith("d"):  return int(period[:-1])
        return 365

    def _load_ohlcv(self) -> Optional[pd.DataFrame]:
        days = self._period_to_days(self.period)
        sym  = self._sym_map.get(self.instrument, "^NSEI")
        # Try HistoricalData first (uses days= param)
        try:
            from data.historical_data import HistoricalData
            hd = HistoricalData()
            df = hd.get_ohlcv(sym, days=days)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"HistoricalData failed ({e}), falling back to yfinance direct")
        # Direct yfinance fallback
        import yfinance as yf
        df = yf.download(sym, period=self.period, auto_adjust=True, progress=False)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        return df if not df.empty else None

    def _load_vix(self) -> Optional[pd.Series]:
        try:
            import yfinance as yf
            vix = yf.download(self._vix_sym, period=self.period,
                              auto_adjust=True, progress=False)
            if vix.empty:
                return None
            col = "Close" if "Close" in vix.columns else vix.columns[0]
            s   = vix[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.index = pd.to_datetime(s.index).normalize()
            return s
        except Exception as e:
            logger.warning(f"VIX load failed: {e}")
            return None

    # ── Feature + signal generation ───────────────────────────────────────────

    def _compute_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            from features.technical import TechnicalFeatures
            return TechnicalFeatures().compute(df)
        except Exception as e:
            logger.error(f"Feature computation failed: {e}")
            return None

    def _run_ensemble(self, feat_row: np.ndarray) -> dict:
        # Load once and cache for the entire backtest run
        if self._predictor is None:
            try:
                from models.ensemble import EnsemblePredictor
                self._predictor = EnsemblePredictor()
                logger.info("EnsemblePredictor cached for backtest run")
            except Exception as e:
                logger.warning(f"EnsemblePredictor load failed ({e}), using random baseline")
                self._predictor = "FAILED"
        if self._predictor == "FAILED":
            prob = float(np.random.uniform(0.45, 0.65))
            return {"direction": "BULLISH" if prob > 0.5 else "BEARISH",
                    "probability": prob, "confidence": prob, "model_scores": {}}
        try:
            return self._predictor.predict(feat_row.reshape(1, -1))
        except Exception:
            prob = float(np.random.uniform(0.45, 0.65))
            return {"direction": "BULLISH" if prob > 0.5 else "BEARISH",
                    "probability": prob, "confidence": prob, "model_scores": {}}

    def _score_signal(self, feat_dict: dict, pred: dict) -> int:
        try:
            from signals.scorer import SignalScorer
            scorer = SignalScorer()
            s = scorer.score_all(
                features_dict   = feat_dict,
                prediction_dict = pred,
                iv_rank         = 50.0,
                pcr             = 1.0,
                gex             = 0.0,
                sentiment_score = 0.0,
                macro_inputs    = {"sgx_change": 0, "fii_flow": 0, "usdinr_change": 0},
                symbol          = self.instrument,
            )
            return int(s.total)
        except Exception:
            return 50

    # ── Trade simulation ──────────────────────────────────────────────────────

    def _simulate_trade(
        self,
        feat_row:   pd.Series,
        pred:       dict,
        score:      int,
        entry_date: date,
        df:         pd.DataFrame,
        vix:        Optional[pd.Series],
    ) -> Optional[Trade]:
        """Simulate a single trade: enter at close[t], exit at close[t+HOLD_DAYS]
        or earlier if profit target / stop loss hit intraday."""

        direction  = pred.get("direction", "NEUTRAL")
        confidence = float(pred.get("confidence", 0.5))

        if direction == "NEUTRAL":
            return None

        # Entry
        entry_spot = float(feat_row.get("close", feat_row.get("Close", 0)))
        if entry_spot <= 0:
            return None

        entry_prem = round(entry_spot * PREMIUM_PCT, 2)

        # Determine future rows
        future_dates  = df.index[df.index > pd.Timestamp(entry_date)]
        future_rows   = df.loc[future_dates].head(HOLD_DAYS + 5)

        exit_prem    = entry_prem       # default: no change
        exit_spot    = entry_spot
        exit_date    = entry_date
        exit_reason  = "HOLD_EXPIRY"

        # Simulate day-by-day
        for fdate, frow in future_rows.iterrows():
            days_held = (fdate.date() - entry_date).days
            if days_held > HOLD_DAYS:
                exit_reason = "TIME_EXIT"
                break

            c   = float(frow.get("Close", frow.get("close", entry_spot)))
            chg = (c - entry_spot) / entry_spot  # spot move

            # Estimate option premium change (delta ~0.5 for ATM)
            if direction == "BULLISH":
                sim_prem = max(0.01, entry_prem + entry_spot * 0.5 * chg * 0.01)
            else:
                sim_prem = max(0.01, entry_prem - entry_spot * 0.5 * chg * 0.01)

            pnl_frac = (sim_prem - entry_prem) / entry_prem

            if pnl_frac >= PROFIT_TARGET:
                exit_prem   = sim_prem
                exit_spot   = c
                exit_date   = fdate.date()
                exit_reason = "PROFIT_TARGET"
                break
            if pnl_frac <= -STOP_LOSS:
                exit_prem   = sim_prem
                exit_spot   = c
                exit_date   = fdate.date()
                exit_reason = "STOP_LOSS"
                break

            exit_prem  = sim_prem
            exit_spot  = c
            exit_date  = fdate.date()

        pnl_pct = (exit_prem - entry_prem) / entry_prem
        pnl_pts = exit_spot - entry_spot if direction == "BULLISH" else entry_spot - exit_spot
        pnl_inr = pnl_pct * entry_prem * LOT_SIZE

        if pnl_pct > 0.05:
            outcome = "WIN"
        elif pnl_pct < -0.05:
            outcome = "LOSS"
        else:
            outcome = "SCRATCH"

        strategy = (
            "Long Call" if direction == "BULLISH" else
            "Long Put"  if direction == "BEARISH" else "Iron Condor"
        )

        return Trade(
            entry_date  = entry_date,
            exit_date   = exit_date,
            direction   = direction,
            strategy    = strategy,
            score       = score,
            confidence  = confidence,
            entry_spot  = entry_spot,
            exit_spot   = exit_spot,
            entry_prem  = entry_prem,
            exit_prem   = exit_prem,
            pnl_pts     = round(pnl_pts, 2),
            pnl_pct     = round(pnl_pct * 100, 2),
            pnl_inr     = round(pnl_inr, 2),
            outcome     = outcome,
            exit_reason = exit_reason,
        )

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        logger.info(f"Backtester: loading {self.instrument} ({self.period})")
        df  = self._load_ohlcv()
        if df is None or df.empty:
            raise RuntimeError("Could not load OHLCV data")

        # Normalise column names
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df.index   = pd.to_datetime(df.index).normalize()
        df         = df.sort_index()

        vix = self._load_vix()

        logger.info(f"Computing features on {len(df)} rows…")
        feat_df = self._compute_features(df)
        if feat_df is None:
            raise RuntimeError("Feature computation failed")

        # Re-attach OHLC close to feature frame
        feat_df = feat_df.copy()
        feat_df.index = pd.to_datetime(feat_df.index).normalize()

        # Align — strip timezone so all comparisons are tz-naive
        df.index        = df.index.tz_localize(None)        if df.index.tz else df.index
        feat_df.index   = feat_df.index.tz_localize(None)   if feat_df.index.tz else feat_df.index
        df.index        = pd.to_datetime(df.index).normalize()
        feat_df.index   = pd.to_datetime(feat_df.index).normalize()

        common_idx   = df.index.intersection(feat_df.index)
        df_aligned   = df.loc[common_idx]
        feat_aligned = feat_df.loc[common_idx]

        # Drop warm-up rows (indicator burn-in); keep at least 50 bars
        WARMUP = min(200, max(0, len(feat_aligned) - 50))
        df_aligned   = df_aligned.iloc[WARMUP:]
        feat_aligned = feat_aligned.iloc[WARMUP:]
        if len(feat_aligned) == 0:
            logger.warning("No bars left after warmup — try a longer period (e.g. 2y)")
            return self._compute_result([])

        trades: List[Trade] = []
        skip_until = pd.Timestamp("1900-01-01")

        logger.info(f"Scanning {len(feat_aligned)} bars for signals…")
        for i, (ts, feat_row) in enumerate(feat_aligned.iterrows()):
            if ts < skip_until:
                continue

            entry_date = ts.date()

            # VIX gate
            if vix is not None and ts in vix.index:
                v = float(vix.loc[ts])
                if v > MAX_VIX:
                    continue

            # Weekend gate (already filtered by market data, but double-check)
            if entry_date.weekday() >= 5:
                continue

            feat_row_orig = df_aligned.loc[ts] if ts in df_aligned.index else feat_row
            close_val = float(feat_row_orig.get("close", 0))
            if close_val <= 0:
                continue

            # Build feature array for ML
            feat_arr = feat_row.values.astype(float)
            feat_arr = np.nan_to_num(feat_arr, nan=0.0)

            pred  = self._run_ensemble(feat_arr)
            score = self._score_signal(feat_row.to_dict(), pred)

            # Gate: min score + confidence
            if score < MIN_SCORE:
                continue
            if float(pred.get("confidence", 0)) < MIN_CONFIDENCE:
                continue

            # Simulate trade
            trade = self._simulate_trade(
                feat_row   = feat_row_orig,
                pred       = pred,
                score      = score,
                entry_date = entry_date,
                df         = df_aligned,
                vix        = vix,
            )
            if trade is None:
                continue

            trades.append(trade)
            # Skip until exit so no overlapping positions
            skip_until = pd.Timestamp(trade.exit_date) + timedelta(days=1)

        return self._compute_result(trades)

    # ── Result computation ────────────────────────────────────────────────────

    def _compute_result(self, trades: List[Trade]) -> BacktestResult:
        n = len(trades)
        if n == 0:
            return BacktestResult(
                instrument=self.instrument, period=self.period,
                total_trades=0, wins=0, losses=0, scratches=0,
                win_rate=0, avg_pnl_pct=0, avg_win_pct=0, avg_loss_pct=0,
                profit_factor=0, sharpe=0, max_drawdown=0,
                total_pnl_inr=0, avg_score=0, avg_conf=0, trades=[]
            )

        wins      = [t for t in trades if t.outcome == "WIN"]
        losses    = [t for t in trades if t.outcome == "LOSS"]
        scratches = [t for t in trades if t.outcome == "SCRATCH"]

        win_rate   = len(wins) / n
        pnls       = [t.pnl_pct for t in trades]
        avg_pnl    = float(np.mean(pnls))
        avg_win    = float(np.mean([t.pnl_pct for t in wins]))   if wins   else 0.0
        avg_loss   = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0

        gross_win  = sum(t.pnl_inr for t in wins)   if wins   else 0.0
        gross_loss = abs(sum(t.pnl_inr for t in losses)) if losses else 1.0
        pf         = round(gross_win / gross_loss, 3) if gross_loss > 0 else 0.0

        # Sharpe (daily P&L series)
        pnl_arr = np.array(pnls)
        sharpe  = float(np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-9) * np.sqrt(252 / HOLD_DAYS))

        # Max drawdown on cumulative equity
        equity  = np.cumsum([t.pnl_inr for t in trades])
        peak    = np.maximum.accumulate(equity)
        dd      = (peak - equity) / (np.abs(peak) + 1e-9)
        max_dd  = float(np.max(dd) * 100)

        return BacktestResult(
            instrument    = self.instrument,
            period        = self.period,
            total_trades  = n,
            wins          = len(wins),
            losses        = len(losses),
            scratches     = len(scratches),
            win_rate      = round(win_rate * 100, 1),
            avg_pnl_pct   = round(avg_pnl, 2),
            avg_win_pct   = round(avg_win, 2),
            avg_loss_pct  = round(avg_loss, 2),
            profit_factor = pf,
            sharpe        = round(sharpe, 3),
            max_drawdown  = round(max_dd, 2),
            total_pnl_inr = round(sum(t.pnl_inr for t in trades), 2),
            avg_score     = round(float(np.mean([t.score for t in trades])), 1),
            avg_conf      = round(float(np.mean([t.confidence for t in trades])), 3),
            trades        = trades,
        )


# ── Report generation ─────────────────────────────────────────────────────────

def print_report(result: BacktestResult):
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  SIGNAL QUALITY REPORT — {result.instrument}  ({result.period})")
    print(sep)
    print(f"  Total Trades   : {result.total_trades}")
    print(f"  Wins           : {result.wins}  ({result.win_rate:.1f}%)")
    print(f"  Losses         : {result.losses}")
    print(f"  Scratches      : {result.scratches}")
    print(f"  Avg P&L        : {result.avg_pnl_pct:+.2f}%")
    print(f"  Avg Win        : {result.avg_win_pct:+.2f}%")
    print(f"  Avg Loss       : {result.avg_loss_pct:+.2f}%")
    print(f"  Profit Factor  : {result.profit_factor:.3f}")
    print(f"  Sharpe Ratio   : {result.sharpe:.3f}")
    print(f"  Max Drawdown   : {result.max_drawdown:.2f}%")
    print(f"  Total P&L      : ₹{result.total_pnl_inr:,.2f}")
    print(f"  Avg Score      : {result.avg_score}/100")
    print(f"  Avg Confidence : {result.avg_conf:.1%}")
    print(sep)

    if result.trades:
        print(f"\n  {'DATE':<12} {'DIR':<10} {'SCORE':<6} {'CONF':<6} {'P&L%':>8}  {'OUTCOME':<10}  EXIT REASON")
        print(f"  {'-'*12} {'-'*10} {'-'*6} {'-'*6} {'-'*8}  {'-'*10}  {'-'*16}")
        for t in result.trades[:40]:   # cap at 40 rows for readability
            icon = "✅" if t.outcome == "WIN" else ("❌" if t.outcome == "LOSS" else "—")
            print(f"  {str(t.entry_date):<12} {t.direction:<10} {t.score:<6} {t.confidence:<6.2f}"
                  f" {t.pnl_pct:>+8.2f}%  {icon} {t.outcome:<8}  {t.exit_reason}")
        if len(result.trades) > 40:
            print(f"  … {len(result.trades) - 40} more trades (see CSV)")
    print(f"\n{sep}\n")


def save_report(result: BacktestResult):
    # Trade log CSV
    csv_path = os.path.join(REPORT_DIR, f"backtest_{result.instrument}_{result.period}.csv")
    rows = [{
        "entry_date": t.entry_date, "exit_date": t.exit_date,
        "direction": t.direction, "strategy": t.strategy,
        "score": t.score, "confidence": t.confidence,
        "entry_spot": t.entry_spot, "exit_spot": t.exit_spot,
        "entry_prem": t.entry_prem, "exit_prem": t.exit_prem,
        "pnl_pct": t.pnl_pct, "pnl_inr": t.pnl_inr,
        "outcome": t.outcome, "exit_reason": t.exit_reason,
    } for t in result.trades]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Summary text
    txt_path = os.path.join(REPORT_DIR, f"backtest_{result.instrument}_{result.period}_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"BACKTEST SUMMARY — {result.instrument} ({result.period})\n")
        f.write(f"Total Trades  : {result.total_trades}\n")
        f.write(f"Win Rate      : {result.win_rate:.1f}%\n")
        f.write(f"Profit Factor : {result.profit_factor}\n")
        f.write(f"Sharpe        : {result.sharpe}\n")
        f.write(f"Max Drawdown  : {result.max_drawdown:.2f}%\n")
        f.write(f"Total P&L     : INR {result.total_pnl_inr:,.2f}\n")

    logger.info(f"Reports saved → {csv_path}")
    logger.info(f"             → {txt_path}")
    return csv_path, txt_path


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options Signal Backtester")
    parser.add_argument("--instrument", default="NIFTY",
                        choices=["NIFTY", "BANKNIFTY", "FINNIFTY"])
    parser.add_argument("--period", default="1y",
                        help="yfinance period string: 6mo, 1y, 2y")
    args = parser.parse_args()

    bt     = Backtester(instrument=args.instrument, period=args.period)
    result = bt.run()
    print_report(result)
    csv_p, txt_p = save_report(result)

    print(f"  Trade log CSV : {csv_p}")
    print(f"  Summary TXT   : {txt_p}\n")
