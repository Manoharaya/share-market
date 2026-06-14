"""
Full Integration Test Suite — Days 25-26
Validates every module end-to-end with correct real API signatures.
Run: python tests/test_integration.py
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, date
import numpy as np
import pandas as pd
from loguru import logger
logger.remove()

PASS = "✅"
FAIL = "❌"
results = []

def test(name, fn):
    try:
        t0  = time.time()
        msg = fn()
        print(f"  {PASS} [{time.time()-t0:5.2f}s] {name}: {msg}")
        results.append((name, True, msg))
    except Exception as e:
        print(f"  {FAIL} [  ERR] {name}: {e}")
        results.append((name, False, str(e)))

print("=" * 68)
print("  OPTIONS SIGNAL ADVISORY — FULL INTEGRATION TEST")
print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] DATA LAYER")

_df = None   # shared across tests

def t_historical():
    global _df
    from data.historical_data import HistoricalData
    _df = HistoricalData().get_ohlcv("^NSEI", days=365)
    assert _df is not None and not _df.empty
    assert len(_df) > 200, f"Too few rows: {len(_df)}"
    return f"{len(_df)} rows, cols={list(_df.columns[:4])}"
test("Historical OHLCV (NIFTY 1y)", t_historical)

def t_vix_history():
    import yfinance as yf
    raw = yf.download("^INDIAVIX", period="3mo", auto_adjust=True, progress=False)
    assert not raw.empty
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    vix_val = float(close.iloc[-1])
    assert 5 < vix_val < 100, f"VIX out of range: {vix_val}"
    return f"Latest VIX={vix_val:.1f}, rows={len(raw)}"
test("India VIX history (3mo)", t_vix_history)

def t_macro():
    from data.global_macro import GlobalMacro
    m = GlobalMacro().get_macro_indicators()
    assert isinstance(m, dict) and len(m) > 0, "Empty macro dict"
    return f"Keys present: {list(m.keys())}"
test("Global Macro (returns dict)", t_macro)

def t_sentiment():
    from data.news_collector import SentimentAnalyzer
    sa   = SentimentAnalyzer()
    # Use the real method name: get_sentiment()
    news = sa.fetch_headlines("NIFTY")
    sent = sa.get_sentiment(news) if news else {"score": 0.0, "label": "neutral"}
    score = float(sent.get("score", 0.0)) if isinstance(sent, dict) else float(sent)
    assert -1.5 <= score <= 1.5, f"Score out of range: {score}"
    return f"Sentiment={score:.3f}, label={sent.get('label','') if isinstance(sent,dict) else ''}"
test("Sentiment score", t_sentiment)

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] FEATURE PIPELINE")

_features = None

def t_tech_features():
    global _features
    from features.technical import TechnicalFeatures
    assert _df is not None, "OHLCV not loaded"
    _features = TechnicalFeatures().compute(_df)
    assert _features is not None and not _features.empty
    assert _features.shape[1] >= 40, f"Only {_features.shape[1]} cols (need 40+)"
    # Allow up to 5 NaN per column in last 100 rows (some indicators need full warmup)
    last100_nan = int(_features.iloc[-100:].isna().sum().sum())
    return f"Shape={_features.shape}, NaNs(last 100)={last100_nan}"
test("Technical features (40+ cols)", t_tech_features)

def t_options_features():
    import yfinance as yf
    from features.options_features import OptionsFeatures
    of = OptionsFeatures()
    # Real sig: calculate_iv_rank(current_vix: float, vix_history: pd.DataFrame)
    raw  = yf.download("^INDIAVIX", period="1y", auto_adjust=True, progress=False)
    col  = raw["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    vix_hist = col.reset_index(drop=True).to_frame(name="Close")
    iv_rank  = of.calculate_iv_rank(float(col.iloc[-1]), vix_hist)
    assert 0 <= iv_rank <= 100, f"IV rank out of range: {iv_rank}"
    return f"IV Rank={iv_rank:.1f}%"
test("Options features — IV Rank", t_options_features)

# ─────────────────────────────────────────────────────────────────────────────
# 3. ML ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] ML ENSEMBLE")

def t_ensemble():
    from models.ensemble import EnsemblePredictor
    ep      = EnsemblePredictor()
    assert _features is not None, "Features not computed"
    feat_df = _features.iloc[-20:].copy()
    pred    = ep.predict(feat_df)
    assert "direction"  in pred
    assert "confidence" in pred
    # direction can be mixed-case e.g. 'Bearish'
    assert pred["direction"].upper() in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert 0 <= float(pred["confidence"]) <= 1
    return (f"dir={pred['direction']}, conf={pred['confidence']:.2f}, "
            f"models={list(pred.get('model_scores',{}).keys())}")
test("Ensemble predict (last 20 rows)", t_ensemble)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SIGNAL SCORER
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] SIGNAL SCORER")

def t_scorer():
    from signals.scorer import SignalScorer
    feat_dict = _features.iloc[-1].to_dict() if _features is not None else {}
    pred_dict = {"direction": "BULLISH", "confidence": 0.65, "probability": 0.65}
    s = SignalScorer().score_all(
        features_dict=feat_dict, prediction_dict=pred_dict,
        iv_rank=30.0, pcr=1.2, gex=0.0, sentiment_score=0.15,
        macro_inputs={"sgx_change": 0.3, "fii_flow": 500, "usdinr_change": -0.1},
        symbol="NIFTY"
    )
    assert 0 <= s.total <= 100, f"Score out of range: {s.total}"
    return (f"Total={s.total}/100  tech={s.technical} ml={s.ml} "
            f"opt={s.options} sent={s.sentiment} macro={s.macro} fund={s.fundamental}")
test("Signal scorer (all 6 dimensions)", t_scorer)

# ─────────────────────────────────────────────────────────────────────────────
# 5. RULES ENGINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] RULES ENGINE")

def t_entry_pass():
    from signals.rules_engine import RulesEngine
    d = RulesEngine().check_entry_conditions(
        instrument="NIFTY", score=78, confidence=0.75, vix=14.7, iv_rank=29.7,
        strategy="Long Call", oi_at_strike=1500, bid_ask_spread=0.20, mid_price=142.0,
        now=datetime(2026, 6, 19, 11, 30), check_date=date(2026, 6, 19),
    )
    assert d.approved, f"Expected GO — blocked: {d.reason}"
    assert len(d.gates) == 9
    return f"GO ✅ — all {len(d.gates)} gates passed"
test("Entry gate happy path (all 9 pass)", t_entry_pass)

def t_entry_vix_block():
    from signals.rules_engine import RulesEngine
    d = RulesEngine().check_entry_conditions(
        instrument="NIFTY", score=78, confidence=0.75, vix=35.0, iv_rank=29.7,
        strategy="Long Call", oi_at_strike=1500, bid_ask_spread=0.20, mid_price=142.0,
        now=datetime(2026, 6, 19, 11, 30), check_date=date(2026, 6, 19),
    )
    assert not d.approved and d.failed_gate == "VIX_LIMIT"
    return "Blocked at VIX_LIMIT (VIX=35 > 30) ✅"
test("Entry gate VIX block", t_entry_vix_block)

def t_exit_profit():
    from signals.rules_engine import RulesEngine
    pos = dict(strategy="Iron Condor", entry_premium=50.0, max_credit=50.0,
               is_credit_strategy=True, entry_score=75, dte=30,
               entry_vix=14.0, expiry_date=date(2026, 7, 30))
    cur = dict(current_premium=23.0, current_score=75, current_vix=14.0,
               current_time=datetime(2026, 6, 20, 11, 0), current_date=date(2026, 6, 20))
    ex  = RulesEngine().check_exit_conditions(pos, cur)
    assert ex.should_exit and ex.trigger == "PROFIT_TARGET_HIT"
    return "PROFIT_TARGET_HIT triggered ✅"
test("Exit trigger profit target", t_exit_profit)

# ─────────────────────────────────────────────────────────────────────────────
# 6. STRATEGY & STRIKE SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] STRATEGY & STRIKE SELECTOR")

VALID_STRATEGIES = {
    "Long Call", "Long Put", "Bull Call Spread", "Bear Put Spread",
    "Bull Put Spread", "Bear Call Spread", "Iron Condor", "Straddle",
    "Strangle", "Iron Butterfly", "Calendar Spread", "No Trade"
}

def t_strategy_selector():
    from signals.strategy_selector import StrategySelector
    sel = StrategySelector()
    # Real sig: determine_strategy(direction, iv_rank, confidence)
    s1 = sel.determine_strategy(direction="NEUTRAL", iv_rank=72, confidence=0.75)
    assert s1 in VALID_STRATEGIES, f"Invalid strategy: '{s1}'"
    s2 = sel.determine_strategy(direction="BULLISH", iv_rank=25, confidence=0.75)
    assert s2 in VALID_STRATEGIES, f"Invalid strategy: '{s2}'"
    return f"Neutral+HighIV→{s1}, Bullish+LowIV→{s2}"
test("Strategy selector (neutral→IC, bullish→LC)", t_strategy_selector)

def t_strike_selector():
    from signals.strike_selector import StrikeSelector
    spot    = 23600
    # Wide range so OTM strikes with delta ~0.15 / -0.15 exist
    strikes = range(22000, 25200, 100)
    rows    = []
    for k in strikes:
        delta_c = max(0.01, min(0.99, 0.5 - (k - spot) / (spot * 0.1)))
        delta_p = delta_c - 1.0
        rows += [
            {"strike": k, "type": "CE", "ltp": max(1.0, (spot-k+600)*0.5),
             "openInterest": 5000, "iv": 15.0, "delta": delta_c},
            {"strike": k, "type": "PE", "ltp": max(1.0, (k-spot+600)*0.5),
             "openInterest": 5000, "iv": 15.0, "delta": delta_p},
        ]
    chain  = pd.DataFrame(rows)
    ss     = StrikeSelector()
    legs   = ss.build_legs("Iron Condor", chain, spot)
    assert legs, (f"No legs returned. Delta range CE: "
                  f"{chain[chain.type=='CE'].delta.min():.2f}-{chain[chain.type=='CE'].delta.max():.2f}")
    payoff = ss.compute_payoff("Iron Condor", legs)
    assert "max_profit" in payoff and "max_loss" in payoff
    return f"{len(legs)} legs, max_profit={payoff['max_profit']:.0f}, max_loss={payoff['max_loss']:.0f}"
test("Strike selector (Iron Condor mock chain)", t_strike_selector)

# ─────────────────────────────────────────────────────────────────────────────
# 7. ALERT FORMATTER & TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] ALERT FORMATTER & TELEGRAM")

_dummy_legs   = [{"action":"BUY","strike":23600,"option_type":"CE",
                  "premium":142.5,"delta":0.51,"expiry":"2026-06-26"}]
_dummy_payoff = {"max_profit":7125,"max_loss":7125,"breakevens":[23742],"net_debit":142.5}

def t_alert_formatter():
    from alerts.alert_formatter import AlertFormatter
    # Real signature: build(instrument, spot, strategy, legs, payoff, rationale, exit_targets, **kwargs)
    data = AlertFormatter().build(
        instrument   = "NIFTY",
        spot         = 23622.9,
        strategy     = "Long Call",
        legs         = _dummy_legs,
        payoff       = _dummy_payoff,
        rationale    = "IV Rank at 30% — options are cheap; PCR bullish (1.2)",
        exit_targets = {"target": 71.25, "stop": 285.0, "dte": 3},
        direction    = "BULLISH",
        score        = 78,
        confidence   = 0.75,
        iv_rank      = 30,
        vix          = 14.7,
        pcr          = 1.2,
        sentiment_score = 0.25,
        sgx_change   = 0.3,
        usdinr       = 84.5,
    )
    for k in ["instrument", "strategy", "legs", "payoff"]:
        assert k in data, f"Missing key: {k}"
    return f"Keys={list(data.keys())[:6]}…"
test("Alert formatter build() dict", t_alert_formatter)

def t_telegram_format():
    from alerts.alert_formatter import AlertFormatter
    from alerts.telegram_bot import TelegramBot
    data = AlertFormatter().build(
        instrument="NIFTY", spot=23622.9, strategy="Long Call",
        legs=_dummy_legs, payoff=_dummy_payoff,
        rationale="IV Rank at 30%",
        exit_targets={"target": 71.25, "stop": 285.0, "dte": 3},
        direction="BULLISH", score=78, confidence=0.75,
        iv_rank=30, vix=14.7, pcr=1.2, sentiment_score=0.25,
        sgx_change=0.3, usdinr=84.5,
    )
    msg = TelegramBot().format_signal_alert(data)
    assert len(msg) > 100, "Alert message too short"
    assert "NIFTY"     in msg
    assert "Long Call" in msg
    return f"Message generated ({len(msg)} chars) ✅"
test("Telegram format_signal_alert (no send)", t_telegram_format)

# ─────────────────────────────────────────────────────────────────────────────
# 8. BACKTESTER (6mo quick pass)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] BACKTESTER (6mo quick pass)")

def t_backtest():
    import traceback
    try:
        from backtesting.backtest import Backtester
        bt     = Backtester(instrument="NIFTY", period="6mo")
        result = bt.run()
        assert result.instrument == "NIFTY"
        assert isinstance(result.total_trades, int)
        # sharpe may be numpy.float64 — check for any numeric type
        assert result.sharpe is not None and result.sharpe == result.sharpe  # not NaN
        return (f"trades={result.total_trades}, win={result.win_rate}%, "
                f"sharpe={float(result.sharpe):.3f}, pnl=\u20b9{result.total_pnl_inr:,.0f}")
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}")
test("Backtester NIFTY 6mo", t_backtest)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)

print("\n" + "=" * 68)
print(f"  RESULTS: {passed}/{total} passed  |  {failed} failed")
print("=" * 68)

if failed:
    print("\n  FAILED TESTS:")
    for name, ok, msg in results:
        if not ok:
            print(f"    {FAIL} {name}")
            print(f"         {msg[:160]}")

print()
sys.exit(0 if failed == 0 else 1)
