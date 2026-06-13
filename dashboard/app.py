"""
Options Signal Advisory System — Live Dashboard
Streamlit app: real-time signals, option chain, macro & sentiment view.
Run: streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
from loguru import logger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Options Signal Advisory",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #8892b0; margin-top: 4px; }
    .bullish  { color: #00e676; }
    .bearish  { color: #ff5252; }
    .neutral  { color: #ffab40; }
    .gate-pass { color: #00e676; font-weight: 600; }
    .gate-fail { color: #ff5252; font-weight: 600; }
    .section-header {
        color: #ccd6f6;
        border-left: 4px solid #5c6bc0;
        padding-left: 10px;
        margin: 20px 0 10px 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_overview():
    """Returns dict with spot, vix, and basic chain stats for NIFTY."""
    try:
        from data.market_data import MarketData
        md = MarketData()
        spot  = md.get_ltp("NSE:NIFTY 50")
        vix   = md.get_india_vix()
        chain = md.get_option_chain("NIFTY")
        pcr   = md.compute_pcr(chain) if chain is not None else None
        mp    = md.compute_max_pain(chain) if chain is not None else None
        return {"spot": spot, "vix": vix, "pcr": pcr, "max_pain": mp,
                "chain": chain, "error": None}
    except Exception as e:
        return {"spot": None, "vix": None, "pcr": None, "max_pain": None,
                "chain": None, "error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_macro():
    try:
        from data.global_macro import GlobalMacro
        return GlobalMacro().get_macro_indicators()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=120, show_spinner=False)
def fetch_sentiment(instrument="NIFTY"):
    try:
        from data.news_collector import SentimentAnalyzer
        sa = SentimentAnalyzer()
        score = sa.get_sentiment_score(instrument)
        label = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")
        return {"score": score, "label": label}
    except Exception as e:
        return {"score": 0, "label": "N/A", "error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_iv_rank():
    try:
        from features.options_features import OptionsFeatures
        of = OptionsFeatures()
        return of.iv_rank_for_vix()
    except Exception as e:
        return None


def color_direction(val):
    if isinstance(val, str):
        v = val.upper()
        if v in ("BULLISH", "BUY", "GO", "PASS"):   return "bullish"
        if v in ("BEARISH", "SELL", "NO-GO", "FAIL"): return "bearish"
    return "neutral"


def fmt_change(val, suffix=""):
    if val is None: return "—"
    sign = "▲" if val >= 0 else "▼"
    color = "#00e676" if val >= 0 else "#ff5252"
    return f'<span style="color:{color}">{sign} {abs(val):.2f}{suffix}</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    instruments = st.multiselect(
        "Instruments to scan",
        ["NIFTY", "BANKNIFTY", "FINNIFTY"],
        default=["NIFTY"]
    )
    auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    show_chain   = st.toggle("Show Option Chain", value=True)
    show_rules   = st.toggle("Show Entry Gate Check", value=True)

    st.divider()
    st.markdown("**Thresholds**")
    try:
        from config import config
        st.caption(f"Min Score: **{config.MIN_SIGNAL_SCORE}**")
        st.caption(f"Min Conf:  **{config.MIN_CONFIDENCE}**")
        st.caption(f"Max VIX:   **{config.MAX_INDIA_VIX}**")
    except Exception:
        pass

    st.divider()
    refresh_btn = st.button("🔄 Refresh Now", use_container_width=True)

    st.markdown("---")
    now_ist = datetime.now()
    st.caption(f"⏰ {now_ist.strftime('%d %b %Y %H:%M:%S')} IST")
    is_market = "🟢 Market Open" if 9*60+15 <= now_ist.hour*60+now_ist.minute <= 14*60+30 else "🔴 Market Closed"
    st.caption(is_market)

if auto_refresh:
    st.rerun()

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#ccd6f6; font-size:1.8rem; margin-bottom:0'>
📊 Options Signal Advisory — Live Dashboard
</h1>
<p style='color:#8892b0; font-size:0.85rem; margin-top:4px'>
Personal use only · No auto-execution · NSE/BSE India
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Fetch data ─────────────────────────────────────────────────────────────────
with st.spinner("Fetching live market data…"):
    mkt   = fetch_market_overview()
    macro = fetch_macro()
    senti = fetch_sentiment(instruments[0] if instruments else "NIFTY")
    iv_rank = fetch_iv_rank()

# ── Top KPI row ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📌 Live Market Snapshot</p>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    spot = mkt.get("spot")
    st.metric("NIFTY Spot", f"₹{spot:,.0f}" if spot else "—")
with k2:
    vix = mkt.get("vix")
    vix_color = "normal" if vix and vix < 20 else "inverse"
    st.metric("India VIX", f"{vix:.1f}" if vix else "—", delta=None)
with k3:
    pcr = mkt.get("pcr")
    pcr_label = "Bullish" if pcr and pcr > 1.0 else "Bearish"
    st.metric("PCR (OI)", f"{pcr:.2f}" if pcr else "—", delta=pcr_label if pcr else None)
with k4:
    mp = mkt.get("max_pain")
    st.metric("Max Pain", f"₹{mp:,.0f}" if mp else "—")
with k5:
    iv_display = f"{iv_rank:.1f}%" if iv_rank is not None else "—"
    iv_env = "Low IV 🟢" if iv_rank and iv_rank < 35 else ("High IV 🔴" if iv_rank and iv_rank > 60 else "Mid IV 🟡")
    st.metric("IV Rank", iv_display, delta=iv_env if iv_rank else None)

st.divider()

# ── Macro + Sentiment row ──────────────────────────────────────────────────────
col_macro, col_senti = st.columns([2, 1])

with col_macro:
    st.markdown('<p class="section-header">🌐 Global Macro</p>', unsafe_allow_html=True)
    if macro and "error" not in macro:
        m_data = {
            "GIFT Nifty Chg %":  macro.get("gift_nifty", {}).get("change_pct", "—"),
            "USD/INR":           macro.get("usdinr", {}).get("rate", "—"),
            "Brent Crude $":     macro.get("crude", {}).get("price", "—"),
            "FII Flow (Cr)":     macro.get("fii", {}).get("net_flow_cr", "—"),
            "US Markets":        macro.get("us_markets", {}).get("sp500_chg", "—"),
        }
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        cols = [mc1, mc2, mc3, mc4, mc5]
        for col, (label, val) in zip(cols, m_data.items()):
            with col:
                if isinstance(val, float):
                    col.metric(label, f"{val:,.2f}")
                else:
                    col.metric(label, str(val))
    else:
        st.warning(f"Macro data unavailable: {macro.get('error','unknown')}")

with col_senti:
    st.markdown('<p class="section-header">📰 News Sentiment</p>', unsafe_allow_html=True)
    score = senti.get("score", 0)
    label = senti.get("label", "N/A")
    css   = color_direction(label)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"color": "#ccd6f6"}, "valueformat": ".2f"},
        gauge={
            "axis": {"range": [-1, 1], "tickcolor": "#8892b0"},
            "bar":  {"color": "#5c6bc0"},
            "steps": [
                {"range": [-1, -0.1], "color": "#3b1c1c"},
                {"range": [-0.1, 0.1],"color": "#2a2a1a"},
                {"range": [0.1,  1],  "color": "#1c3b1c"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 2}, "value": score},
        },
        title={"text": f"<b>{label}</b>", "font": {"color": "#ccd6f6", "size": 14}},
    ))
    gauge.update_layout(
        height=200, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#1e2130", font_color="#ccd6f6"
    )
    st.plotly_chart(gauge, use_container_width=True)

st.divider()

# ── Signal Scanner ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">🤖 Signal Scanner</p>', unsafe_allow_html=True)

for instrument in instruments:
    with st.expander(f"📊 {instrument}", expanded=True):
        with st.spinner(f"Running full pipeline for {instrument}…"):
            try:
                # --- Features
                from data.historical_data import HistoricalData
                from features.technical import TechnicalFeatures
                hd  = HistoricalData()
                sym_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "^CNXFIN"}
                df  = hd.get_ohlcv(sym_map.get(instrument, "^NSEI"), period="1y")
                tf  = TechnicalFeatures()
                features = tf.compute(df) if df is not None and not df.empty else None

                # --- ML Prediction
                pred = None
                try:
                    from models.ensemble import EnsemblePredictor
                    ep = EnsemblePredictor()
                    if features is not None:
                        pred = ep.predict(features.iloc[-1:].values)
                except Exception as e:
                    pred = {"direction": "NEUTRAL", "probability": 0.5,
                            "confidence": 0.5, "model_scores": {}}

                # --- Score
                score_obj = None
                try:
                    from signals.scorer import SignalScorer
                    scorer = SignalScorer()
                    last_feat = features.iloc[-1].to_dict() if features is not None else {}
                    macro_inputs = {
                        "sgx_change":    macro.get("gift_nifty", {}).get("change_pct", 0.0) if macro else 0.0,
                        "fii_flow":      macro.get("fii", {}).get("net_flow_cr", 0.0)       if macro else 0.0,
                        "usdinr_change": macro.get("usdinr", {}).get("change_pct", 0.0)    if macro else 0.0,
                    }
                    score_obj = scorer.score_all(
                        features_dict   = last_feat,
                        prediction_dict = pred or {},
                        iv_rank         = iv_rank or 50.0,
                        pcr             = mkt.get("pcr") or 1.0,
                        gex             = 0.0,
                        sentiment_score = senti.get("score", 0.0),
                        macro_inputs    = macro_inputs,
                        symbol          = instrument,
                    )
                except Exception as e:
                    pass

                # --- Entry gate
                entry_decision = None
                if show_rules:
                    try:
                        from signals.rules_engine import RulesEngine
                        re = RulesEngine()
                        entry_decision = re.check_entry_conditions(
                            instrument=instrument,
                            score=score_obj.total if score_obj else 50,
                            confidence=pred.get("confidence", 0.5) if pred else 0.5,
                            vix=vix or 15.0,
                            iv_rank=iv_rank or 50.0,
                            strategy=pred.get("strategy", "") if pred else "",
                            oi_at_strike=9999,
                            bid_ask_spread=0.10,
                            mid_price=100.0,
                        )
                    except Exception as e:
                        pass

                # ── Layout
                sc1, sc2, sc3, sc4 = st.columns(4)
                total = score_obj.total if score_obj else "—"
                direction = (pred.get("direction", "NEUTRAL") if pred else "NEUTRAL")
                conf = pred.get("confidence", 0) if pred else 0
                css = color_direction(direction)

                sc1.metric("Signal Score",  f"{total}/100" if total != "—" else "—")
                sc2.metric("Direction",     direction)
                sc3.metric("Confidence",    f"{conf:.1%}")
                sc4.metric("Entry Gate",
                           "GO ✅"   if entry_decision and entry_decision.approved else
                           "NO-GO ❌" if entry_decision else "—")

                # ── Score breakdown bar chart
                if score_obj:
                    dims = {
                        "Technical": score_obj.technical,
                        "ML Model":  score_obj.ml,
                        "Options":   score_obj.options,
                        "Sentiment": score_obj.sentiment,
                        "Macro":     score_obj.macro,
                        "Fundamental": score_obj.fundamental,
                    }
                    maxes = {"Technical":25,"ML Model":25,"Options":20,
                             "Sentiment":15,"Macro":10,"Fundamental":5}
                    df_dims = pd.DataFrame({
                        "Dimension": list(dims.keys()),
                        "Score":     list(dims.values()),
                        "Max":       [maxes[k] for k in dims],
                    })
                    df_dims["Pct"] = df_dims["Score"] / df_dims["Max"] * 100
                    fig = px.bar(df_dims, x="Dimension", y="Score",
                                 color="Pct",
                                 color_continuous_scale=["#ff5252","#ffab40","#00e676"],
                                 range_color=[0,100],
                                 text="Score",
                                 title=f"{instrument} — Signal Score Breakdown")
                    fig.update_layout(
                        paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                        font_color="#ccd6f6", height=300,
                        margin=dict(t=40, b=20, l=20, r=20),
                        showlegend=False,
                        coloraxis_showscale=False,
                    )
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

                # ── Entry gate detail
                if show_rules and entry_decision:
                    st.markdown("**Entry Gate Results:**")
                    gate_cols = st.columns(3)
                    for idx, gate in enumerate(entry_decision.gates):
                        with gate_cols[idx % 3]:
                            icon = "✅" if gate.passed else "❌"
                            color = "#00e676" if gate.passed else "#ff5252"
                            st.markdown(
                                f'<span style="color:{color}">{icon} **{gate.gate}**</span><br>'
                                f'<span style="font-size:0.75rem;color:#8892b0">{gate.reason[:60]}</span>',
                                unsafe_allow_html=True
                            )

            except Exception as e:
                st.error(f"Pipeline error for {instrument}: {e}")
                logger.exception(e)

st.divider()

# ── Option Chain Viewer ────────────────────────────────────────────────────────
if show_chain:
    st.markdown('<p class="section-header">📋 Option Chain — NIFTY</p>', unsafe_allow_html=True)
    chain = mkt.get("chain")
    if chain is not None and not chain.empty:
        spot_val = mkt.get("spot") or 0
        # Show ±5 strikes around ATM
        strikes = chain["strike"].unique()
        atm = min(strikes, key=lambda x: abs(x - spot_val))
        atm_idx = list(strikes).index(atm)
        nearby = strikes[max(0, atm_idx-5): atm_idx+6]
        filtered = chain[chain["strike"].isin(nearby)].copy()

        # Pivot: CE vs PE side by side
        ce = filtered[filtered["option_type"]=="CE"][
            ["strike","lastPrice","openInterest","iv","delta"]
        ].rename(columns={"lastPrice":"CE LTP","openInterest":"CE OI","iv":"CE IV","delta":"CE Δ"})
        pe = filtered[filtered["option_type"]=="PE"][
            ["strike","lastPrice","openInterest","iv","delta"]
        ].rename(columns={"lastPrice":"PE LTP","openInterest":"PE OI","iv":"PE IV","delta":"PE Δ"})

        chain_view = ce.merge(pe, on="strike", how="outer").sort_values("strike")

        def highlight_atm(row):
            if row["strike"] == atm:
                return ["background-color: #252840"] * len(row)
            return [""] * len(row)

        st.dataframe(
            chain_view.style.apply(highlight_atm, axis=1).format({
                "CE LTP":"₹{:.2f}","PE LTP":"₹{:.2f}",
                "CE OI":"{:,.0f}","PE OI":"{:,.0f}",
                "CE IV":"{:.1f}%","PE IV":"{:.1f}%",
                "CE Δ":"{:.3f}","PE Δ":"{:.3f}",
            }),
            use_container_width=True, height=350
        )
        # OI bar chart
        oi_data = filtered.groupby(["strike","option_type"])["openInterest"].sum().reset_index()
        fig_oi = px.bar(oi_data, x="strike", y="openInterest", color="option_type",
                        barmode="group",
                        color_discrete_map={"CE":"#00e676","PE":"#ff5252"},
                        title="Open Interest by Strike",
                        labels={"openInterest":"Open Interest","strike":"Strike"})
        fig_oi.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#ccd6f6", height=300,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_oi, use_container_width=True)
    else:
        st.info("Option chain unavailable — market may be closed or API limit reached.")

st.divider()

# ── Historical VIX Chart ───────────────────────────────────────────────────────
st.markdown('<p class="section-header">📈 India VIX — 6 Month History</p>', unsafe_allow_html=True)
try:
    import yfinance as yf
    vix_hist = yf.download("^INDIAVIX", period="6mo", auto_adjust=True, progress=False)
    if not vix_hist.empty:
        fig_vix = go.Figure()
        fig_vix.add_trace(go.Scatter(
            x=vix_hist.index, y=vix_hist["Close"].squeeze(),
            fill="tozeroy",
            line=dict(color="#5c6bc0", width=1.5),
            fillcolor="rgba(92,107,192,0.15)",
            name="India VIX"
        ))
        fig_vix.add_hline(y=20, line_dash="dash", line_color="#ffab40",
                          annotation_text="Caution (20)", annotation_position="right")
        fig_vix.add_hline(y=30, line_dash="dash", line_color="#ff5252",
                          annotation_text="No-Trade (30)", annotation_position="right")
        fig_vix.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#ccd6f6", height=280,
            margin=dict(t=20, b=20, l=20, r=60),
            xaxis=dict(gridcolor="#2e3250"),
            yaxis=dict(gridcolor="#2e3250"),
        )
        st.plotly_chart(fig_vix, use_container_width=True)
except Exception as e:
    st.caption(f"VIX chart unavailable: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:0.75rem; padding:8px'>
⚠️ This system is for personal educational use only. No signals constitute financial advice.
All trades must be manually placed and verified. Not SEBI registered. NSE/BSE India.
</div>
""", unsafe_allow_html=True)
