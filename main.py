"""
Main Entry Point - Master Scheduler
Runs options scanning, AI models, and signal generation.
"""
import sys
import time
import argparse
from datetime import datetime
from loguru import logger
import pandas as pd

from config import config
from database.signal_store import SignalStore
from data.market_data import MarketData
from data.historical_data import HistoricalData
from data.global_macro import GlobalMacro
from data.news_collector import NewsCollector
from features.technical import TechnicalIndicators
from features.options_features import OptionsFeatures
from features.sentiment import SentimentAnalyzer
from signals.strike_selector import StrikeSelector
from signals.strategy_selector import StrategySelector
from signals.scorer import SignalScorer
from alerts.alert_formatter import AlertFormatter
from alerts.telegram_bot import TelegramBot

def is_market_open() -> bool:
    """Checks if the current time is within trading hours (Mon-Fri)"""
    now = datetime.now()
    if now.weekday() >= 5:  # Saturday and Sunday
        return False
        
    try:
        open_hour, open_min = map(int, config.MARKET_OPEN.split(':'))
        close_hour, close_min = map(int, config.MARKET_CLOSE.split(':'))
        
        market_open = now.replace(hour=open_hour, minute=open_min, second=0, microsecond=0)
        market_close = now.replace(hour=close_hour, minute=close_min, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except Exception as e:
        logger.error(f"Error parsing market hours: {e}")
        return False

def run_scan(dry_run: bool = False):
    """Orchestrates the entire scan and alert pipeline"""
    logger.info("Initializing Options Signal Advisory System scan...")
    
    # Instantiate modules
    db = SignalStore()
    md = MarketData()
    hd = HistoricalData()
    macro = GlobalMacro()
    news = NewsCollector()
    ti = TechnicalIndicators()
    op_feat = OptionsFeatures()
    sentiment_ana = SentimentAnalyzer()
    strike_sel = StrikeSelector()
    strat_sel = StrategySelector()
    scorer = SignalScorer()
    formatter = AlertFormatter()
    bot = TelegramBot()

    # Fetch global/macro metrics once per scan
    macro_data = macro.get_macro_indicators()
    logger.info(f"Macro Indicators fetched: {macro_data}")
    
    headlines = news.get_latest_headlines()
    sentiment = sentiment_ana.calculate_sentiment(headlines)
    logger.info(f"News sentiment score: {sentiment:.2f} (from {len(headlines)} headlines)")

    vix_df = hd.get_vix_data()
    if not vix_df.empty:
        current_vix = float(vix_df['Close'].iloc[-1])
    else:
        current_vix = 15.0  # Safe default VIX
    logger.info(f"Current India VIX: {current_vix:.2f}")

    # Process each instrument
    for inst in config.INSTRUMENTS:
        logger.info(f"─── Scanning {inst} ───")
        
        # 1. Get Live Price
        ltp = md.get_ltp(inst)
        if ltp <= 0:
            logger.warning(f"Skipping {inst} due to invalid LTP: {ltp}")
            continue
        logger.info(f"{inst} Live Spot: {ltp:.2f}")
        
        # 2. Get Historical OHLCV & Project Live Price
        daily_df = hd.get_historical_daily(inst)
        if daily_df.empty:
            logger.warning(f"Skipping {inst} due to empty historical daily data")
            continue
            
        proj_df = daily_df.copy()
        today_date = pd.Timestamp.now().normalize()
        
        # Append live LTP as current day if not already in index
        if proj_df.index[-1].normalize() != today_date:
            new_row = pd.DataFrame({
                'Open': [ltp],
                'High': [ltp],
                'Low': [ltp],
                'Close': [ltp],
                'Volume': [0]
            }, index=[today_date])
            proj_df = pd.concat([proj_df, new_row])
            
        # Calculate technical indicators
        feat_df = ti.calculate_indicators(proj_df)
        if feat_df.empty:
            logger.warning(f"Failed to calculate technical indicators for {inst}")
            continue
            
        latest_tech = feat_df.iloc[-1].to_dict()
        atr = latest_tech.get('Atr', 0.0)
        
        # 3. Get Expiries & Option Chain
        expiries = md.get_expiry_dates(inst)
        if not expiries:
            logger.warning(f"No active expiries found for {inst}. Skipping options analysis.")
            continue
            
        nearest_expiry = expiries[0]
        logger.info(f"{inst} Nearest Expiry: {nearest_expiry}")
        
        chain = md.get_option_chain(inst, nearest_expiry)
        pcr = md.calculate_pcr(chain)
        max_pain = md.calculate_max_pain(chain)
        
        # 4. Options Features (IV Rank)
        iv_rank = op_feat.calculate_iv_rank(current_vix, vix_df)
        
        logger.info(f"{inst} Options: PCR={pcr:.2f}, Max Pain={max_pain:.2f}, IV Rank={iv_rank:.1f}%")
        
        # 5. Composite Direction & Score
        direction, score, confidence = scorer.calculate_score(
            latest_tech, pcr, iv_rank, current_vix, macro_data, sentiment
        )
        
        # 6. Strategy & Strike Leg Selection
        strategy = strat_sel.determine_strategy(direction, iv_rank)
        atm_strike = op_feat.get_atm_strike(ltp, inst)
        legs = strike_sel.select_strikes(inst, atm_strike, strategy, atr)
        
        logger.info(f"Signal: {direction} (Score: {score}/100, Conf: {confidence:.2f}, Strategy: {strategy})")
        
        # 7. Database Log
        db.save_signal(
            instrument=inst,
            strategy=strategy,
            direction=direction,
            score=score,
            confidence=confidence,
            iv_rank=iv_rank,
            vix=current_vix,
            sentiment=sentiment,
            alerted=False
        )
        
        # 8. Alert Dispatch Gating
        if score >= config.MIN_SIGNAL_SCORE and confidence >= config.MIN_CONFIDENCE:
            # Check if already alerted today
            today_signals = db.get_today_signals()
            already_alerted = any(s.instrument == inst and s.alerted for s in today_signals)
            
            if already_alerted and not dry_run:
                logger.info(f"Signal already alerted for {inst} today. Skipping duplicate alert.")
            else:
                indicators_dict = {
                    'pcr': pcr,
                    'vix': current_vix,
                    'iv_rank': iv_rank,
                    'rsi': latest_tech.get('Rsi', 50.0),
                    'sentiment': sentiment
                }
                msg = formatter.format_signal_message(
                    inst, ltp, direction, score, confidence, strategy, legs, indicators_dict
                )
                
                logger.info(f"Sending Telegram alert for {inst}...")
                success = bot.send_message(msg)
                if success:
                    db.mark_alerted(inst)
        else:
            logger.info(f"Signal for {inst} did not meet thresholds (Score: {score}/{config.MIN_SIGNAL_SCORE}, Conf: {confidence:.2f}/{config.MIN_CONFIDENCE})")

def main():
    parser = argparse.ArgumentParser(description="Options Signal Advisory System")
    parser.add_argument('--dry-run', action='store_true', help="Run a single-pass scan immediately and output logs")
    parser.add_argument('--loop', action='store_true', help="Start scheduler loop checking every 5 mins during market hours")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Executing Single Dry Run Scan...")
        run_scan(dry_run=True)
        logger.info("Dry run execution completed.")
        sys.exit(0)

    if args.loop or not len(sys.argv) > 1:
        logger.info(f"Starting Advisory Scheduler. Scanning every {config.SCAN_INTERVAL_MINUTES} minutes...")
        while True:
            try:
                if is_market_open():
                    logger.info("Market is open. Running scheduled scan.")
                    run_scan(dry_run=False)
                else:
                    logger.info("Market is closed. Scheduled scans run Mon-Fri during trading hours.")
                
                # Sleep interval
                time.sleep(config.SCAN_INTERVAL_MINUTES * 60)
            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user request.")
                break
            except Exception as e:
                logger.error(f"Error in scheduler execution: {e}")
                time.sleep(60) # Sleep for a minute before retry

if __name__ == '__main__':
    main()
