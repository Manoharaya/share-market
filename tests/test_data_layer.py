"""
Data Layer Integration Test
Runs and verifies all data collector modules (MarketData, SentimentAnalyzer, GlobalMacro).
Logs results to logs/data_test.log.
"""
import os
import sys
import time
from loguru import logger

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import MarketData
from data.news_collector import SentimentAnalyzer
from data.global_macro import GlobalMacro

def run_tests():
    # Setup log file
    os.makedirs('logs', exist_ok=True)
    logger.add("logs/data_test.log", rotation="10 MB", level="INFO")
    
    logger.info("================ STARTING DATA LAYER INTEGRATION TESTS ================")
    
    md = MarketData()
    sa = SentimentAnalyzer()
    gm = GlobalMacro()
    
    success = True

    # ----------------------------------------------------
    # TEST 1: MarketData — LTP, VIX, option chain
    # ----------------------------------------------------
    logger.info("Running Test 1: MarketData Core Functionality...")
    try:
        time.sleep(0.5)
        ltp = md.get_ltp('NIFTY')
        logger.info(f"NIFTY LTP: {ltp}")
        assert ltp > 0, "LTP must be greater than 0"
        
        time.sleep(0.5)
        vix = md.get_india_vix()
        logger.info(f"India VIX: {vix}")
        assert vix > 0, "VIX must be greater than 0"
        
        time.sleep(0.5)
        expiries = md.get_expiry_dates('NIFTY')
        assert len(expiries) > 0, "Should return at least one expiry"
        logger.info(f"Option expiries available: {expiries[:5]}")
        
        time.sleep(0.5)
        chain = md.get_option_chain('NIFTY', expiries[0])
        assert not chain.empty, "Option chain should not be empty"
        logger.info(f"Option chain shape: {chain.shape}")
        
        logger.info("Test 1: PASSED")
    except Exception as e:
        logger.error(f"Test 1 FAILED: {e}")
        success = False

    # ----------------------------------------------------
    # TEST 2: SentimentAnalyzer — score between -1 and +1
    # ----------------------------------------------------
    logger.info("Running Test 2: SentimentAnalyzer...")
    try:
        # Test for NIFTY
        time.sleep(0.5)
        nifty_sent = sa.get_sentiment('NIFTY')
        logger.info(f"NIFTY news sentiment score: {nifty_sent}")
        assert -1.0 <= nifty_sent <= 1.0, "Sentiment score must be between -1.0 and +1.0"
        
        # Test for BANKNIFTY
        time.sleep(0.5)
        banknifty_sent = sa.get_sentiment('BANKNIFTY')
        logger.info(f"BANKNIFTY news sentiment score: {banknifty_sent}")
        assert -1.0 <= banknifty_sent <= 1.0, "Sentiment score must be between -1.0 and +1.0"
        
        logger.info("Test 2: PASSED")
    except Exception as e:
        logger.error(f"Test 2 FAILED: {e}")
        success = False

    # ----------------------------------------------------
    # TEST 3: GlobalMacro — dictionary structure and keys
    # ----------------------------------------------------
    logger.info("Running Test 3: GlobalMacro...")
    try:
        time.sleep(0.5)
        macro_signals = gm.get_signals()
        logger.info(f"Global Macro Signals: {macro_signals}")
        
        required_keys = ['gift_nifty', 'us_markets', 'crude_oil', 'usdinr', 'fii_dii']
        for key in required_keys:
            assert key in macro_signals, f"Missing key '{key}' in macro signals output"
            
        # Check sub-keys for us_markets
        assert 'dow' in macro_signals['us_markets'], "Missing 'dow' in us_markets"
        assert 'sp500' in macro_signals['us_markets'], "Missing 'sp500' in us_markets"
        assert 'nasdaq' in macro_signals['us_markets'], "Missing 'nasdaq' in us_markets"
        
        # Check sub-keys for fii_dii
        assert 'fii_net' in macro_signals['fii_dii'], "Missing 'fii_net' in fii_dii"
        assert 'dii_net' in macro_signals['fii_dii'], "Missing 'dii_net' in fii_dii"
        
        logger.info("Test 3: PASSED")
    except Exception as e:
        logger.error(f"Test 3 FAILED: {e}")
        success = False

    # ----------------------------------------------------
    # TEST 4: Full Chain Option chain calculations
    # ----------------------------------------------------
    logger.info("Running Test 4: Options Analysis Calculations...")
    try:
        time.sleep(0.5)
        expiries = md.get_expiry_dates('NIFTY')
        chain = md.get_option_chain('NIFTY', expiries[0])
        
        pcr = md.compute_pcr(chain)
        max_pain = md.compute_max_pain(chain)
        
        logger.info(f"PCR: {pcr:.4f}")
        logger.info(f"Max Pain Strike: {max_pain}")
        
        assert pcr >= 0.0, "PCR must be greater than or equal to 0"
        assert max_pain > 0.0, "Max pain strike must be greater than 0"
        
        logger.info("Test 4: PASSED")
    except Exception as e:
        logger.error(f"Test 4 FAILED: {e}")
        success = False

    logger.info("=======================================================================")
    if success:
        logger.info("ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ✅")
        sys.exit(0)
    else:
        logger.error("SOME INTEGRATION TESTS FAILED! ❌")
        sys.exit(1)

if __name__ == '__main__':
    run_tests()
