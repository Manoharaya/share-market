"""
Strike Selector
Selects option contract legs matching target deltas and calculates payoff metrics.
"""
import calendar
from datetime import date, timedelta
import numpy as np
import pandas as pd
from loguru import logger

class StrikeSelector:
    def next_weekly_expiry(self) -> date:
        """Returns the upcoming Thursday's date"""
        today = date.today()
        # Thursday is weekday 3
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)

    def next_monthly_expiry(self) -> date:
        """Returns the last Thursday date of the current month (or next month if already passed)"""
        today = date.today()
        year = today.year
        month = today.month
        
        last_day = calendar.monthrange(year, month)[1]
        d = date(year, month, last_day)
        while d.weekday() != 3:  # Thursday
            d -= timedelta(days=1)
            
        # If last Thursday of current month has already passed, fetch next month's
        if d < today:
            month += 1
            if month > 12:
                month = 1
                year += 1
            last_day = calendar.monthrange(year, month)[1]
            d = date(year, month, last_day)
            while d.weekday() != 3:
                d -= timedelta(days=1)
        return d

    def find_closest_strike(self, df: pd.DataFrame, option_type: str, target_delta: float) -> tuple:
        """Finds the strike closest to target delta for the given option type"""
        sub = df[df['type'].str.upper() == option_type.upper()].copy()
        if sub.empty:
            return 0.0, 0.0
            
        sub['diff'] = (sub['delta'].abs() - abs(target_delta)).abs()
        closest = sub.loc[sub['diff'].idxmin()]
        return float(closest['strike']), float(closest['ltp'])

    def build_legs(self, strategy: str, chain_df: pd.DataFrame, spot: float) -> dict:
        """Finds actual strikes and premium costs matching delta targets for the strategy legs"""
        if chain_df.empty or spot <= 0:
            return {}

        legs = {}
        strategy = strategy.strip().title()

        try:
            if strategy == "Long Call":
                strike, premium = self.find_closest_strike(chain_df, 'CE', 0.50)
                legs['buy_call'] = {'strike': strike, 'premium': premium}
            elif strategy == "Long Put":
                strike, premium = self.find_closest_strike(chain_df, 'PE', -0.50)
                legs['buy_put'] = {'strike': strike, 'premium': premium}
            elif strategy == "Bull Call Spread":
                strike_b, prem_b = self.find_closest_strike(chain_df, 'CE', 0.50)
                strike_s, prem_s = self.find_closest_strike(chain_df, 'CE', 0.30)
                legs['buy_call'] = {'strike': strike_b, 'premium': prem_b}
                legs['sell_call'] = {'strike': strike_s, 'premium': prem_s}
            elif strategy == "Bear Put Spread":
                strike_b, prem_b = self.find_closest_strike(chain_df, 'PE', -0.50)
                strike_s, prem_s = self.find_closest_strike(chain_df, 'PE', -0.30)
                legs['buy_put'] = {'strike': strike_b, 'premium': prem_b}
                legs['sell_put'] = {'strike': strike_s, 'premium': prem_s}
            elif strategy == "Bull Put Spread":
                strike_s, prem_s = self.find_closest_strike(chain_df, 'PE', -0.30)
                strike_b, prem_b = self.find_closest_strike(chain_df, 'PE', -0.15)
                legs['sell_put'] = {'strike': strike_s, 'premium': prem_s}
                legs['buy_put'] = {'strike': strike_b, 'premium': prem_b}
            elif strategy == "Bear Call Spread":
                strike_s, prem_s = self.find_closest_strike(chain_df, 'CE', 0.30)
                strike_b, prem_b = self.find_closest_strike(chain_df, 'CE', 0.15)
                legs['sell_call'] = {'strike': strike_s, 'premium': prem_s}
                legs['buy_call'] = {'strike': strike_b, 'premium': prem_b}
            elif strategy == "Iron Condor":
                strike_bp, prem_bp = self.find_closest_strike(chain_df, 'PE', -0.15)
                strike_sp, prem_sp = self.find_closest_strike(chain_df, 'PE', -0.30)
                strike_sc, prem_sc = self.find_closest_strike(chain_df, 'CE', 0.30)
                strike_bc, prem_bc = self.find_closest_strike(chain_df, 'CE', 0.15)
                legs['buy_put'] = {'strike': strike_bp, 'premium': prem_bp}
                legs['sell_put'] = {'strike': strike_sp, 'premium': prem_sp}
                legs['sell_call'] = {'strike': strike_sc, 'premium': prem_sc}
                legs['buy_call'] = {'strike': strike_bc, 'premium': prem_bc}
            elif strategy == "Iron Butterfly":
                strike_sc, prem_sc = self.find_closest_strike(chain_df, 'CE', 0.50)
                strike_sp, prem_sp = self.find_closest_strike(chain_df, 'PE', -0.50)
                strike_bc, prem_bc = self.find_closest_strike(chain_df, 'CE', 0.15)
                strike_bp, prem_bp = self.find_closest_strike(chain_df, 'PE', -0.15)
                legs['sell_call'] = {'strike': strike_sc, 'premium': prem_sc}
                legs['sell_put'] = {'strike': strike_sp, 'premium': prem_sp}
                legs['buy_call'] = {'strike': strike_bc, 'premium': prem_bc}
                legs['buy_put'] = {'strike': strike_bp, 'premium': prem_bp}
            elif strategy == "Straddle":
                strike_sc, prem_sc = self.find_closest_strike(chain_df, 'CE', 0.50)
                strike_sp, prem_sp = self.find_closest_strike(chain_df, 'PE', -0.50)
                legs['sell_call'] = {'strike': strike_sc, 'premium': prem_sc}
                legs['sell_put'] = {'strike': strike_sp, 'premium': prem_sp}
            elif strategy == "Strangle":
                strike_sc, prem_sc = self.find_closest_strike(chain_df, 'CE', 0.20)
                strike_sp, prem_sp = self.find_closest_strike(chain_df, 'PE', -0.20)
                legs['sell_call'] = {'strike': strike_sc, 'premium': prem_sc}
                legs['sell_put'] = {'strike': strike_sp, 'premium': prem_sp}
            elif strategy == "Calendar Spread":
                strike_sc, prem_sc = self.find_closest_strike(chain_df, 'CE', 0.50)
                legs['sell_near_call'] = {'strike': strike_sc, 'premium': prem_sc}
                legs['buy_far_call'] = {'strike': strike_sc, 'premium': prem_sc * 1.5}  # proxy
        except Exception as e:
            logger.error(f"Error building legs for strategy {strategy}: {e}")

        return legs

    def compute_payoff(self, strategy: str, legs: dict) -> dict:
        """Calculates net cost, max profit, max loss, and breakevens for the legs"""
        if not legs:
            return {}

        payoff = {
            'net_premium': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'breakevens': []
        }
        strategy = strategy.strip().title()

        try:
            if strategy == "Long Call":
                net_debit = legs['buy_call']['premium']
                payoff['net_premium'] = -net_debit
                payoff['max_profit'] = float('inf')
                payoff['max_loss'] = net_debit
                payoff['breakevens'] = [legs['buy_call']['strike'] + net_debit]
            elif strategy == "Long Put":
                net_debit = legs['buy_put']['premium']
                payoff['net_premium'] = -net_debit
                payoff['max_profit'] = legs['buy_put']['strike'] - net_debit
                payoff['max_loss'] = net_debit
                payoff['breakevens'] = [legs['buy_put']['strike'] - net_debit]
            elif strategy == "Bull Call Spread":
                net_debit = legs['buy_call']['premium'] - legs['sell_call']['premium']
                payoff['net_premium'] = -net_debit
                payoff['max_profit'] = (legs['sell_call']['strike'] - legs['buy_call']['strike']) - net_debit
                payoff['max_loss'] = net_debit
                payoff['breakevens'] = [legs['buy_call']['strike'] + net_debit]
            elif strategy == "Bear Put Spread":
                net_debit = legs['buy_put']['premium'] - legs['sell_put']['premium']
                payoff['net_premium'] = -net_debit
                payoff['max_profit'] = (legs['buy_put']['strike'] - legs['sell_put']['strike']) - net_debit
                payoff['max_loss'] = net_debit
                payoff['breakevens'] = [legs['buy_put']['strike'] - net_debit]
            elif strategy == "Bull Put Spread":
                net_credit = legs['sell_put']['premium'] - legs['buy_put']['premium']
                payoff['net_premium'] = net_credit
                payoff['max_profit'] = net_credit
                payoff['max_loss'] = (legs['sell_put']['strike'] - legs['buy_put']['strike']) - net_credit
                payoff['breakevens'] = [legs['sell_put']['strike'] - net_credit]
            elif strategy == "Bear Call Spread":
                net_credit = legs['sell_call']['premium'] - legs['buy_call']['premium']
                payoff['net_premium'] = net_credit
                payoff['max_profit'] = net_credit
                payoff['max_loss'] = (legs['buy_call']['strike'] - legs['sell_call']['strike']) - net_credit
                payoff['breakevens'] = [legs['sell_call']['strike'] + net_credit]
            elif strategy == "Iron Condor":
                net_credit = (legs['sell_put']['premium'] + legs['sell_call']['premium']) - (legs['buy_put']['premium'] + legs['buy_call']['premium'])
                payoff['net_premium'] = net_credit
                payoff['max_profit'] = net_credit
                width = max(
                    legs['sell_put']['strike'] - legs['buy_put']['strike'],
                    legs['buy_call']['strike'] - legs['sell_call']['strike']
                )
                payoff['max_loss'] = width - net_credit
                payoff['breakevens'] = [
                    legs['sell_put']['strike'] - net_credit,
                    legs['sell_call']['strike'] + net_credit
                ]
            elif strategy == "Iron Butterfly":
                net_credit = (legs['sell_call']['premium'] + legs['sell_put']['premium']) - (legs['buy_call']['premium'] + legs['buy_put']['premium'])
                payoff['net_premium'] = net_credit
                payoff['max_profit'] = net_credit
                width = legs['sell_call']['strike'] - legs['buy_put']['strike']
                payoff['max_loss'] = width - net_credit
                payoff['breakevens'] = [
                    legs['sell_put']['strike'] - net_credit,
                    legs['sell_call']['strike'] + net_credit
                ]
            elif strategy == "Straddle":
                net_credit = legs['sell_call']['premium'] + legs['sell_put']['premium']
                payoff['net_premium'] = net_credit
                payoff['max_profit'] = net_credit
                payoff['max_loss'] = float('inf')
                payoff['breakevens'] = [
                    legs['sell_call']['strike'] - net_credit,
                    legs['sell_call']['strike'] + net_credit
                ]
            elif strategy == "Strangle":
                net_credit = legs['sell_call']['premium'] + legs['sell_put']['premium']
                payoff['net_premium'] = net_credit
                payoff['max_profit'] = net_credit
                payoff['max_loss'] = float('inf')
                payoff['breakevens'] = [
                    legs['sell_put']['strike'] - net_credit,
                    legs['sell_call']['strike'] + net_credit
                ]
            elif strategy == "Calendar Spread":
                net_debit = legs['buy_far_call']['premium'] - legs['sell_near_call']['premium']
                payoff['net_premium'] = -net_debit
                payoff['max_profit'] = float('nan')
                payoff['max_loss'] = net_debit
                payoff['breakevens'] = []
        except Exception as e:
            logger.error(f"Error calculating payoff for strategy {strategy}: {e}")

        return payoff
