"""
Event Calendar
Hard-codes RBI MPC scheduled dates and auto-fetches FOMC dates from FRED.
"""
from datetime import date, timedelta
from loguru import logger
from config import config

# ── RBI MPC 2025-2026 scheduled dates (hard-coded) ───────────────────────────
# Source: RBI official calendar — updated manually each financial year
_RBI_MPC_DATES_2025_2026 = [
    date(2025, 4, 9),
    date(2025, 6, 6),
    date(2025, 8, 6),
    date(2025, 10, 8),
    date(2025, 12, 5),
    date(2026, 2, 7),
    date(2026, 4, 9),
    date(2026, 6, 6),
    date(2026, 8, 5),
    date(2026, 10, 7),
    date(2026, 12, 4),
]

# ── FOMC 2025-2026 scheduled dates (hard-coded; FRED fetch as enrichment) ────
_FOMC_DATES_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


class EventCalendar:
    def __init__(self):
        self._rbi_dates = list(_RBI_MPC_DATES_2025_2026)
        self._fomc_dates = list(_FOMC_DATES_2026)
        self._try_fetch_fomc_from_fred()

    def _try_fetch_fomc_from_fred(self):
        """Attempt to enrich FOMC dates from FRED API if key configured."""
        if not config.FRED_API_KEY:
            return
        try:
            from fredapi import Fred
            fred = Fred(api_key=config.FRED_API_KEY)
            # FEDfunds announcements — use series observation dates as proxy
            series = fred.get_series('FEDFUNDS')
            if series is not None and not series.empty:
                # Fred returns monthly data; extract unique month-end dates
                recent = series.loc[str(date.today().year - 1):]
                for ts in recent.index:
                    d = ts.date() if hasattr(ts, 'date') else ts
                    if d not in self._fomc_dates:
                        self._fomc_dates.append(d)
                logger.info(f"EventCalendar: enriched FOMC dates from FRED ({len(self._fomc_dates)} total)")
        except Exception as e:
            logger.warning(f"EventCalendar: could not fetch FOMC from FRED: {e}")

    def is_rbi_blackout(self, check_date: date = None) -> bool:
        """Returns True if check_date is an RBI MPC day or the day before."""
        if check_date is None:
            check_date = date.today()
        for mpc in self._rbi_dates:
            if check_date == mpc or check_date == mpc - timedelta(days=1):
                return True
        return False

    def is_fomc_blackout(self, check_date: date = None) -> bool:
        """Returns True if check_date is an FOMC meeting day or the day before."""
        if check_date is None:
            check_date = date.today()
        for fomc in self._fomc_dates:
            if check_date == fomc or check_date == fomc - timedelta(days=1):
                return True
        return False

    def is_blackout(self, check_date: date = None) -> tuple:
        """Returns (is_blackout: bool, reason: str)."""
        if check_date is None:
            check_date = date.today()
        if self.is_rbi_blackout(check_date):
            return True, f"RBI MPC blackout on {check_date}"
        if self.is_fomc_blackout(check_date):
            return True, f"FOMC meeting blackout on {check_date}"
        return False, ""

    def get_rbi_dates(self) -> list:
        return list(self._rbi_dates)

    def get_fomc_dates(self) -> list:
        return list(self._fomc_dates)
