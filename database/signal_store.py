from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, time
from config import config
from database.models import Base, Signal

class SignalStore:
    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_signal(self, instrument: str, strategy: str, direction: str,
                    score: int, confidence: float, iv_rank: float,
                    vix: float, sentiment: float, alerted: bool = False) -> Signal:
        """Saves a new trade signal to the database."""
        session = self.Session()
        try:
            signal = Signal(
                instrument=instrument,
                strategy=strategy,
                direction=direction,
                score=score,
                confidence=confidence,
                iv_rank=iv_rank,
                vix=vix,
                sentiment=sentiment,
                alerted=alerted
            )
            session.add(signal)
            session.commit()
            # Refresh to load id/timestamp
            session.refresh(signal)
            # Detach from session to allow use outside session scope
            session.expunge(signal)
            return signal
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_today_signals(self) -> list[Signal]:
        """Fetches all signals generated today."""
        session = self.Session()
        try:
            today_start = datetime.combine(datetime.today(), time.min)
            today_end = datetime.combine(datetime.today(), time.max)
            
            signals = session.query(Signal).filter(
                Signal.timestamp >= today_start,
                Signal.timestamp <= today_end
            ).all()
            
            # Detach signals from session
            for s in signals:
                session.expunge(s)
            return signals
        finally:
            session.close()

    def count_today(self) -> int:
        """Returns count of signals generated today."""
        session = self.Session()
        try:
            today_start = datetime.combine(datetime.today(), time.min)
            today_end = datetime.combine(datetime.today(), time.max)
            
            return session.query(Signal).filter(
                Signal.timestamp >= today_start,
                Signal.timestamp <= today_end
            ).count()
        finally:
            session.close()

    def mark_alerted(self, instrument: str):
        """Marks the latest signal for the given instrument as alerted."""
        session = self.Session()
        try:
            # Get latest signal for this instrument
            signal = session.query(Signal).filter(
                Signal.instrument == instrument
            ).order_by(Signal.timestamp.desc()).first()
            
            if signal:
                signal.alerted = True
                session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
