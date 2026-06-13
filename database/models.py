from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Signal(Base):
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    instrument = Column(String(50), nullable=False)
    strategy = Column(String(100), nullable=False)
    direction = Column(String(20), nullable=False)
    score = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    iv_rank = Column(Float, nullable=False)
    vix = Column(Float, nullable=False)
    sentiment = Column(Float, nullable=False)
    alerted = Column(Boolean, default=False, nullable=False)

    def __repr__(self):
        return (f"<Signal(id={self.id}, instrument='{self.instrument}', "
                f"strategy='{self.strategy}', direction='{self.direction}', "
                f"score={self.score}, confidence={self.confidence}, alerted={self.alerted})>")
