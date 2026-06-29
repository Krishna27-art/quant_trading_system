import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class OrderEvent(Base):
    __tablename__ = "order_events"

    event_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id = Column(String, index=True, nullable=False)
    parent_order_id = Column(String, index=True, nullable=True)
    event_type = Column(String, nullable=False)  # e.g. "ORDER_REQUESTED", "RISK_APPROVED"
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
