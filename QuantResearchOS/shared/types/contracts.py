from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class OHLCVRecord(BaseModel):
    timestamp: datetime
    symbol: str = Field(..., min_length=1)
    exchange: str = Field(default="NSE")
    open: float = Field(..., ge=0.0)
    high: float = Field(..., ge=0.0)
    low: float = Field(..., ge=0.0)
    close: float = Field(..., ge=0.0)
    volume: int = Field(..., ge=0)

    @field_validator("high")
    @classmethod
    def high_must_be_gte_low(cls, v, info):
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("high price cannot be less than low price")
        return v


class MarketDataBatch(BaseModel):
    records: list[OHLCVRecord]
