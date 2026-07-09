"""
Outcome Resolver

Resolves what happened to a prediction.
Determines if target was hit, stop was hit, timeout occurred, or trade was cancelled.
"""

from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime
import pandas as pd

from utils.logger import get_logger

logger = get_logger("continuous_learning.outcome_engine")


@dataclass
class OutcomeType:
    """Enumeration of outcome types."""
    TARGET_HIT: str = "target_hit"
    STOP_HIT: str = "stop_hit"
    TIMEOUT: str = "timeout"
    CANCELLED: str = "cancelled"
    PENDING: str = "pending"


@dataclass
class ResolvedOutcome:
    """Resolved outcome of a prediction."""
    outcome_type: str
    entry_price: float
    exit_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime]
    holding_period_days: Optional[int]
    max_favorable_excursion: float
    max_adverse_excursion: float
    return_percentage: float
    return_amount: float
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate resolved outcome.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check outcome type is valid
        valid_types = [
            OutcomeType.TARGET_HIT,
            OutcomeType.STOP_HIT,
            OutcomeType.TIMEOUT,
            OutcomeType.CANCELLED,
            OutcomeType.PENDING,
        ]
        if self.outcome_type not in valid_types:
            errors.append(f"Invalid outcome type: {self.outcome_type}")
        
        # Check entry price
        if self.entry_price <= 0:
            errors.append(f"Entry price must be positive, got {self.entry_price}")
        
        # Check return percentage is reasonable
        if abs(self.return_percentage) > 1.0:
            errors.append(f"Return percentage too extreme: {self.return_percentage}")
        
        # Check MFE and MAE are non-negative
        if self.max_favorable_excursion < 0:
            errors.append(f"MFE must be non-negative, got {self.max_favorable_excursion}")
        if self.max_adverse_excursion < 0:
            errors.append(f"MAE must be non-negative, got {self.max_adverse_excursion}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "outcome_type": self.outcome_type,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "target_price": round(self.target_price, 2) if self.target_price else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "holding_period_days": self.holding_period_days,
            "max_favorable_excursion": round(self.max_favorable_excursion, 4),
            "max_adverse_excursion": round(self.max_adverse_excursion, 4),
            "return_percentage": round(self.return_percentage, 4),
            "return_amount": round(self.return_amount, 2),
        }


class OutcomeResolver:
    """
    Resolves what happened to a prediction.
    
    Determines:
    - Target hit
    - Stop hit
    - Timeout
    - Cancelled
    - Holding period
    - MFE/MAE
    - Return
    """
    
    def __init__(self):
        """Initialize outcome resolver."""
        self._logger = get_logger("continuous_learning.outcome_engine")
    
    def resolve(
        self,
        entry_price: float,
        entry_timestamp: datetime,
        target_price: Optional[float],
        stop_loss: Optional[float],
        historical_data: pd.DataFrame,
        max_holding_days: int = 30,
        action: str = "BUY",
    ) -> ResolvedOutcome:
        """
        Resolve outcome from historical data.
        
        Args:
            entry_price: Entry price
            entry_timestamp: Entry timestamp
            target_price: Target price
            stop_loss: Stop loss price
            historical_data: Historical OHLCV data
            max_holding_days: Maximum holding period in days
            action: "BUY" or "SELL"
            
        Returns:
            ResolvedOutcome
        """
        # Filter data after entry
        historical_data = historical_data.copy()
        historical_data = historical_data[historical_data.index >= entry_timestamp]
        
        if len(historical_data) == 0:
            return self._pending_outcome(entry_price, entry_timestamp)
        
        # Calculate price changes
        if action == "BUY":
            historical_data["high_change"] = (historical_data["high"] - entry_price) / entry_price
            historical_data["low_change"] = (historical_data["low"] - entry_price) / entry_price
        else:  # SELL
            historical_data["high_change"] = (entry_price - historical_data["low"]) / entry_price
            historical_data["low_change"] = (entry_price - historical_data["high"]) / entry_price
        
        # Calculate target and stop changes
        if target_price:
            target_change = (target_price - entry_price) / entry_price if action == "BUY" else (entry_price - target_price) / entry_price
        else:
            target_change = None
        
        if stop_loss:
            stop_change = (stop_loss - entry_price) / entry_price if action == "BUY" else (entry_price - stop_loss) / entry_price
        else:
            stop_change = None
        
        # Check for target hit
        if target_change is not None:
            target_hit = historical_data[historical_data["high_change"] >= target_change]
            if len(target_hit) > 0:
                first_target = target_hit.iloc[0]
                return self._target_hit_outcome(
                    entry_price,
                    entry_timestamp,
                    target_price,
                    stop_loss,
                    first_target,
                    historical_data.loc[:first_target.name],
                    action,
                )
        
        # Check for stop hit
        if stop_change is not None:
            stop_hit = historical_data[historical_data["low_change"] <= stop_change]
            if len(stop_hit) > 0:
                first_stop = stop_hit.iloc[0]
                return self._stop_hit_outcome(
                    entry_price,
                    entry_timestamp,
                    target_price,
                    stop_loss,
                    first_stop,
                    historical_data.loc[:first_stop.name],
                    action,
                )
        
        # Check for timeout
        max_exit_date = entry_timestamp + pd.Timedelta(days=max_holding_days)
        timeout_data = historical_data[historical_data.index <= max_exit_date]
        
        if len(timeout_data) > 0:
            last_bar = timeout_data.iloc[-1]
            return self._timeout_outcome(
                entry_price,
                entry_timestamp,
                target_price,
                stop_loss,
                last_bar,
                timeout_data,
                action,
            )
        
        # Still pending
        return self._pending_outcome(entry_price, entry_timestamp)
    
    def _target_hit_outcome(
        self,
        entry_price: float,
        entry_timestamp: datetime,
        target_price: Optional[float],
        stop_loss: Optional[float],
        exit_bar: pd.Series,
        path_data: pd.DataFrame,
        action: str,
    ) -> ResolvedOutcome:
        """Create target hit outcome."""
        exit_price = target_price
        exit_timestamp = exit_bar.name
        
        # Calculate MFE and MAE
        if action == "BUY":
            mfe = path_data["high_change"].max()
            mae = abs(path_data["low_change"].min())
        else:
            mfe = path_data["high_change"].max()
            mae = abs(path_data["low_change"].min())
        
        # Calculate return
        return_pct = (exit_price - entry_price) / entry_price if action == "BUY" else (entry_price - exit_price) / entry_price
        return_amt = return_pct * entry_price
        
        # Calculate holding period
        holding_period = (exit_timestamp - entry_timestamp).days
        
        return ResolvedOutcome(
            outcome_type=OutcomeType.TARGET_HIT,
            entry_price=entry_price,
            exit_price=exit_price,
            target_price=target_price,
            stop_loss=stop_loss,
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
            holding_period_days=holding_period,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            return_percentage=return_pct,
            return_amount=return_amt,
        )
    
    def _stop_hit_outcome(
        self,
        entry_price: float,
        entry_timestamp: datetime,
        target_price: Optional[float],
        stop_loss: Optional[float],
        exit_bar: pd.Series,
        path_data: pd.DataFrame,
        action: str,
    ) -> ResolvedOutcome:
        """Create stop hit outcome."""
        exit_price = stop_loss
        exit_timestamp = exit_bar.name
        
        # Calculate MFE and MAE
        if action == "BUY":
            mfe = path_data["high_change"].max()
            mae = abs(path_data["low_change"].min())
        else:
            mfe = path_data["high_change"].max()
            mae = abs(path_data["low_change"].min())
        
        # Calculate return
        return_pct = (exit_price - entry_price) / entry_price if action == "BUY" else (entry_price - exit_price) / entry_price
        return_amt = return_pct * entry_price
        
        # Calculate holding period
        holding_period = (exit_timestamp - entry_timestamp).days
        
        return ResolvedOutcome(
            outcome_type=OutcomeType.STOP_HIT,
            entry_price=entry_price,
            exit_price=exit_price,
            target_price=target_price,
            stop_loss=stop_loss,
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
            holding_period_days=holding_period,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            return_percentage=return_pct,
            return_amount=return_amt,
        )
    
    def _timeout_outcome(
        self,
        entry_price: float,
        entry_timestamp: datetime,
        target_price: Optional[float],
        stop_loss: Optional[float],
        exit_bar: pd.Series,
        path_data: pd.DataFrame,
        action: str,
    ) -> ResolvedOutcome:
        """Create timeout outcome."""
        exit_price = exit_bar["close"]
        exit_timestamp = exit_bar.name
        
        # Calculate MFE and MAE
        if action == "BUY":
            mfe = path_data["high_change"].max()
            mae = abs(path_data["low_change"].min())
        else:
            mfe = path_data["high_change"].max()
            mae = abs(path_data["low_change"].min())
        
        # Calculate return
        return_pct = (exit_price - entry_price) / entry_price if action == "BUY" else (entry_price - exit_price) / entry_price
        return_amt = return_pct * entry_price
        
        # Calculate holding period
        holding_period = (exit_timestamp - entry_timestamp).days
        
        return ResolvedOutcome(
            outcome_type=OutcomeType.TIMEOUT,
            entry_price=entry_price,
            exit_price=exit_price,
            target_price=target_price,
            stop_loss=stop_loss,
            entry_timestamp=entry_timestamp,
            exit_timestamp=exit_timestamp,
            holding_period_days=holding_period,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            return_percentage=return_pct,
            return_amount=return_amt,
        )
    
    def _pending_outcome(
        self,
        entry_price: float,
        entry_timestamp: datetime,
    ) -> ResolvedOutcome:
        """Create pending outcome."""
        return ResolvedOutcome(
            outcome_type=OutcomeType.PENDING,
            entry_price=entry_price,
            exit_price=None,
            target_price=None,
            stop_loss=None,
            entry_timestamp=entry_timestamp,
            exit_timestamp=None,
            holding_period_days=None,
            max_favorable_excursion=0.0,
            max_adverse_excursion=0.0,
            return_percentage=0.0,
            return_amount=0.0,
        )


def resolve_outcome(
    entry_price: float,
    entry_timestamp: datetime,
    target_price: Optional[float],
    stop_loss: Optional[float],
    historical_data: pd.DataFrame,
    action: str = "BUY",
) -> ResolvedOutcome:
    """
    Convenience function to resolve outcome.
    
    Args:
        entry_price: Entry price
        entry_timestamp: Entry timestamp
        target_price: Target price
        stop_loss: Stop loss price
        historical_data: Historical OHLCV data
        action: "BUY" or "SELL"
        
    Returns:
        ResolvedOutcome
    """
    resolver = OutcomeResolver()
    return resolver.resolve(entry_price, entry_timestamp, target_price, stop_loss, historical_data, action=action)
