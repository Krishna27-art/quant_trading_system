"""
Signal Performance Tracker

Tracks historical performance of individual signals and signal combinations.
Calculates win rates, average returns, and other metrics for each signal.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from signal_engine.base import SignalCategory
from utils.logger import get_logger

logger = get_logger("signal_engine.performance")


@dataclass
class SignalPerformance:
    """Performance metrics for a single signal."""
    category: SignalCategory
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_return: float
    average_win: float
    average_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    last_updated: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'category': self.category.value,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 4),
            'average_return': round(self.average_return, 4),
            'average_win': round(self.average_win, 4),
            'average_loss': round(self.average_loss, 4),
            'profit_factor': round(self.profit_factor, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'sortino_ratio': round(self.sortino_ratio, 4),
            'last_updated': self.last_updated.isoformat(),
        }


@dataclass
class TradeRecord:
    """Record of a single trade."""
    symbol: str
    signal_category: SignalCategory
    signal_direction: str
    entry_price: float
    exit_price: float
    return_pct: float
    entry_time: datetime
    exit_time: datetime
    holding_period_days: int


class SignalPerformanceTracker:
    """
    Signal Performance Tracker.
    
    Tracks:
    - Win rate per signal category
    - Average return per signal category
    - Profit factor
    - Maximum drawdown
    - Sharpe ratio
    - Sortino ratio
    
    Helps identify which signals actually work.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        self.trade_records: List[TradeRecord] = []
        self.signal_performance: Dict[SignalCategory, SignalPerformance] = {}
        self._logger = get_logger("signal_engine.performance")
    
    def record_trade(
        self,
        symbol: str,
        signal_category: SignalCategory,
        signal_direction: str,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
    ) -> None:
        """
        Record a completed trade.
        
        Args:
            symbol: Stock symbol
            signal_category: Category of the signal
            signal_direction: Direction of the signal (BULLISH/BEARISH)
            entry_price: Entry price
            exit_price: Exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
        """
        # Calculate return
        if signal_direction == "BULLISH":
            return_pct = (exit_price - entry_price) / entry_price * 100
        else:
            return_pct = (entry_price - exit_price) / entry_price * 100
        
        # Calculate holding period
        holding_period = (exit_time - entry_time).days
        
        # Create trade record
        trade_record = TradeRecord(
            symbol=symbol,
            signal_category=signal_category,
            signal_direction=signal_direction,
            entry_price=entry_price,
            exit_price=exit_price,
            return_pct=return_pct,
            entry_time=entry_time,
            exit_time=exit_time,
            holding_period_days=holding_period,
        )
        
        self.trade_records.append(trade_record)
        
        # Update performance metrics
        self._update_performance_metrics(signal_category)
        
        self._logger.info(
            f"Recorded trade: {symbol} {signal_category.value} "
            f"return={return_pct:.2f}% direction={signal_direction}"
        )
    
    def _update_performance_metrics(self, category: SignalCategory) -> None:
        """Update performance metrics for a signal category."""
        # Filter trades for this category
        category_trades = [
            trade for trade in self.trade_records
            if trade.signal_category == category
        ]
        
        if not category_trades:
            return
        
        # Calculate basic metrics
        total_trades = len(category_trades)
        winning_trades = sum(1 for trade in category_trades if trade.return_pct > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        returns = [trade.return_pct for trade in category_trades]
        average_return = sum(returns) / total_trades if total_trades > 0 else 0.0
        
        winning_returns = [trade.return_pct for trade in category_trades if trade.return_pct > 0]
        losing_returns = [trade.return_pct for trade in category_trades if trade.return_pct < 0]
        
        average_win = sum(winning_returns) / len(winning_returns) if winning_returns else 0.0
        average_loss = sum(losing_returns) / len(losing_returns) if losing_returns else 0.0
        
        # Profit factor
        total_wins = sum(winning_returns)
        total_losses = abs(sum(losing_returns))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Maximum drawdown
        cumulative_returns = []
        cumulative = 0.0
        for ret in returns:
            cumulative += ret / 100  # Convert to decimal
            cumulative_returns.append(cumulative)
        
        if cumulative_returns:
            peak = max(cumulative_returns)
            drawdowns = [peak - x for x in cumulative_returns]
            max_drawdown = max(drawdowns) * 100  # Convert back to percentage
        else:
            max_drawdown = 0.0
        
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        if len(returns) > 1:
            import numpy as np
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            
            # Sortino ratio (using downside deviation)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_deviation = (sum(r**2 for r in downside_returns) / len(downside_returns)) ** 0.5
                sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0.0
            else:
                sortino_ratio = 0.0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
        
        # Update performance
        self.signal_performance[category] = SignalPerformance(
            category=category,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            average_return=average_return,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            last_updated=datetime.now(),
        )
    
    def get_performance(self, category: SignalCategory) -> Optional[SignalPerformance]:
        """Get performance metrics for a signal category."""
        return self.signal_performance.get(category)
    
    def get_all_performance(self) -> Dict[SignalCategory, SignalPerformance]:
        """Get performance metrics for all signal categories."""
        return self.signal_performance
    
    def get_best_performing_signal(self, metric: str = 'win_rate') -> Optional[SignalCategory]:
        """
        Get the best performing signal category by metric.
        
        Args:
            metric: Metric to compare (win_rate, average_return, profit_factor, sharpe_ratio)
            
        Returns:
            Best performing signal category
        """
        if not self.signal_performance:
            return None
        
        best_category = None
        best_value = -float('inf')
        
        for category, performance in self.signal_performance.items():
            value = getattr(performance, metric, 0)
            if value > best_value:
                best_value = value
                best_category = category
        
        return best_category
    
    def get_performance_summary(self) -> Dict:
        """Get summary of all signal performance."""
        summary = {
            'total_trades': len(self.trade_records),
            'categories_tracked': len(self.signal_performance),
            'category_performance': {},
        }
        
        for category, performance in self.signal_performance.items():
            summary['category_performance'][category.value] = performance.to_dict()
        
        return summary
    
    def get_recent_trades(
        self,
        days: int = 30,
        category: Optional[SignalCategory] = None,
    ) -> List[TradeRecord]:
        """
        Get recent trades within a time window.
        
        Args:
            days: Number of days to look back
            category: Optional category filter
            
        Returns:
            List of recent trade records
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_trades = [
            trade for trade in self.trade_records
            if trade.exit_time >= cutoff_date
        ]
        
        if category:
            recent_trades = [
                trade for trade in recent_trades
                if trade.signal_category == category
            ]
        
        return recent_trades
