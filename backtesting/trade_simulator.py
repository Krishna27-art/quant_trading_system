"""
Trade Simulator Module

Simulates entry, holding, and exit for predictions.
Implements realistic trade execution with:
- Entry at next open or limit price
- Stop loss based on ATR or fixed percentage
- Target based on resistance or risk/reward ratio
- Exit conditions (target hit, stop hit, time exit)
- Slippage and transaction costs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("trade_simulator")


class ExitReason(Enum):
    """Reason for trade exit."""
    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    TIME_EXIT = "time_exit"
    REVERSAL_SIGNAL = "reversal_signal"
    MANUAL_EXIT = "manual_exit"
    MAX_HOLDING_REACHED = "max_holding_reached"


@dataclass
class Trade:
    """Represents a single trade."""
    
    symbol: str
    entry_date: date
    exit_date: date | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    quantity: int = 100
    stop_loss: float | None = None
    target: float | None = None
    
    # Trade outcomes
    pnl: float | None = None
    pnl_pct: float | None = None
    is_winner: bool | None = None
    
    # Trade metrics
    holding_days: int | None = None
    max_drawdown: float | None = None
    max_profit: float | None = None
    exit_reason: ExitReason | None = None
    
    # Costs
    slippage_pct: float = 0.1  # 0.1% slippage
    transaction_cost_pct: float = 0.05  # 0.05% transaction cost
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "is_winner": self.is_winner,
            "holding_days": self.holding_days,
            "max_drawdown": self.max_drawdown,
            "max_profit": self.max_profit,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "slippage_pct": self.slippage_pct,
            "transaction_cost_pct": self.transaction_cost_pct,
        }


@dataclass
class TradeConfig:
    """Configuration for trade simulation."""
    
    stop_loss_atr_multiplier: float = 2.0
    stop_loss_fixed_pct: float | None = None  # If set, overrides ATR
    target_rr_ratio: float = 2.0  # Risk/Reward ratio
    target_fixed_pct: float | None = None  # If set, overrides RR
    max_holding_days: int = 30
    slippage_pct: float = 0.1
    transaction_cost_pct: float = 0.05
    position_size_pct: float = 0.02  # 2% of capital per trade


class TradeSimulator:
    """
    Simulates trades based on predictions and price data.
    
    Realistically models trade execution with costs and slippage.
    """
    
    def __init__(self, config: TradeConfig | None = None):
        """
        Initialize the trade simulator.
        
        Args:
            config: Trade configuration (uses defaults if None)
        """
        self.config = config or TradeConfig()
        self.logger = logger
    
    def simulate_trade(
        self,
        symbol: str,
        prediction_date: date,
        prediction: str,  # "BUY" or "SELL"
        price_data: pd.DataFrame,
        atr: float | None = None,
        entry_price: float | None = None,
    ) -> Trade:
        """
        Simulate a single trade.
        
        Args:
            symbol: Stock symbol
            prediction_date: Date of prediction
            prediction: "BUY" or "SELL"
            price_data: DataFrame with OHLCV data (must include prediction_date onwards)
            atr: ATR value for stop loss calculation
            entry_price: Specific entry price (if None, uses next open)
            
        Returns:
            Trade object with simulation results
        """
        trade = Trade(
            symbol=symbol,
            entry_date=prediction_date,
            quantity=100,  # Default quantity
            slippage_pct=self.config.slippage_pct,
            transaction_cost_pct=self.config.transaction_cost_pct,
        )
        
        try:
            # Find entry
            entry_row = self._find_entry(price_data, prediction_date, entry_price)
            if entry_row is None:
                self.logger.warning(f"No entry found for {symbol} on {prediction_date}")
                return trade
            
            trade.entry_price = entry_row['close']  # Use close as entry
            trade.entry_date = entry_row['timestamp'].date()
            
            # Calculate stop loss and target
            trade.stop_loss = self._calculate_stop_loss(
                trade.entry_price, atr, prediction
            )
            trade.target = self._calculate_target(
                trade.entry_price, trade.stop_loss, prediction
            )
            
            # Simulate holding period
            exit_row, exit_reason = self._simulate_holding(
                trade, price_data, prediction
            )
            
            if exit_row is not None:
                trade.exit_date = exit_row['timestamp'].date()
                trade.exit_price = exit_row['close']
                trade.exit_reason = exit_reason
            
            # Calculate P&L
            self._calculate_pnl(trade)
            
            # Calculate trade metrics
            self._calculate_trade_metrics(trade, price_data)
            
        except Exception as e:
            self.logger.error(f"Error simulating trade for {symbol}: {e}")
        
        return trade
    
    def _find_entry(
        self,
        price_data: pd.DataFrame,
        prediction_date: date,
        entry_price: float | None = None,
    ) -> pd.Series | None:
        """Find entry row in price data."""
        if entry_price is not None:
            # Find first row where price is close to entry_price
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            mask = price_data['timestamp'].dt.date >= prediction_date
            available_data = price_data[mask]
            
            if len(available_data) == 0:
                return None
            
            # Use first available day
            return available_data.iloc[0]
        else:
            # Use next open after prediction date
            price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
            mask = price_data['timestamp'].dt.date > prediction_date
            available_data = price_data[mask]
            
            if len(available_data) == 0:
                return None
            
            return available_data.iloc[0]
    
    def _calculate_stop_loss(
        self,
        entry_price: float,
        atr: float | None,
        prediction: str,
    ) -> float:
        """Calculate stop loss price."""
        if self.config.stop_loss_fixed_pct:
            sl_pct = self.config.stop_loss_fixed_pct
        elif atr is not None:
            sl_pct = (self.config.stop_loss_atr_multiplier * atr) / entry_price
        else:
            sl_pct = 0.05  # Default 5%
        
        if prediction == "BUY":
            return entry_price * (1 - sl_pct)
        else:  # SELL
            return entry_price * (1 + sl_pct)
    
    def _calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        prediction: str,
    ) -> float:
        """Calculate target price."""
        if self.config.target_fixed_pct:
            target_pct = self.config.target_fixed_pct
        else:
            # Use risk/reward ratio
            risk = abs(entry_price - stop_loss)
            target_pct = (risk * self.config.target_rr_ratio) / entry_price
        
        if prediction == "BUY":
            return entry_price * (1 + target_pct)
        else:  # SELL
            return entry_price * (1 - target_pct)
    
    def _simulate_holding(
        self,
        trade: Trade,
        price_data: pd.DataFrame,
        prediction: str,
    ) -> tuple[pd.Series | None, ExitReason]:
        """Simulate holding period and find exit."""
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        # Filter data from entry date onwards
        mask = price_data['timestamp'].dt.date >= trade.entry_date
        holding_data = price_data[mask].copy()
        
        if len(holding_data) == 0:
            return None, ExitReason.MANUAL_EXIT
        
        # Track max profit and drawdown during holding
        max_profit = 0.0
        max_drawdown = 0.0
        
        for idx, row in holding_data.iterrows():
            if row['timestamp'].date() == trade.entry_date:
                continue  # Skip entry day
            
            # Check holding period limit
            holding_days = (row['timestamp'].date() - trade.entry_date).days
            if holding_days > self.config.max_holding_days:
                return row, ExitReason.MAX_HOLDING_REACHED
            
            # Calculate current P&L
            if prediction == "BUY":
                current_pnl_pct = (row['close'] - trade.entry_price) / trade.entry_price
            else:
                current_pnl_pct = (trade.entry_price - row['close']) / trade.entry_price
            
            # Update max profit/drawdown
            max_profit = max(max_profit, current_pnl_pct)
            max_drawdown = min(max_drawdown, current_pnl_pct)
            
            # Check stop loss
            if prediction == "BUY":
                if row['low'] <= trade.stop_loss:
                    return row, ExitReason.STOP_HIT
            else:
                if row['high'] >= trade.stop_loss:
                    return row, ExitReason.STOP_HIT
            
            # Check target
            if prediction == "BUY":
                if row['high'] >= trade.target:
                    return row, ExitReason.TARGET_HIT
            else:
                if row['low'] <= trade.target:
                    return row, ExitReason.TARGET_HIT
        
        # If no exit condition met, use last available day
        return holding_data.iloc[-1], ExitReason.TIME_EXIT
    
    def _calculate_pnl(self, trade: Trade) -> None:
        """Calculate P&L with costs."""
        if trade.entry_price is None or trade.exit_price is None:
            return
        
        # Apply slippage
        entry_with_slippage = trade.entry_price * (1 + trade.slippage_pct / 100)
        exit_with_slippage = trade.exit_price * (1 - trade.slippage_pct / 100)
        
        # Calculate gross P&L
        gross_pnl = (exit_with_slippage - entry_with_slippage) * trade.quantity
        
        # Apply transaction costs
        entry_cost = entry_with_slippage * trade.quantity * trade.transaction_cost_pct / 100
        exit_cost = exit_with_slippage * trade.quantity * trade.transaction_cost_pct / 100
        total_cost = entry_cost + exit_cost
        
        # Net P&L
        trade.pnl = gross_pnl - total_cost
        
        # P&L percentage
        trade.pnl_pct = (trade.pnl / (entry_with_slippage * trade.quantity)) * 100
        
        # Winner/Loser
        trade.is_winner = trade.pnl > 0
    
    def _calculate_trade_metrics(
        self,
        trade: Trade,
        price_data: pd.DataFrame,
    ) -> None:
        """Calculate additional trade metrics."""
        if trade.entry_date is None or trade.exit_date is None:
            return
        
        # Holding days
        trade.holding_days = (trade.exit_date - trade.entry_date).days
        
        # Calculate max drawdown and max profit during holding
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        mask = (
            (price_data['timestamp'].dt.date >= trade.entry_date) &
            (price_data['timestamp'].dt.date <= trade.exit_date)
        )
        holding_data = price_data[mask]
        
        if len(holding_data) > 0:
            prices = holding_data['close'].values
            entry_idx = 0
            entry_price = prices[entry_idx]
            
            max_profit = 0.0
            max_drawdown = 0.0
            
            for price in prices[1:]:
                pnl_pct = (price - entry_price) / entry_price * 100
                max_profit = max(max_profit, pnl_pct)
                max_drawdown = min(max_drawdown, pnl_pct)
            
            trade.max_profit = max_profit
            trade.max_drawdown = max_drawdown
    
    def simulate_trades(
        self,
        predictions: list[dict[str, Any]],
        price_data: pd.DataFrame,
        atr_data: dict[str, float] | None = None,
    ) -> list[Trade]:
        """
        Simulate multiple trades.
        
        Args:
            predictions: List of prediction dicts with symbol, date, prediction
            price_data: DataFrame with all price data
            atr_data: Dict mapping symbol to ATR values
            
        Returns:
            List of Trade objects
        """
        trades = []
        
        for pred in predictions:
            symbol = pred['symbol']
            pred_date = pred['date']
            prediction_type = pred['prediction']
            
            # Get symbol-specific price data
            symbol_data = price_data[price_data['symbol'] == symbol].copy()
            
            if len(symbol_data) == 0:
                self.logger.warning(f"No price data for {symbol}")
                continue
            
            atr = atr_data.get(symbol) if atr_data else None
            
            trade = self.simulate_trade(
                symbol=symbol,
                prediction_date=pred_date,
                prediction=prediction_type,
                price_data=symbol_data,
                atr=atr,
            )
            
            trades.append(trade)
        
        self.logger.info(f"Simulated {len(trades)} trades")
        return trades


def simulate_backtest(
    predictions: list[dict[str, Any]],
    price_data: pd.DataFrame,
    config: TradeConfig | None = None,
) -> dict[str, Any]:
    """
    Convenience function to run a complete backtest simulation.
    
    Args:
        predictions: List of prediction dicts
        price_data: DataFrame with price data
        config: Trade configuration
        
    Returns:
        Backtest results with trades and summary
    """
    simulator = TradeSimulator(config)
    trades = simulator.simulate_trades(predictions, price_data)
    
    # Calculate summary statistics
    if trades:
        completed_trades = [t for t in trades if t.exit_price is not None]
        
        if completed_trades:
            win_rate = sum(1 for t in completed_trades if t.is_winner) / len(completed_trades) * 100
            avg_pnl = sum(t.pnl for t in completed_trades if t.pnl is not None) / len(completed_trades)
            avg_win = sum(t.pnl for t in completed_trades if t.pnl and t.pnl > 0) / max(1, sum(1 for t in completed_trades if t.pnl and t.pnl > 0))
            avg_loss = sum(t.pnl for t in completed_trades if t.pnl and t.pnl < 0) / max(1, sum(1 for t in completed_trades if t.pnl and t.pnl < 0))
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_pnl = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
    else:
        win_rate = 0
        avg_pnl = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    return {
        "trades": [t.to_dict() for t in trades],
        "summary": {
            "total_trades": len(trades),
            "completed_trades": len(completed_trades) if trades else 0,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        },
    }
