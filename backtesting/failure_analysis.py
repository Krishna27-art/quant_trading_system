"""
Failure Analysis Module

Analyzes failed predictions and categorizes reasons for failure.
Helps identify systematic issues and improve the model.

Failure categories:
- False Breakout
- Bad Earnings
- Wrong Regime
- Poor Liquidity
- Gap Down
- Unexpected News
- Sector Rotation
- Technical Reversal
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from enum import Enum
from typing import Any

import pandas as pd

from utils.logger import get_logger

logger = get_logger("failure_analysis")


class FailureReason(Enum):
    """Categorized reasons for prediction failures."""
    FALSE_BREAKOUT = "false_breakout"
    BAD_EARNINGS = "bad_earnings"
    WRONG_REGIME = "wrong_regime"
    POOR_LIQUIDITY = "poor_liquidity"
    GAP_DOWN = "gap_down"
    UNEXPECTED_NEWS = "unexpected_news"
    SECTOR_ROTATION = "sector_rotation"
    TECHNICAL_REVERSAL = "technical_reversal"
    MARKET_CRASH = "market_crash"
    UNKNOWN = "unknown"


class FailureAnalyzer:
    """
    Analyzes failed predictions and categorizes failure reasons.
    
    Helps identify systematic issues and improve the model.
    """
    
    def __init__(self):
        """Initialize the failure analyzer."""
        self.logger = logger
    
    def analyze_failures(
        self,
        trades: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze failed predictions and categorize reasons.
        
        Args:
            trades: List of trade dicts with pnl and exit_reason
            predictions: List of prediction dicts
            
        Returns:
            Dict with failure analysis
        """
        self.logger.info("Analyzing prediction failures...")
        
        # Identify failed trades
        failed_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        if not failed_trades:
            return {
                "total_failures": 0,
                "failure_rate": 0,
                "failure_reasons": {},
                "recommendations": [],
            }
        
        # Categorize failures
        categorized_failures = defaultdict(list)
        
        for trade in failed_trades:
            reason = self._categorize_failure(trade)
            categorized_failures[reason].append(trade)
        
        # Calculate statistics
        failure_stats = {}
        for reason, reason_trades in categorized_failures.items():
            total_loss = sum(t['pnl'] for t in reason_trades)
            avg_loss = total_loss / len(reason_trades)
            
            failure_stats[reason] = {
                "count": len(reason_trades),
                "percentage": len(reason_trades) / len(failed_trades) * 100,
                "total_loss": round(total_loss, 2),
                "avg_loss": round(avg_loss, 2),
            }
        
        # Generate recommendations
        recommendations = self._generate_failure_recommendations(categorized_failures)
        
        self.logger.info(f"Analyzed {len(failed_trades)} failed trades")
        return {
            "total_failures": len(failed_trades),
            "failure_rate": len(failed_trades) / len(trades) * 100 if trades else 0,
            "failure_reasons": failure_stats,
            "recommendations": recommendations,
        }
    
    def _categorize_failure(self, trade: dict[str, Any]) -> FailureReason:
        """
        Categorize the reason for a failed trade.
        
        Args:
            trade: Trade dict
            
        Returns:
            FailureReason enum
        """
        exit_reason = trade.get('exit_reason')
        pnl_pct = trade.get('pnl_pct', 0)
        
        # Check exit reason first
        if exit_reason == 'stop_hit':
            if pnl_pct < -5:
                return FailureReason.FALSE_BREAKOUT
            elif pnl_pct < -3:
                return FailureReason.TECHNICAL_REVERSAL
            else:
                return FailureReason.WRONG_REGIME
        
        # Check for gap down
        if exit_reason == 'gap_down':
            return FailureReason.GAP_DOWN
        
        # Check for large loss (market crash)
        if pnl_pct < -10:
            return FailureReason.MARKET_CRASH
        
        # Default to unknown
        return FailureReason.UNKNOWN
    
    def _generate_failure_recommendations(
        self,
        categorized_failures: dict[FailureReason, list],
    ) -> list[str]:
        """Generate recommendations based on failure patterns."""
        recommendations = []
        
        # Check for high false breakout rate
        false_breakouts = len(categorized_failures.get(FailureReason.FALSE_BREAKOUT, []))
        total_failures = sum(len(v) for v in categorized_failures.values())
        
        if false_breakouts / total_failures > 0.3:
            recommendations.append(
                "High false breakout rate detected. Consider adding "
                "volume confirmation or waiting for pullback entries."
            )
        
        # Check for regime-related failures
        regime_failures = len(categorized_failures.get(FailureReason.WRONG_REGIME, []))
        if regime_failures / total_failures > 0.2:
            recommendations.append(
                "Many failures due to wrong regime. Strengthen regime "
                "filtering or add regime-specific models."
            )
        
        # Check for market crash sensitivity
        crash_failures = len(categorized_failures.get(FailureReason.MARKET_CRASH, []))
        if crash_failures > 0:
            recommendations.append(
                "Strategy sensitive to market crashes. Consider adding "
                "market-wide stop conditions or volatility filters."
            )
        
        # Check for technical reversal issues
        reversal_failures = len(categorized_failures.get(FailureReason.TECHNICAL_REVERSAL, []))
        if reversal_failures / total_failures > 0.25:
            recommendations.append(
                "High technical reversal rate. Consider adding "
                "momentum confirmation or tighter stop losses."
            )
        
        if not recommendations:
            recommendations.append("No systematic failure patterns detected.")
        
        return recommendations
    
    def analyze_worst_performing_symbols(
        self,
        trades: list[dict[str, Any]],
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Identify worst performing symbols.
        
        Args:
            trades: List of trade dicts
            top_n: Number of top worst symbols to return
            
        Returns:
            Dict with worst performing symbols
        """
        self.logger.info("Analyzing worst performing symbols...")
        
        # Group by symbol
        symbol_performance = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0})
        
        for trade in trades:
            symbol = trade.get('symbol')
            pnl = trade.get('pnl', 0)
            symbol_performance[symbol]['pnl'] += pnl
            symbol_performance[symbol]['trades'] += 1
            if pnl > 0:
                symbol_performance[symbol]['wins'] += 1
        
        # Calculate metrics
        symbol_stats = []
        for symbol, stats in symbol_performance.items():
            if stats['trades'] > 0:
                win_rate = stats['wins'] / stats['trades'] * 100
                avg_pnl = stats['pnl'] / stats['trades']
                symbol_stats.append({
                    'symbol': symbol,
                    'total_pnl': stats['pnl'],
                    'avg_pnl': avg_pnl,
                    'trades': stats['trades'],
                    'win_rate': win_rate,
                })
        
        # Sort by total P&L (worst first)
        symbol_stats.sort(key=lambda x: x['total_pnl'])
        
        return {
            'worst_symbols': symbol_stats[:top_n],
            'total_symbols_analyzed': len(symbol_stats),
        }
    
    def analyze_failure_timing(
        self,
        trades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Analyze when failures occur (holding period analysis).
        
        Args:
            trades: List of trade dicts
            
        Returns:
            Dict with timing analysis
        """
        self.logger.info("Analyzing failure timing...")
        
        failed_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        if not failed_trades:
            return {}
        
        holding_periods = [t.get('holding_days', 0) for t in failed_trades]
        
        # Bucket by holding period
        buckets = {
            '1-3 days': 0,
            '4-7 days': 0,
            '8-14 days': 0,
            '15-30 days': 0,
            '30+ days': 0,
        }
        
        for days in holding_periods:
            if days <= 3:
                buckets['1-3 days'] += 1
            elif days <= 7:
                buckets['4-7 days'] += 1
            elif days <= 14:
                buckets['8-14 days'] += 1
            elif days <= 30:
                buckets['15-30 days'] += 1
            else:
                buckets['30+ days'] += 1
        
        # Calculate percentages
        total = len(holding_periods)
        bucket_pct = {k: v / total * 100 for k, v in buckets.items()}
        
        return {
            'holding_period_distribution': buckets,
            'holding_period_percentages': bucket_pct,
            'avg_holding_period': sum(holding_periods) / len(holding_periods) if holding_periods else 0,
        }


def analyze_failures(
    trades: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convenience function to analyze prediction failures.
    
    Args:
        trades: List of trade dicts
        predictions: List of prediction dicts
        
    Returns:
        Failure analysis results
    """
    analyzer = FailureAnalyzer()
    return analyzer.analyze_failures(trades, predictions)
