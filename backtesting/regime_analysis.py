"""
Regime Analysis Module

Analyzes performance breakdown by market regime.
Helps identify where the strategy has edge and where it fails.

Regimes analyzed:
- Strong Bull
- Bull
- Sideways
- Bear
- High Volatility
- Event Day
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from typing import Any

import pandas as pd

from regime.regime_history import RegimeHistoryManager
from utils.logger import get_logger

logger = get_logger("regime_analysis")


class RegimeAnalyzer:
    """
    Analyzes performance by market regime.
    
    Identifies which regimes the strategy performs well in
    and which regimes should be filtered out.
    """
    
    def __init__(self):
        """Initialize the regime analyzer."""
        self.logger = logger
        self.regime_manager = RegimeHistoryManager()
    
    def analyze_performance_by_regime(
        self,
        trades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Break down performance by market regime.
        
        Args:
            trades: List of trade dicts with entry_date and pnl
            
        Returns:
            Dict with performance metrics by regime
        """
        self.logger.info("Analyzing performance by regime...")
        
        # Get regime for each trade
        trades_with_regime = []
        for trade in trades:
            entry_date = trade.get('entry_date')
            if isinstance(entry_date, str):
                entry_date = date.fromisoformat(entry_date)
            
            regime = self.regime_manager.get_regime_for_date(entry_date)
            if regime:
                trade_with_regime = trade.copy()
                trade_with_regime['regime'] = regime.regime.value
                trades_with_regime.append(trade_with_regime)
        
        if not trades_with_regime:
            self.logger.warning("No trades with regime data found")
            return {}
        
        # Group by regime
        regime_groups = defaultdict(list)
        for trade in trades_with_regime:
            regime_groups[trade['regime']].append(trade)
        
        # Calculate metrics for each regime
        regime_metrics = {}
        for regime, regime_trades in regime_groups.items():
            regime_metrics[regime] = self._calculate_regime_metrics(regime_trades)
        
        # Add overall comparison
        regime_metrics['comparison'] = self._compare_regimes(regime_metrics)
        
        self.logger.info(f"Regime analysis complete for {len(regime_groups)} regimes")
        return regime_metrics
    
    def _calculate_regime_metrics(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate performance metrics for a specific regime."""
        completed_trades = [t for t in trades if t.get('pnl') is not None]
        
        if not completed_trades:
            return {
                "count": len(trades),
                "completed": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "total_pnl": 0,
            }
        
        pnls = [t['pnl'] for t in completed_trades]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        total_pnl = sum(pnls)
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')
        
        return {
            "count": len(trades),
            "completed": len(completed_trades),
            "win_rate": round(win_rate, 2),
            "avg_pnl": round(avg_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "best_trade": round(max(pnls), 2) if pnls else 0,
            "worst_trade": round(min(pnls), 2) if pnls else 0,
        }
    
    def _compare_regimes(self, regime_metrics: dict[str, dict]) -> dict[str, Any]:
        """Compare performance across regimes."""
        comparison = {
            "best_regime_win_rate": None,
            "best_regime_total_pnl": None,
            "worst_regime_win_rate": None,
            "worst_regime_total_pnl": None,
            "regime_rankings": {},
        }
        
        # Exclude 'comparison' key
        metrics_only = {k: v for k, v in regime_metrics.items() if k != 'comparison'}
        
        if not metrics_only:
            return comparison
        
        # Find best/worst by win rate
        win_rates = {r: m['win_rate'] for r, m in metrics_only.items()}
        comparison['best_regime_win_rate'] = max(win_rates, key=win_rates.get)
        comparison['worst_regime_win_rate'] = min(win_rates, key=win_rates.get)
        
        # Find best/worst by total P&L
        total_pnls = {r: m['total_pnl'] for r, m in metrics_only.items()}
        comparison['best_regime_total_pnl'] = max(total_pnls, key=total_pnls.get)
        comparison['worst_regime_total_pnl'] = min(total_pnls, key=total_pnls.get)
        
        # Rank regimes by total P&L
        comparison['regime_rankings'] = dict(
            sorted(total_pnls.items(), key=lambda x: x[1], reverse=True)
        )
        
        return comparison
    
    def analyze_sector_performance_by_regime(
        self,
        trades: list[dict[str, Any]],
        sector_mapping: dict[str, str],
    ) -> dict[str, Any]:
        """
        Analyze performance by sector within each regime.
        
        Args:
            trades: List of trade dicts with symbol and entry_date
            sector_mapping: Dict mapping symbol to sector
            
        Returns:
            Dict with sector performance by regime
        """
        self.logger.info("Analyzing sector performance by regime...")
        
        # Add sector to trades
        trades_with_sector = []
        for trade in trades:
            symbol = trade.get('symbol')
            sector = sector_mapping.get(symbol, 'Unknown')
            trade_with_sector = trade.copy()
            trade_with_sector['sector'] = sector
            trades_with_sector.append(trade_with_sector)
        
        # Group by regime and sector
        regime_sector_groups = defaultdict(lambda: defaultdict(list))
        
        for trade in trades_with_sector:
            entry_date = trade.get('entry_date')
            if isinstance(entry_date, str):
                entry_date = date.fromisoformat(entry_date)
            
            regime = self.regime_manager.get_regime_for_date(entry_date)
            if regime:
                regime_sector_groups[regime.regime.value][trade['sector']].append(trade)
        
        # Calculate metrics for each regime-sector combination
        results = {}
        for regime, sectors in regime_sector_groups.items():
            results[regime] = {}
            for sector, sector_trades in sectors.items():
                results[regime][sector] = self._calculate_regime_metrics(sector_trades)
        
        return results
    
    def get_regime_recommendations(
        self,
        regime_metrics: dict[str, dict],
    ) -> dict[str, Any]:
        """
        Generate recommendations based on regime analysis.
        
        Args:
            regime_metrics: Output from analyze_performance_by_regime
            
        Returns:
            Dict with recommendations
        """
        recommendations = {
            "filter_regimes": [],
            "boost_regimes": [],
            "neutral_regimes": [],
        }
        
        if 'comparison' not in regime_metrics:
            return recommendations
        
        comparison = regime_metrics['comparison']
        metrics_only = {k: v for k, v in regime_metrics.items() if k != 'comparison'}
        
        # Filter regimes with poor performance
        for regime, metrics in metrics_only.items():
            if metrics['win_rate'] < 40 or metrics['total_pnl'] < 0:
                recommendations['filter_regimes'].append(regime)
            elif metrics['win_rate'] > 60 and metrics['total_pnl'] > 0:
                recommendations['boost_regimes'].append(regime)
            else:
                recommendations['neutral_regimes'].append(regime)
        
        return recommendations


def analyze_by_regime(
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convenience function to analyze performance by regime.
    
    Args:
        trades: List of trade dicts
        
    Returns:
        Regime analysis results
    """
    analyzer = RegimeAnalyzer()
    return analyzer.analyze_performance_by_regime(trades)
