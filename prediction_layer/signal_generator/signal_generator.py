"""
Signal Generator

Converts model predictions into actionable trading signals.
Generates entry price, stop loss, targets, and confidence metrics.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from utils.time_utils import now_ist


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Complete trading signal with all required information."""
    symbol: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    expected_return_pct: float
    expected_holding_days: int
    probability: float
    confidence: float
    worst_case_pct: float
    best_case_pct: float
    win_probability: float
    
    # Explainability
    reasons: List[str]
    feature_importance: Dict[str, float]
    
    # Quality scores
    trend_score: float
    momentum_score: float
    liquidity_score: float
    volume_score: float
    options_score: float
    news_score: float
    macro_score: float
    sector_score: float
    institutional_score: float
    overall_score: float
    
    # Versioning
    model_version: str
    feature_version: str
    dataset_version: str
    
    timestamp: datetime


class SignalGenerator:
    """
    Generates actionable trading signals from model predictions.
    
    Calculates:
    - Entry price (current or limit)
    - Stop loss (ATR-based or percentage-based)
    - Targets (risk-reward based)
    - Expected return and holding period
    - Probability and confidence metrics
    - Quality scores across multiple dimensions
    """
    
    def __init__(self, atr_multiplier: float = 2.0, risk_reward_ratio: float = 2.0):
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio
    
    def generate_signal(
        self,
        symbol: str,
        current_price: float,
        prediction: str,
        probability: float,
        feature_importance: Dict[str, float],
        atr: Optional[float] = None,
        volatility_pct: Optional[float] = None,
        model_version: str = "v1.0.0",
        feature_version: str = "v1.0.0",
        dataset_version: str = "v1.0.0"
    ) -> TradingSignal:
        """
        Generate a complete trading signal.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            prediction: Model prediction (BUY/SELL/HOLD)
            probability: Prediction probability
            feature_importance: Feature importance scores
            atr: Average True Range for stop loss calculation
            volatility_pct: Volatility percentage
            model_version: Model version
            feature_version: Feature version
            dataset_version: Dataset version
            
        Returns:
            Complete trading signal
        """
        direction = SignalDirection(prediction.upper())
        
        if direction == SignalDirection.HOLD:
            return self._generate_hold_signal(
                symbol, current_price, probability, feature_importance,
                model_version, feature_version, dataset_version
            )
        
        # Calculate stop loss
        if atr:
            stop_distance = atr * self.atr_multiplier
        elif volatility_pct:
            stop_distance = current_price * (volatility_pct / 100) * self.atr_multiplier
        else:
            stop_distance = current_price * 0.02  # Default 2%
        
        if direction == SignalDirection.BUY:
            stop_loss = current_price - stop_distance
            target_1 = current_price + (stop_distance * self.risk_reward_ratio)
            target_2 = current_price + (stop_distance * self.risk_reward_ratio * 1.5)
        else:  # SELL
            stop_loss = current_price + stop_distance
            target_1 = current_price - (stop_distance * self.risk_reward_ratio)
            target_2 = current_price - (stop_distance * self.risk_reward_ratio * 1.5)
        
        # Calculate expected metrics
        expected_return_pct = abs((target_1 - current_price) / current_price * 100)
        worst_case_pct = abs((stop_loss - current_price) / current_price * 100)
        best_case_pct = abs((target_2 - current_price) / current_price * 100)
        
        # Estimate holding period based on volatility
        if volatility_pct:
            expected_holding_days = int(10 / volatility_pct * 100)  # Lower vol = longer hold
        else:
            expected_holding_days = 5
        
        # Generate reasons from feature importance
        reasons = self._generate_reasons(feature_importance, direction)
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(
            feature_importance, probability, volatility_pct or 2.0
        )
        
        return TradingSignal(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            expected_return_pct=expected_return_pct,
            expected_holding_days=expected_holding_days,
            probability=probability,
            confidence=min(probability + 0.05, 1.0),  # Slightly higher than raw probability
            worst_case_pct=worst_case_pct,
            best_case_pct=best_case_pct,
            win_probability=probability,
            reasons=reasons,
            feature_importance=feature_importance,
            **quality_scores,
            model_version=model_version,
            feature_version=feature_version,
            dataset_version=dataset_version,
            timestamp=now_ist()
        )
    
    def _generate_hold_signal(
        self,
        symbol: str,
        current_price: float,
        probability: float,
        feature_importance: Dict[str, float],
        model_version: str,
        feature_version: str,
        dataset_version: str
    ) -> TradingSignal:
        """Generate a HOLD signal."""
        return TradingSignal(
            symbol=symbol,
            direction=SignalDirection.HOLD,
            entry_price=current_price,
            stop_loss=current_price,
            target_1=current_price,
            target_2=None,
            expected_return_pct=0.0,
            expected_holding_days=0,
            probability=probability,
            confidence=probability,
            worst_case_pct=0.0,
            best_case_pct=0.0,
            win_probability=probability,
            reasons=["No clear directional signal"],
            feature_importance=feature_importance,
            trend_score=50.0,
            momentum_score=50.0,
            liquidity_score=50.0,
            volume_score=50.0,
            options_score=50.0,
            news_score=50.0,
            macro_score=50.0,
            sector_score=50.0,
            institutional_score=50.0,
            overall_score=50.0,
            model_version=model_version,
            feature_version=feature_version,
            dataset_version=dataset_version,
            timestamp=now_ist()
        )
    
    def _generate_reasons(
        self,
        feature_importance: Dict[str, float],
        direction: SignalDirection
    ) -> List[str]:
        """Generate human-readable reasons from feature importance."""
        reasons = []
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        for feature, importance in sorted_features:
            if abs(importance) < 0.01:
                continue
            
            feature_lower = feature.lower()
            
            if 'rsi' in feature_lower:
                if importance > 0 and direction == SignalDirection.BUY:
                    reasons.append("RSI oversold - potential reversal")
                elif importance < 0 and direction == SignalDirection.SELL:
                    reasons.append("RSI overbought - potential reversal")
            elif 'macd' in feature_lower:
                if importance > 0:
                    reasons.append("MACD bullish crossover")
                else:
                    reasons.append("MACD bearish crossover")
            elif 'ema' in feature_lower:
                if importance > 0:
                    reasons.append("Price above key EMA - bullish")
                else:
                    reasons.append("Price below key EMA - bearish")
            elif 'volume' in feature_lower:
                if importance > 0:
                    reasons.append("Strong volume confirmation")
            elif 'oi' in feature_lower or 'option' in feature_lower:
                if importance > 0:
                    reasons.append("Positive options flow")
                else:
                    reasons.append("Negative options flow")
            elif 'news' in feature_lower or 'sentiment' in feature_lower:
                if importance > 0:
                    reasons.append("Positive news sentiment")
                else:
                    reasons.append("Negative news sentiment")
            elif 'sector' in feature_lower:
                if importance > 0:
                    reasons.append("Strong sector performance")
                else:
                    reasons.append("Weak sector performance")
        
        if not reasons:
            reasons.append("Model prediction based on multiple factors")
        
        return reasons
    
    def _calculate_quality_scores(
        self,
        feature_importance: Dict[str, float],
        probability: float,
        volatility_pct: float
    ) -> Dict[str, float]:
        """Calculate quality scores across multiple dimensions."""
        scores = {}
        
        # Trend score (based on trend-related features)
        trend_features = ['ema', 'sma', 'macd', 'adx']
        trend_imp = sum(abs(v) for k, v in feature_importance.items() if any(f in k.lower() for f in trend_features))
        scores['trend_score'] = min(trend_imp * 100, 100)
        
        # Momentum score (based on momentum features)
        momentum_features = ['rsi', 'roc', 'momentum', 'stochastic']
        momentum_imp = sum(abs(v) for k, v in feature_importance.items() if any(f in k.lower() for f in momentum_features))
        scores['momentum_score'] = min(momentum_imp * 100, 100)
        
        # Liquidity score (inverse of volatility)
        scores['liquidity_score'] = max(0, 100 - volatility_pct * 10)
        
        # Volume score
        volume_imp = sum(abs(v) for k, v in feature_importance.items() if 'volume' in k.lower())
        scores['volume_score'] = min(volume_imp * 100, 100)
        
        # Options score
        options_imp = sum(abs(v) for k, v in feature_importance.items() if any(f in k.lower() for f in ['oi', 'option', 'pcr', 'iv']))
        scores['options_score'] = min(options_imp * 100, 100)
        
        # News score
        news_imp = sum(abs(v) for k, v in feature_importance.items() if any(f in k.lower() for f in ['news', 'sentiment']))
        scores['news_score'] = min(news_imp * 100, 100)
        
        # Macro score
        macro_imp = sum(abs(v) for k, v in feature_importance.items() if any(f in k.lower() for f in ['macro', 'vix', 'nifty', 'usd']))
        scores['macro_score'] = min(macro_imp * 100, 100)
        
        # Sector score
        sector_imp = sum(abs(v) for k, v in feature_importance.items() if 'sector' in k.lower())
        scores['sector_score'] = min(sector_imp * 100, 100)
        
        # Institutional score
        inst_imp = sum(abs(v) for k, v in feature_importance.items() if any(f in k.lower() for f in ['fii', 'dii', 'institutional']))
        scores['institutional_score'] = min(inst_imp * 100, 100)
        
        # Overall score (weighted average)
        weights = {
            'trend_score': 0.20,
            'momentum_score': 0.15,
            'liquidity_score': 0.10,
            'volume_score': 0.10,
            'options_score': 0.15,
            'news_score': 0.10,
            'macro_score': 0.10,
            'sector_score': 0.05,
            'institutional_score': 0.05
        }
        
        overall = sum(scores[k] * weights[k] for k in weights)
        scores['overall_score'] = min(overall + probability * 10, 100)  # Boost by probability
        
        return scores
