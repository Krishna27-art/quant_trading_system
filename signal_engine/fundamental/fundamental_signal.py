"""
Fundamental Signal Generator

Analyzes company fundamentals.
Looks at ROE, ROCE, Sales Growth, Debt, Cash Flow.
"""

from typing import Dict, Optional

from signal_engine.base import BaseSignalGenerator, Signal, SignalCategory, SignalDirection
from utils.logger import get_logger

logger = get_logger("signal_engine.fundamental")


class FundamentalSignalGenerator(BaseSignalGenerator):
    """
    Fundamental Signal Generator.
    
    Analyzes:
    - Growth (Sales growth, EPS growth)
    - Quality (ROE, ROCE, profit margins)
    - Valuation (P/E, P/B, EV/EBITDA)
    - Financial Health (Debt ratios, cash flow)
    
    Output: Excellent, Average, or Poor with score (0-100)
    """
    
    def __init__(self):
        super().__init__(name="fundamental", category=SignalCategory.FUNDAMENTAL)
    
    def generate(self, data: Dict[str, Dict]) -> Signal:
        """
        Generate fundamental signal from financial data.
        
        Args:
            data: Dictionary with fundamental data:
                  - roe: Return on Equity (%)
                  - roce: Return on Capital Employed (%)
                  - sales_growth: Sales growth (%)
                  - eps_growth: EPS growth (%)
                  - pe_ratio: Price to Earnings ratio
                  - pb_ratio: Price to Book ratio
                  - debt_to_equity: Debt to Equity ratio
                  - current_ratio: Current ratio
                  - profit_margin: Net profit margin (%)
                  
        Returns:
            Signal object
        """
        # Extract fundamental data
        roe = data.get('roe')
        roce = data.get('roce')
        sales_growth = data.get('sales_growth')
        eps_growth = data.get('eps_growth')
        pe_ratio = data.get('pe_ratio')
        pb_ratio = data.get('pb_ratio')
        debt_to_equity = data.get('debt_to_equity')
        current_ratio = data.get('current_ratio')
        profit_margin = data.get('profit_margin')
        
        # Analyze each metric
        growth_analysis = self._analyze_growth(sales_growth, eps_growth)
        quality_analysis = self._analyze_quality(roe, roce, profit_margin)
        valuation_analysis = self._analyze_valuation(pe_ratio, pb_ratio)
        health_analysis = self._analyze_financial_health(debt_to_equity, current_ratio)
        
        # Count bullish/bearish indicators
        bullish_count = (
            growth_analysis['bullish'] +
            quality_analysis['bullish'] +
            valuation_analysis['bullish'] +
            health_analysis['bullish']
        )
        
        bearish_count = (
            growth_analysis['bearish'] +
            quality_analysis['bearish'] +
            valuation_analysis['bearish'] +
            health_analysis['bearish']
        )
        
        neutral_count = (
            growth_analysis['neutral'] +
            quality_analysis['neutral'] +
            valuation_analysis['neutral'] +
            health_analysis['neutral']
        )
        
        # Calculate score and direction
        score, direction = self._calculate_score(bullish_count, bearish_count, neutral_count)
        
        # Adjust confidence based on data availability
        confidence = self._calculate_confidence(
            roe, roce, sales_growth, eps_growth, pe_ratio, pb_ratio, debt_to_equity, current_ratio, profit_margin
        )
        
        # Build reason
        reason_parts = []
        if growth_analysis['bullish'] > 0:
            reason_parts.append("Strong growth metrics")
        elif growth_analysis['bearish'] > 0:
            reason_parts.append("Weak growth metrics")
        
        if quality_analysis['bullish'] > 0:
            reason_parts.append("High profitability")
        elif quality_analysis['bearish'] > 0:
            reason_parts.append("Low profitability")
        
        if valuation_analysis['bullish'] > 0:
            reason_parts.append("Attractive valuation")
        elif valuation_analysis['bearish'] > 0:
            reason_parts.append("Expensive valuation")
        
        if health_analysis['bullish'] > 0:
            reason_parts.append("Strong financial health")
        elif health_analysis['bearish'] > 0:
            reason_parts.append("Weak financial health")
        
        reason = "; ".join(reason_parts) if reason_parts else "Mixed fundamentals"
        
        # Store raw values
        raw_values = {
            'roe': roe,
            'roce': roce,
            'sales_growth': sales_growth,
            'eps_growth': eps_growth,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'profit_margin': profit_margin,
        }
        
        return Signal(
            name="Fundamental",
            category=self.category,
            score=score,
            direction=direction,
            confidence=confidence,
            reason=reason,
            raw_values=raw_values,
        )
    
    def _analyze_growth(
        self,
        sales_growth: Optional[float],
        eps_growth: Optional[float],
    ) -> Dict[str, int]:
        """
        Analyze growth metrics.
        
        Sales growth > 15%: Strong growth (bullish)
        EPS growth > 15%: Strong earnings growth (bullish)
        """
        if sales_growth is None and eps_growth is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        bullish_count = 0
        bearish_count = 0
        
        if sales_growth is not None:
            if sales_growth > 15:
                bullish_count += 1
            elif sales_growth < 5:
                bearish_count += 1
        
        if eps_growth is not None:
            if eps_growth > 15:
                bullish_count += 1
            elif eps_growth < 5:
                bearish_count += 1
        
        if bullish_count > bearish_count:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif bearish_count > bullish_count:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_quality(
        self,
        roe: Optional[float],
        roce: Optional[float],
        profit_margin: Optional[float],
    ) -> Dict[str, int]:
        """
        Analyze quality metrics.
        
        ROE > 15%: High return on equity (bullish)
        ROCE > 15%: High return on capital (bullish)
        Profit margin > 10%: Good profitability (bullish)
        """
        if roe is None and roce is None and profit_margin is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        bullish_count = 0
        bearish_count = 0
        
        if roe is not None:
            if roe > 15:
                bullish_count += 1
            elif roe < 10:
                bearish_count += 1
        
        if roce is not None:
            if roce > 15:
                bullish_count += 1
            elif roce < 10:
                bearish_count += 1
        
        if profit_margin is not None:
            if profit_margin > 10:
                bullish_count += 1
            elif profit_margin < 5:
                bearish_count += 1
        
        if bullish_count > bearish_count:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif bearish_count > bullish_count:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_valuation(
        self,
        pe_ratio: Optional[float],
        pb_ratio: Optional[float],
    ) -> Dict[str, int]:
        """
        Analyze valuation metrics.
        
        P/E < 20: Attractive valuation (bullish)
        P/E > 30: Expensive (bearish)
        P/B < 3: Reasonable (bullish)
        P/B > 5: Expensive (bearish)
        """
        if pe_ratio is None and pb_ratio is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        bullish_count = 0
        bearish_count = 0
        
        if pe_ratio is not None:
            if pe_ratio < 20:
                bullish_count += 1
            elif pe_ratio > 30:
                bearish_count += 1
        
        if pb_ratio is not None:
            if pb_ratio < 3:
                bullish_count += 1
            elif pb_ratio > 5:
                bearish_count += 1
        
        if bullish_count > bearish_count:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif bearish_count > bullish_count:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _analyze_financial_health(
        self,
        debt_to_equity: Optional[float],
        current_ratio: Optional[float],
    ) -> Dict[str, int]:
        """
        Analyze financial health.
        
        Debt/Equity < 1: Low leverage (bullish)
        Debt/Equity > 2: High leverage (bearish)
        Current ratio > 1.5: Good liquidity (bullish)
        Current ratio < 1: Poor liquidity (bearish)
        """
        if debt_to_equity is None and current_ratio is None:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
        
        bullish_count = 0
        bearish_count = 0
        
        if debt_to_equity is not None:
            if debt_to_equity < 1:
                bullish_count += 1
            elif debt_to_equity > 2:
                bearish_count += 1
        
        if current_ratio is not None:
            if current_ratio > 1.5:
                bullish_count += 1
            elif current_ratio < 1:
                bearish_count += 1
        
        if bullish_count > bearish_count:
            return {'bullish': 1, 'bearish': 0, 'neutral': 0}
        elif bearish_count > bullish_count:
            return {'bullish': 0, 'bearish': 1, 'neutral': 0}
        else:
            return {'bullish': 0, 'bearish': 0, 'neutral': 1}
    
    def _calculate_confidence(
        self,
        roe: Optional[float],
        roce: Optional[float],
        sales_growth: Optional[float],
        eps_growth: Optional[float],
        pe_ratio: Optional[float],
        pb_ratio: Optional[float],
        debt_to_equity: Optional[float],
        current_ratio: Optional[float],
        profit_margin: Optional[float],
    ) -> float:
        """Calculate confidence based on data availability."""
        available_count = sum([
            roe is not None,
            roce is not None,
            sales_growth is not None,
            eps_growth is not None,
            pe_ratio is not None,
            pb_ratio is not None,
            debt_to_equity is not None,
            current_ratio is not None,
            profit_margin is not None,
        ])
        
        total_indicators = 9
        confidence = (available_count / total_indicators) * 100
        
        return confidence
    
    def _create_neutral_signal(self, reason: str) -> Signal:
        """Create a neutral signal when data is insufficient."""
        return Signal(
            name="Fundamental",
            category=self.category,
            score=50.0,
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            reason=reason,
            raw_values={},
        )
