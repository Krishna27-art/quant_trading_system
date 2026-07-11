"""
Alpha Explainer

Generates human-readable explanations for alpha scores.

STEP 12: Alpha Explanation

This module:
1. Explains why a stock received its alpha score
2. Breaks down category contributions
3. Highlights key factors
4. Provides actionable insights
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from alpha_engine.alpha_builder import AlphaCategory, AlphaGrade, AlphaResult
from utils.logger import get_logger

logger = get_logger("alpha_engine.explainer")


@dataclass
class CategoryExplanation:
    """
    Explanation for a single category.
    """
    category: str
    score: float
    weight: float
    contribution: float
    rating: str  # ★★★★★ to ★☆☆☆☆
    key_factors: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "score": round(self.score, 2),
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 2),
            "rating": self.rating,
            "key_factors": self.key_factors,
        }


@dataclass
class AlphaExplanation:
    """
    Complete explanation for an alpha score.
    """
    symbol: str
    alpha_score: float
    grade: AlphaGrade
    summary: str
    category_explanations: Dict[str, CategoryExplanation]
    strengths: List[str]
    weaknesses: List[str]
    risk_factors: List[str]
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "alpha_score": round(self.alpha_score, 2),
            "grade": self.grade.value,
            "summary": self.summary,
            "category_explanations": {
                k: v.to_dict() for k, v in self.category_explanations.items()
            },
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "risk_factors": self.risk_factors,
            "recommendation": self.recommendation,
        }


class AlphaExplainer:
    """
    Generates human-readable explanations for alpha scores.
    
    This makes the alpha scoring system explainable and actionable.
    """
    
    def __init__(self):
        """Initialize Alpha Explainer."""
        self._logger = logger
        
        # Rating thresholds
        self.rating_thresholds = {
            5: 90,
            4: 80,
            3: 70,
            2: 60,
            1: 0,
        }
    
    def explain_alpha(
        self,
        alpha_result: AlphaResult,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> AlphaExplanation:
        """
        Generate explanation for an alpha score.
        
        Args:
            alpha_result: AlphaResult to explain
            market_data: Optional market data for context
            
        Returns:
            AlphaExplanation with detailed breakdown
        """
        self._logger.info(f"Generating explanation for {alpha_result.symbol}")
        
        # Step 1: Generate category explanations
        category_explanations = self._explain_categories(alpha_result)
        
        # Step 2: Generate summary
        summary = self._generate_summary(alpha_result, category_explanations)
        
        # Step 3: Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(
            alpha_result,
            category_explanations,
        )
        
        # Step 4: Identify risk factors
        risk_factors = self._identify_risk_factors(alpha_result, market_data)
        
        # Step 5: Generate recommendation
        recommendation = self._generate_recommendation(alpha_result)
        
        explanation = AlphaExplanation(
            symbol=alpha_result.symbol,
            alpha_score=alpha_result.final_alpha_score,
            grade=alpha_result.grade,
            summary=summary,
            category_explanations=category_explanations,
            strengths=strengths,
            weaknesses=weaknesses,
            risk_factors=risk_factors,
            recommendation=recommendation,
        )
        
        return explanation
    
    def _explain_categories(
        self,
        alpha_result: AlphaResult,
    ) -> Dict[str, CategoryExplanation]:
        """
        Generate explanations for each category.
        
        Args:
            alpha_result: AlphaResult to explain
            
        Returns:
            Dictionary of category_name -> CategoryExplanation
        """
        explanations = {}
        
        for cat_name, cat_obj in alpha_result.categories.items():
            score = cat_obj.score
            weight = cat_obj.weight
            contribution = score * weight
            
            # Generate rating
            rating = self._generate_rating(score)
            
            # Generate key factors
            key_factors = self._generate_category_factors(cat_name, score)
            
            explanations[cat_name] = CategoryExplanation(
                category=cat_name,
                score=score,
                weight=weight,
                contribution=contribution,
                rating=rating,
                key_factors=key_factors,
            )
        
        return explanations
    
    def _generate_rating(self, score: float) -> str:
        """
        Generate star rating based on score.
        
        Args:
            score: Category score (0-100)
            
        Returns:
            Star rating string (★★★★★ to ★☆☆☆☆)
        """
        for stars, threshold in sorted(self.rating_thresholds.items(), reverse=True):
            if score >= threshold:
                return "★" * stars + "☆" * (5 - stars)
        return "★☆☆☆☆"
    
    def _generate_category_factors(self, category: str, score: float) -> List[str]:
        """
        Generate key factors for a category based on score.
        
        Args:
            category: Category name
            score: Category score
            
        Returns:
            List of key factor descriptions
        """
        factors = []
        
        # Category-specific factor generation
        if category == "technical":
            if score >= 80:
                factors.append("Strong trend alignment")
                factors.append("Momentum indicators bullish")
            elif score >= 60:
                factors.append("Moderate trend strength")
            else:
                factors.append("Weak or conflicting trend signals")
        
        elif category == "volume":
            if score >= 80:
                factors.append("Volume spike detected")
                factors.append("Strong institutional participation")
            elif score >= 60:
                factors.append("Above-average volume")
            else:
                factors.append("Low volume activity")
        
        elif category == "options":
            if score >= 80:
                factors.append("Bullish options flow")
                factors.append("High call put ratio")
            elif score >= 60:
                factors.append("Moderate options activity")
            else:
                factors.append("Bearish options positioning")
        
        elif category == "fundamental":
            if score >= 80:
                factors.append("Strong fundamentals")
                factors.append("Attractive valuation")
            elif score >= 60:
                factors.append("Decent fundamentals")
            else:
                factors.append("Weak fundamentals")
        
        elif category == "sentiment":
            if score >= 80:
                factors.append("Positive market sentiment")
                factors.append("Social media buzz positive")
            elif score >= 60:
                factors.append("Neutral sentiment")
            else:
                factors.append("Negative sentiment")
        
        elif category == "macro":
            if score >= 80:
                factors.append("Favorable macro environment")
            elif score >= 60:
                factors.append("Neutral macro conditions")
            else:
                factors.append("Challenging macro environment")
        
        elif category == "sector":
            if score >= 80:
                factors.append("Sector outperforming market")
            elif score >= 60:
                factors.append("Sector performance neutral")
            else:
                factors.append("Sector underperforming")
        
        return factors
    
    def _generate_summary(
        self,
        alpha_result: AlphaResult,
        category_explanations: Dict[str, CategoryExplanation],
    ) -> str:
        """
        Generate summary of alpha score.
        
        Args:
            alpha_result: AlphaResult
            category_explanations: Category explanations
            
        Returns:
            Summary string
        """
        grade = alpha_result.grade
        score = alpha_result.final_alpha_score
        
        # Find top contributing categories
        top_categories = sorted(
            category_explanations.items(),
            key=lambda x: x[1].contribution,
            reverse=True,
        )[:3]
        
        top_cat_names = [cat for cat, _ in top_categories]
        
        if grade == AlphaGrade.INSTITUTIONAL:
            summary = (
                f"{alpha_result.symbol} shows exceptional institutional-grade setup "
                f"with alpha score of {score:.1f}. Strong performance in "
                f"{', '.join(top_cat_names)}. All filters passed."
            )
        elif grade == AlphaGrade.EXCELLENT:
            summary = (
                f"{alpha_result.symbol} exhibits excellent characteristics "
                f"with alpha score of {score:.1f}. Key strengths in "
                f"{', '.join(top_cat_names)}. Suitable for prediction."
            )
        elif grade == AlphaGrade.GOOD:
            summary = (
                f"{alpha_result.symbol} displays good setup with alpha score of {score:.1f}. "
                f"Moderate strength across multiple categories. Acceptable for prediction."
            )
        elif grade == AlphaGrade.AVERAGE:
            summary = (
                f"{alpha_result.symbol} shows average characteristics "
                f"with alpha score of {score:.1f}. Mixed signals across categories. "
                f"Proceed with caution."
            )
        else:
            summary = (
                f"{alpha_result.symbol} fails to meet minimum criteria "
                f"with alpha score of {score:.1f}. Not recommended for prediction."
            )
        
        return summary
    
    def _identify_strengths_weaknesses(
        self,
        alpha_result: AlphaResult,
        category_explanations: Dict[str, CategoryExplanation],
    ) -> tuple[List[str], List[str]]:
        """
        Identify strengths and weaknesses from category scores.
        
        Args:
            alpha_result: AlphaResult
            category_explanations: Category explanations
            
        Returns:
            Tuple of (strengths, weaknesses)
        """
        strengths = []
        weaknesses = []
        
        for cat_name, cat_expl in category_explanations.items():
            if cat_expl.score >= 80:
                strengths.append(f"{cat_name.title()} strong ({cat_expl.rating})")
            elif cat_expl.score <= 40:
                weaknesses.append(f"{cat_name.title()} weak ({cat_expl.rating})")
        
        # Check filter status
        if alpha_result.passed_filters:
            strengths.append("All risk filters passed")
        else:
            weaknesses.append(f"Filters failed: {', '.join(alpha_result.filter_reasons)}")
        
        return strengths, weaknesses
    
    def _identify_risk_factors(
        self,
        alpha_result: AlphaResult,
        market_data: Optional[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify risk factors from market data and alpha result.
        
        Args:
            alpha_result: AlphaResult
            market_data: Optional market data
            
        Returns:
            List of risk factor descriptions
        """
        risk_factors = []
        
        # Check filter failures
        if not alpha_result.passed_filters:
            risk_factors.extend(alpha_result.filter_reasons)
        
        # Check market data if available
        if market_data:
            volatility = market_data.get("volatility", 0)
            if volatility > 0.10:
                risk_factors.append(f"High volatility ({volatility:.1%})")
            
            spread_bps = market_data.get("spread_bps", 0)
            if spread_bps > 5:
                risk_factors.append(f"Wide spread ({spread_bps:.1f} bps)")
            
            adv = market_data.get("adv", 0)
            if adv < 500000:
                risk_factors.append(f"Low liquidity (ADV: {adv:,.0f})")
        
        # Check for low scores in critical categories
        if alpha_result.categories.get("technical", AlphaCategory("technical", 0, 0)).score < 50:
            risk_factors.append("Weak technical setup")
        
        if alpha_result.categories.get("volume", AlphaCategory("volume", 0, 0)).score < 50:
            risk_factors.append("Low volume confirmation")
        
        return risk_factors
    
    def _generate_recommendation(self, alpha_result: AlphaResult) -> str:
        """
        Generate recommendation based on alpha result.
        
        Args:
            alpha_result: AlphaResult
            
        Returns:
            Recommendation string
        """
        grade = alpha_result.grade
        
        if grade == AlphaGrade.INSTITUTIONAL:
            return "STRONG BUY - Institutional-grade setup with high confidence"
        elif grade == AlphaGrade.EXCELLENT:
            return "BUY - Excellent setup with good risk/reward"
        elif grade == AlphaGrade.GOOD:
            return "MODERATE BUY - Acceptable setup, monitor closely"
        elif grade == AlphaGrade.AVERAGE:
            return "HOLD - Average setup, not ideal for new positions"
        else:
            return "AVOID - Fails minimum criteria, do not predict"
    
    def explain_batch(
        self,
        alpha_results: Dict[str, AlphaResult],
        market_data_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, AlphaExplanation]:
        """
        Generate explanations for multiple stocks.
        
        Args:
            alpha_results: Dictionary of symbol -> AlphaResult
            market_data_map: Optional dictionary of symbol -> market data
            
        Returns:
            Dictionary of symbol -> AlphaExplanation
        """
        self._logger.info(f"Generating batch explanations for {len(alpha_results)} stocks")
        
        explanations = {}
        for symbol, result in alpha_results.items():
            market_data = market_data_map.get(symbol) if market_data_map else None
            explanations[symbol] = self.explain_alpha(result, market_data)
        
        return explanations
    
    def format_explanation_for_display(
        self,
        explanation: AlphaExplanation,
    ) -> str:
        """
        Format explanation for display in dashboard or logs.
        
        Args:
            explanation: AlphaExplanation
            
        Returns:
            Formatted string
        """
        lines = [
            f"{'='*60}",
            f"SYMBOL: {explanation.symbol}",
            f"ALPHA SCORE: {explanation.alpha_score:.2f} ({explanation.grade.value.upper()})",
            f"{'='*60}",
            f"\nSUMMARY:",
            f"  {explanation.summary}",
            f"\nCATEGORY BREAKDOWN:",
        ]
        
        for cat_name, cat_expl in explanation.category_explanations.items():
            lines.append(
                f"  {cat_name.title():15s}: {cat_expl.score:5.1f} "
                f"(weight: {cat_expl.weight:.3f}) {cat_expl.rating}"
            )
            for factor in cat_expl.key_factors:
                lines.append(f"    - {factor}")
        
        if explanation.strengths:
            lines.append(f"\nSTRENGTHS:")
            for strength in explanation.strengths:
                lines.append(f"  ✓ {strength}")
        
        if explanation.weaknesses:
            lines.append(f"\nWEAKNESSES:")
            for weakness in explanation.weaknesses:
                lines.append(f"  ✗ {weakness}")
        
        if explanation.risk_factors:
            lines.append(f"\nRISK FACTORS:")
            for risk in explanation.risk_factors:
                lines.append(f"  ⚠ {risk}")
        
        lines.append(f"\nRECOMMENDATION:")
        lines.append(f"  {explanation.recommendation}")
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)
