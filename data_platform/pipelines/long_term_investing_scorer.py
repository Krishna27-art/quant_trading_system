"""
Long-Term Investing Scorer - Business Quality and Compounding Engine

Institutional-grade long-term investing analysis focused on finding businesses
that can compound earnings for 10-30 years.

Scoring weights (institutional framework):
- Industry Growth: 15%
- Revenue Growth: 15%
- Earnings Growth: 20%
- ROE/ROCE: 10%
- Debt Quality: 10%
- Cash Flow: 10%
- Management & Governance: 10%
- Competitive Moat: 5%
- Valuation: 5%

Goal: Find companies that can become 2x, 5x, 10x, 50x, 100x over long periods
through earnings compounding and multiple expansion.

Institutional process:
Economy → Industry → Business → Management → Financials → Valuation → Portfolio → Long Hold
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


from config.universe import NSE_UNIVERSE
from utils.logger import get_logger

logger = get_logger("long_term_investing_scorer")


class StructuralTheme(str, Enum):
    """Long-term structural themes for India."""
    MANUFACTURING = "manufacturing"
    DIGITALIZATION = "digitalization"
    AI = "ai"
    DEFENCE = "defence"
    ELECTRONICS = "electronics"
    RENEWABLE_ENERGY = "renewable_energy"
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    LOGISTICS = "logistics"
    PREMIUM_CONSUMER = "premium_consumer"
    INFRASTRUCTURE = "infrastructure"
    CHEMICALS_SPECIALITY = "chemicals_speciality"


class MoatType(str, Enum):
    """Types of competitive moats."""
    BRAND = "brand"
    COST_ADVANTAGE = "cost_advantage"
    NETWORK_EFFECT = "network_effect"
    SWITCHING_COSTS = "switching_costs"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    REGULATORY_LICENSES = "regulatory_licenses"
    DISTRIBUTION_NETWORK = "distribution_network"
    SCALE = "scale"
    NONE = "none"


@dataclass
class IndustryThemeMetrics:
    """Industry/theme analysis for structural trends."""
    sector: str
    theme: Optional[StructuralTheme]
    theme_growth_potential: float  # 0-100
    market_size_tam_cr: float  # Total Addressable Market in crores
    cagr_expected_10y: float  # Expected 10-year CAGR
    government_support: bool  # Policy tailwinds
    export_potential: bool
    theme_score: float  # 0-100
    timestamp: datetime


@dataclass
class FinancialQualityMetrics:
    """Financial quality analysis."""
    symbol: str
    
    # Growth metrics
    revenue_cagr_3y: float
    revenue_cagr_5y: float
    revenue_cagr_10y: Optional[float]
    earnings_cagr_3y: float
    earnings_cagr_5y: float
    earnings_cagr_10y: Optional[float]
    
    # Consistency
    revenue_growth_consistency: float  # 0-100 (how consistent is growth)
    earnings_growth_consistency: float
    
    # Margins
    operating_margin_avg_3y: float
    operating_margin_trend: str  # improving, stable, declining
    net_margin_avg_3y: float
    net_margin_trend: str
    
    # Return ratios
    roe_avg_3y: float
    roe_trend: str
    roce_avg_3y: float
    roce_trend: str
    roic_avg_3y: float
    
    # Cash flow
    operating_cash_flow_cagr_3y: float
    fcf_margin_avg_3y: float
    fcf_conversion_pct: float  # FCF as % of net income
    
    # Balance sheet
    debt_to_equity: float
    interest_coverage_ratio: float
    net_debt_to_ebitda: float
    
    # Quality score
    financial_quality_score: float  # 0-100
    
    timestamp: datetime


@dataclass
class ManagementQualityMetrics:
    """Management quality assessment."""
    symbol: str
    
    # Capital allocation
    capex_efficiency: float  # 0-100
    roi_on_capex: float
    dividend_payout_consistency: float  # 0-100
    
    # Governance
    promoter_holdings_pct: float
    institutional_holdings_pct: float
    corporate_governance_score: float  # 0-100
    
    # Execution
    project_execution_track_record: float  # 0-100
    strategic_clarity: float  # 0-100
    
    # Transparency
    disclosure_quality: float  # 0-100
    communication_frequency: float  # 0-100
    
    # Overall score
    management_quality_score: float  # 0-100
    
    timestamp: datetime


@dataclass
class MoatMetrics:
    """Competitive moat analysis."""
    symbol: str
    has_moat: bool
    moat_type: Optional[MoatType]
    moat_strength: float  # 0-100
    moat_durability: str  # sustainable, weakening, strengthening
    
    # Moat indicators
    market_share_trend: str  # gaining, stable, losing
    pricing_power: float  # 0-100
    customer_loyalty: float  # 0-100
    competitive_advantages: list[str]
    
    timestamp: datetime


@dataclass
class ValuationMetrics:
    """Valuation analysis using GARP/GARV methodology."""
    symbol: str
    
    # Multiples
    pe_ratio: float
    pe_historical_percentile: float  # 0-100
    pb_ratio: float
    pb_historical_percentile: float
    ev_ebitda: float
    ev_ebitda_historical_percentile: float
    peg_ratio: float
    
    # Growth vs Valuation
    growth_adjusted_pe: float
    garv_score: float  # Growth at Reasonable Valuation score 0-100
    
    # Intrinsic value
    intrinsic_value_estimate: float
    margin_of_safety_pct: float
    
    # Valuation score
    valuation_score: float  # 0-100
    
    timestamp: datetime


@dataclass
class ScalabilityMetrics:
    """Scalability assessment for 10x potential."""
    symbol: str
    can_scale_10x: bool
    current_market_cap_cr: float
    potential_market_cap_10y_cr: float
    
    # Scalability factors
    addressable_market_size: float  # 0-100
    product_scalability: float  # 0-100
    geographic_expansion_potential: float  # 0-100
    business_model_scalability: float  # 0-100
    
    # Constraints
    scalability_constraints: list[str]
    
    # Overall score
    scalability_score: float  # 0-100
    
    timestamp: datetime


@dataclass
class LongTermInvestingScore:
    """Complete long-term investing score."""
    symbol: str
    name: str
    sector: str
    
    # Component scores (0-100)
    industry_growth_score: float
    revenue_growth_score: float
    earnings_growth_score: float
    roe_roce_score: float
    debt_quality_score: float
    cash_flow_score: float
    management_governance_score: float
    competitive_moat_score: float
    valuation_score: float
    
    # Weighted final score
    final_score: float  # 0-100
    
    # Supporting metrics
    industry_theme: Optional[IndustryThemeMetrics]
    financial_quality: Optional[FinancialQualityMetrics]
    management_quality: Optional[ManagementQualityMetrics]
    moat: Optional[MoatMetrics]
    valuation: Optional[ValuationMetrics]
    scalability: Optional[ScalabilityMetrics]
    
    # Investment thesis
    investment_thesis: str
    key_risks: list[str]
    
    # Potential returns
    expected_10y_return_pct: float
    potential_multibagger: str  # 2x, 5x, 10x, 50x, 100x, none
    
    # Position sizing
    conviction_level: str  # high, medium, low
    suggested_allocation_pct: float
    
    # Ranking
    overall_rank: int
    qualifies: bool  # True if score >= 80
    
    timestamp: datetime


class LongTermInvestingScorer:
    """
    Institutional long-term investing scoring engine.
    
    Scoring weights:
    - Industry Growth: 15%
    - Revenue Growth: 15%
    - Earnings Growth: 20%
    - ROE/ROCE: 10%
    - Debt Quality: 10%
    - Cash Flow: 10%
    - Management & Governance: 10%
    - Competitive Moat: 5%
    - Valuation: 5%
    """

    def __init__(
        self,
        min_score_threshold: float = 80.0,
        use_mock_data: bool = True
    ):
        """
        Initialize long-term investing scorer.
        
        Parameters
        ----------
        min_score_threshold
            Minimum score to qualify (default: 80)
        use_mock_data
            Use mock data for development
        """
        self.min_score_threshold = min_score_threshold
        self.use_mock_data = use_mock_data
        
        # Scoring weights
        self.weights = {
            "industry_growth": 0.15,
            "revenue_growth": 0.15,
            "earnings_growth": 0.20,
            "roe_roce": 0.10,
            "debt_quality": 0.10,
            "cash_flow": 0.10,
            "management_governance": 0.10,
            "competitive_moat": 0.05,
            "valuation": 0.05
        }

    def score_universe(self) -> list[LongTermInvestingScore]:
        """
        Score entire NSE universe for long-term investing.
        
        Returns
        -------
        list[LongTermInvestingScore]
            Ranked list of stocks with scores
        """
        logger.info("Starting long-term investing universe scoring")
        
        universe = NSE_UNIVERSE
        logger.info(f"Scoring {len(universe)} stocks")
        
        # Score each stock
        all_scores = []
        for stock in universe:
            try:
                score = self._score_stock(stock)
                all_scores.append(score)
            except Exception as e:
                logger.error(f"Failed to score {stock['symbol']}: {e}")
                continue
        
        # Rank by final score
        all_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign ranks
        for idx, score in enumerate(all_scores, 1):
            score.overall_rank = idx
            score.qualifies = score.final_score >= self.min_score_threshold
        
        qualified_count = sum(1 for s in all_scores if s.qualifies)
        logger.info(f"Scoring complete: {qualified_count}/{len(all_scores)} stocks qualified")
        
        return all_scores

    def _score_stock(self, stock: dict) -> LongTermInvestingScore:
        """Score a single stock."""
        symbol = stock["symbol"]
        
        # Calculate component scores
        industry_theme = self._analyze_industry_theme(stock["sector"], symbol)
        industry_growth_score = industry_theme.theme_score if industry_theme else 50
        
        financial_quality = self._analyze_financial_quality(symbol)
        revenue_growth_score = self._score_revenue_growth(financial_quality)
        earnings_growth_score = self._score_earnings_growth(financial_quality)
        roe_roce_score = self._score_roe_roce(financial_quality)
        debt_quality_score = self._score_debt_quality(financial_quality)
        cash_flow_score = self._score_cash_flow(financial_quality)
        
        management_quality = self._analyze_management_quality(symbol)
        management_governance_score = management_quality.management_quality_score if management_quality else 50
        
        moat = self._analyze_moat(symbol)
        competitive_moat_score = moat.moat_strength if moat else 50
        
        valuation = self._analyze_valuation(symbol)
        valuation_score = valuation.garv_score if valuation else 50
        
        scalability = self._analyze_scalability(symbol)
        
        # Calculate weighted final score
        final_score = (
            industry_growth_score * self.weights["industry_growth"] +
            revenue_growth_score * self.weights["revenue_growth"] +
            earnings_growth_score * self.weights["earnings_growth"] +
            roe_roce_score * self.weights["roe_roce"] +
            debt_quality_score * self.weights["debt_quality"] +
            cash_flow_score * self.weights["cash_flow"] +
            management_governance_score * self.weights["management_governance"] +
            competitive_moat_score * self.weights["competitive_moat"] +
            valuation_score * self.weights["valuation"]
        )
        
        # Generate investment thesis
        thesis = self._generate_investment_thesis(
            stock, industry_theme, financial_quality, moat, valuation
        )
        
        # Calculate potential returns
        expected_return, multibagger = self._calculate_potential_returns(
            financial_quality, valuation, scalability
        )
        
        # Determine conviction and allocation
        conviction, allocation = self._determine_conviction(final_score, thesis)
        
        return LongTermInvestingScore(
            symbol=symbol,
            name=stock["name"],
            sector=stock["sector"],
            industry_growth_score=industry_growth_score,
            revenue_growth_score=revenue_growth_score,
            earnings_growth_score=earnings_growth_score,
            roe_roce_score=roe_roce_score,
            debt_quality_score=debt_quality_score,
            cash_flow_score=cash_flow_score,
            management_governance_score=management_governance_score,
            competitive_moat_score=competitive_moat_score,
            valuation_score=valuation_score,
            final_score=final_score,
            industry_theme=industry_theme,
            financial_quality=financial_quality,
            management_quality=management_quality,
            moat=moat,
            valuation=valuation,
            scalability=scalability,
            investment_thesis=thesis,
            key_risks=self._identify_key_risks(financial_quality, valuation),
            expected_10y_return_pct=expected_return,
            potential_multibagger=multibagger,
            conviction_level=conviction,
            suggested_allocation_pct=allocation,
            overall_rank=0,
            qualifies=False,
            timestamp=datetime.now()
        )

    def _analyze_industry_theme(self, sector: str, symbol: str) -> Optional[IndustryThemeMetrics]:
        """Analyze industry theme for structural trends."""
        if self.use_mock_data:
            import random
            
            # Map sectors to themes
            theme_mapping = {
                "Technology": StructuralTheme.DIGITALIZATION,
                "Financial Services": StructuralTheme.FINANCIAL_SERVICES,
                "Healthcare": StructuralTheme.HEALTHCARE,
                "Energy": StructuralTheme.RENEWABLE_ENERGY,
                "Capital Goods": StructuralTheme.MANUFACTURING,
                "Chemicals": StructuralTheme.CHEMICALS_SPECIALITY,
                "Consumer Goods": StructuralTheme.PREMIUM_CONSUMER,
                "Automobile": StructuralTheme.MANUFACTURING,
                "Telecommunication": StructuralTheme.DIGITALIZATION,
                "Materials": StructuralTheme.INFRASTRUCTURE
            }
            
            theme = theme_mapping.get(sector)
            
            if theme:
                return IndustryThemeMetrics(
                    sector=sector,
                    theme=theme,
                    theme_growth_potential=random.uniform(60, 95),
                    market_size_tam_cr=random.uniform(100000, 5000000),
                    cagr_expected_10y=random.uniform(10.0, 25.0),
                    government_support=random.random() < 0.6,
                    support_potential=random.random() < 0.5,
                    theme_score=random.uniform(65, 95),
                    timestamp=datetime.now()
                )
        
        return None

    def _analyze_financial_quality(self, symbol: str) -> Optional[FinancialQualityMetrics]:
        """Analyze financial quality."""
        if self.use_mock_data:
            import random
            
            revenue_cagr_3y = random.uniform(5.0, 30.0)
            revenue_cagr_5y = random.uniform(8.0, 25.0)
            earnings_cagr_3y = random.uniform(8.0, 35.0)
            earnings_cagr_5y = random.uniform(10.0, 30.0)
            
            operating_margin = random.uniform(10.0, 35.0)
            net_margin = random.uniform(8.0, 25.0)
            roe = random.uniform(12.0, 30.0)
            roce = random.uniform(14.0, 28.0)
            
            # Calculate quality score
            quality = (
                min(100, revenue_cagr_5y * 3) +
                min(100, earnings_cagr_5y * 3) +
                operating_margin * 2 +
                net_margin * 2 +
                roe * 2 +
                roce * 2
            ) / 12.0
            
            return FinancialQualityMetrics(
                symbol=symbol,
                revenue_cagr_3y=revenue_cagr_3y,
                revenue_cagr_5y=revenue_cagr_5y,
                revenue_cagr_10y=None,
                earnings_cagr_3y=earnings_cagr_3y,
                earnings_cagr_5y=earnings_cagr_5y,
                earnings_cagr_10y=None,
                revenue_growth_consistency=random.uniform(60, 95),
                earnings_growth_consistency=random.uniform(55, 90),
                operating_margin_avg_3y=operating_margin,
                operating_margin_trend=random.choice(["improving", "stable", "declining"]),
                net_margin_avg_3y=net_margin,
                net_margin_trend=random.choice(["improving", "stable", "declining"]),
                roe_avg_3y=roe,
                roe_trend=random.choice(["improving", "stable", "declining"]),
                roce_avg_3y=roce,
                roce_trend=random.choice(["improving", "stable", "declining"]),
                roic_avg_3y=random.uniform(10.0, 25.0),
                operating_cash_flow_cagr_3y=random.uniform(8.0, 30.0),
                fcf_margin_avg_3y=random.uniform(5.0, 15.0),
                fcf_conversion_pct=random.uniform(70.0, 95.0),
                debt_to_equity=random.uniform(0.1, 1.5),
                interest_coverage_ratio=random.uniform(3.0, 15.0),
                net_debt_to_ebitda=random.uniform(-0.5, 3.0),
                financial_quality_score=quality,
                timestamp=datetime.now()
            )
        
        return None

    def _analyze_management_quality(self, symbol: str) -> Optional[ManagementQualityMetrics]:
        """Analyze management quality."""
        if self.use_mock_data:
            import random
            
            quality = random.uniform(50, 95)
            
            return ManagementQualityMetrics(
                symbol=symbol,
                capex_efficiency=random.uniform(60, 90),
                roi_on_capex=random.uniform(12.0, 25.0),
                dividend_payout_consistency=random.uniform(50, 90),
                promoter_holdings_pct=random.uniform(30.0, 75.0),
                institutional_holdings_pct=random.uniform(15.0, 50.0),
                corporate_governance_score=random.uniform(60, 90),
                project_execution_track_record=random.uniform(55, 90),
                strategic_clarity=random.uniform(60, 85),
                disclosure_quality=random.uniform(55, 90),
                communication_frequency=random.uniform(60, 85),
                management_quality_score=quality,
                timestamp=datetime.now()
            )
        
        return None

    def _analyze_moat(self, symbol: str) -> Optional[MoatMetrics]:
        """Analyze competitive moat."""
        if self.use_mock_data:
            import random
            
            # 50% chance of having a moat
            if random.random() < 0.5:
                moat_types = [
                    MoatType.BRAND,
                    MoatType.COST_ADVANTAGE,
                    MoatType.NETWORK_EFFECT,
                    MoatType.SWITCHING_COSTS,
                    MoatType.SCALE
                ]
                
                return MoatMetrics(
                    symbol=symbol,
                    has_moat=True,
                    moat_type=random.choice(moat_types),
                    moat_strength=random.uniform(60, 95),
                    moat_durability=random.choice(["sustainable", "weakening", "strengthening"]),
                    market_share_trend=random.choice(["gaining", "stable", "losing"]),
                    pricing_power=random.uniform(50, 85),
                    customer_loyalty=random.uniform(55, 90),
                    competitive_advantages=["Brand strength", "Distribution network", "Technology"],
                    timestamp=datetime.now()
                )
        
        return MoatMetrics(
            symbol=symbol,
            has_moat=False,
            moat_type=None,
            moat_strength=30,
            moat_durability="weakening",
            market_share_trend="losing",
            pricing_power=30,
            customer_loyalty=40,
            competitive_advantages=[],
            timestamp=datetime.now()
        )

    def _analyze_valuation(self, symbol: str) -> Optional[ValuationMetrics]:
        """Analyze valuation using GARP/GARV."""
        if self.use_mock_data:
            import random
            
            pe = random.uniform(15.0, 45.0)
            peg = random.uniform(1.0, 3.0)
            
            # GARV score: lower PEG and reasonable historical percentile
            garv = max(0, 100 - (peg - 1) * 30)
            
            return ValuationMetrics(
                symbol=symbol,
                pe_ratio=pe,
                pe_historical_percentile=random.uniform(30, 80),
                pb_ratio=random.uniform(2.0, 8.0),
                pb_historical_percentile=random.uniform(35, 75),
                ev_ebitda=random.uniform(10.0, 25.0),
                ev_ebitda_historical_percentile=random.uniform(30, 70),
                peg_ratio=peg,
                growth_adjusted_pe=pe / peg if peg > 0 else pe,
                garv_score=garv,
                intrinsic_value_estimate=random.uniform(100.0, 3000.0),
                margin_of_safety_pct=random.uniform(-20.0, 40.0),
                valuation_score=garv,
                timestamp=datetime.now()
            )
        
        return None

    def _analyze_scalability(self, symbol: str) -> Optional[ScalabilityMetrics]:
        """Analyze scalability for 10x potential."""
        if self.use_mock_data:
            import random
            
            can_scale = random.random() < 0.4
            
            return ScalabilityMetrics(
                symbol=symbol,
                can_scale_10x=can_scale,
                current_market_cap_cr=random.uniform(5000, 200000),
                potential_market_cap_10y_cr=random.uniform(50000, 2000000),
                addressable_market_size=random.uniform(60, 95),
                product_scalability=random.uniform(50, 90),
                geographic_expansion_potential=random.uniform(40, 85),
                business_model_scalability=random.uniform(55, 90),
                scalability_constraints=random.sample(["Regulatory", "Competition", "Capital"], k=random.randint(0, 2)),
                scalability_score=random.uniform(50, 90) if can_scale else 30,
                timestamp=datetime.now()
            )
        
        return None

    def _score_revenue_growth(self, financial: Optional[FinancialQualityMetrics]) -> float:
        """Score revenue growth."""
        if not financial:
            return 50
        return min(100, financial.revenue_cagr_5y * 3)

    def _score_earnings_growth(self, financial: Optional[FinancialQualityMetrics]) -> float:
        """Score earnings growth."""
        if not financial:
            return 50
        return min(100, financial.earnings_cagr_5y * 3)

    def _score_roe_roce(self, financial: Optional[FinancialQualityMetrics]) -> float:
        """Score ROE/ROCE."""
        if not financial:
            return 50
        avg_return = (financial.roe_avg_3y + financial.roce_avg_3y) / 2
        return min(100, avg_return * 3)

    def _score_debt_quality(self, financial: Optional[FinancialQualityMetrics]) -> float:
        """Score debt quality."""
        if not financial:
            return 50
        # Lower debt is better
        if financial.debt_to_equity < 0.3:
            return 100
        elif financial.debt_to_equity < 0.5:
            return 80
        elif financial.debt_to_equity < 0.8:
            return 60
        elif financial.debt_to_equity < 1.2:
            return 40
        else:
            return 20

    def _score_cash_flow(self, financial: Optional[FinancialQualityMetrics]) -> float:
        """Score cash flow quality."""
        if not financial:
            return 50
        return min(100, financial.fcf_conversion_pct)

    def _generate_investment_thesis(
        self,
        stock: dict,
        theme: Optional[IndustryThemeMetrics],
        financial: Optional[FinancialQualityMetrics],
        moat: Optional[MoatMetrics],
        valuation: Optional[ValuationMetrics]
    ) -> str:
        """Generate investment thesis."""
        parts = []
        
        if theme and theme.theme:
            parts.append(f"Beneficiary of {theme.theme.value.replace('_', ' ').title()} theme")
        
        if financial:
            if financial.revenue_cagr_5y > 15:
                parts.append(f"Strong revenue growth ({financial.revenue_cagr_5y:.1f}% CAGR)")
            if financial.roe_avg_3y > 18:
                parts.append(f"High ROE ({financial.roe_avg_3y:.1f}%)")
        
        if moat and moat.has_moat:
            parts.append(f"Sustainable {moat.moat_type.value.replace('_', ' ')} moat")
        
        if not parts:
            return "Adequate business with moderate growth potential"
        
        return ". ".join(parts) + "."

    def _identify_key_risks(
        self,
        financial: Optional[FinancialQualityMetrics],
        valuation: Optional[ValuationMetrics]
    ) -> list[str]:
        """Identify key risks."""
        risks = []
        
        if financial:
            if financial.debt_to_equity > 1.0:
                risks.append("High leverage")
            if financial.operating_margin_trend == "declining":
                risks.append("Margin pressure")
            if financial.revenue_growth_consistency < 60:
                risks.append("Inconsistent growth")
        
        if valuation and valuation.margin_of_safety_pct < 0:
            risks.append("Valuation risk")
        
        if not risks:
            risks.append("Execution risk")
        
        return risks

    def _calculate_potential_returns(
        self,
        financial: Optional[FinancialQualityMetrics],
        valuation: Optional[ValuationMetrics],
        scalability: Optional[ScalabilityMetrics]
    ) -> tuple[float, str]:
        """Calculate potential 10-year returns."""
        if not financial:
            return 12.0, "none"
        
        # Base return from earnings growth
        base_return = financial.earnings_cagr_5y if financial.earnings_cagr_5y else 12.0
        
        # Add potential multiple expansion
        multiple_expansion = 2.0 if valuation and valuation.garv_score > 70 else 0.0
        
        total_return = base_return + multiple_expansion
        
        # Determine multibagger potential
        if scalability and scalability.can_scale_10x and total_return > 25:
            multibagger = "10x"
        elif total_return > 20:
            multibagger = "5x"
        elif total_return > 15:
            multibagger = "2x"
        else:
            multibagger = "none"
        
        return total_return, multibagger

    def _determine_conviction(self, score: float, thesis: str) -> tuple[str, float]:
        """Determine conviction level and allocation."""
        if score >= 90:
            return "high", 8.0  # 8% allocation
        elif score >= 80:
            return "medium", 5.0  # 5% allocation
        elif score >= 70:
            return "low", 3.0  # 3% allocation
        else:
            return "low", 2.0  # 2% allocation
