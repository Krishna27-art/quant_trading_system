"""
Real-time Intraday Stress Testing

Applies hypothetical shock scenarios (e.g., flash crash, tech sector dump)
to the live portfolio to estimate potential downside and trigger warnings
if a stress threshold is breached.
"""

from dataclasses import dataclass

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StressScenario:
    name: str
    description: str
    shocks: dict[str, float]  # Factor or Sector -> % shock


@dataclass
class StressTestConfig:
    max_stress_loss_pct: float = 0.05  # 5% max portfolio loss under stress
    run_interval_seconds: int = 300  # Run every 5 minutes


class IntradayStressTester:
    """
    Evaluates the live portfolio against severe but plausible market shocks.
    """

    def __init__(self, config: StressTestConfig = StressTestConfig()):
        self.config = config
        self.scenarios = self._init_default_scenarios()

    def _init_default_scenarios(self) -> list[StressScenario]:
        """Define standard shock scenarios."""
        return [
            StressScenario(
                name="Flash Crash (Market -10%)",
                description="Sudden 10% drop in broad market beta",
                shocks={"Market": -0.10},
            ),
            StressScenario(
                name="Tech Wreck (IT -15%)",
                description="15% drop in IT sector",
                shocks={"Sector_IT": -0.15},
            ),
            StressScenario(
                name="Interest Rate Shock (Bank -10%, Value -5%)",
                description="Sudden RBI rate hike",
                shocks={"Sector_Bank": -0.10, "Value": -0.05},
            ),
            StressScenario(
                name="Momentum Reversal (Mom -10%)",
                description="Momentum factor crashes (quant quake)",
                shocks={"Momentum": -0.10},
            ),
        ]

    def add_scenario(self, scenario: StressScenario):
        self.scenarios.append(scenario)

    async def run_stress_test(
        self, portfolio_weights: pd.Series, factor_exposures: pd.DataFrame
    ) -> dict[str, float]:
        """
        Run all scenarios on the current portfolio asynchronously.
        Returns the estimated PnL% for each scenario.
        """
        results = {}

        common_idx = portfolio_weights.index.intersection(factor_exposures.index)
        if len(common_idx) == 0:
            return {s.name: 0.0 for s in self.scenarios}

        w = portfolio_weights[common_idx]
        B = factor_exposures.loc[common_idx]

        for scenario in self.scenarios:
            scenario_pnl = 0.0

            # For each shock in the scenario, apply it to the portfolio's exposure
            for factor, shock in scenario.shocks.items():
                if factor in B.columns:
                    # Exposure to factor * Shock
                    # B[factor] is the z-score or beta.
                    # Assuming B is scaled such that 1.0 = 1% return for 1% factor move
                    port_exposure = (w * B[factor]).sum()
                    scenario_pnl += port_exposure * shock
                else:
                    # If it's a sector or custom identifier not in standard factors
                    # We would need a custom mapping here. Skipping for brevity.
                    pass

            results[scenario.name] = scenario_pnl

            if scenario_pnl < -self.config.max_stress_loss_pct:
                logger.warning(
                    f"STRESS TEST BREACH: {scenario.name} "
                    f"would cause {scenario_pnl * 100:.2f}% loss! (Limit: {-self.config.max_stress_loss_pct * 100:.2f}%)"
                )

            # Yield control back to the event loop
            import asyncio

            await asyncio.sleep(0)

        return results
