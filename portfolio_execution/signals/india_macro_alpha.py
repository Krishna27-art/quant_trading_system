import pandas as pd

from data_platform.pipelines.india_macro import IndiaMacroPipeline
from data_platform.pipelines.nse_options import NSEOptionsPipeline
from portfolio_execution.signals.base import AlphaModel, SignalDirection, SignalNorm
from utils.logger import get_logger

logger = get_logger("india_macro_alpha")


class IndiaMacroAlpha(AlphaModel):
    """
    Alpha model that leverages native Indian market data (RBI Macro & NSE Options).
    Generates cross-sectional signals by comparing symbol behavior against macro trends
    and options implied sentiment.
    """

    def __init__(self, name: str = "IndiaMacroAlpha", lookback: int = 10, **kwargs):
        super().__init__(
            name=name,
            lookback=lookback,
            norm=SignalNorm.ZSCORE,
            direction=SignalDirection.LONG_SHORT,
            **kwargs,
        )
        self.macro_pipeline = IndiaMacroPipeline()
        self.options_pipeline = NSEOptionsPipeline()
        self.sleeve = "macro"

    def _compute_raw_signal(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Uses NSE Option Chain Put-Call Ratio (PCR) to derive sentiment
        and adjusts using RBI repo rate context.
        """
        signals = {}
        symbols = data.columns.tolist()

        # Pull latest macro context
        repo_df = self.macro_pipeline.fetch_rbi_repo_rate()
        repo_rate = repo_df["repo_rate"].iloc[-1] if not repo_df.empty else 6.5

        forex_df = self.macro_pipeline.fetch_rbi_forex_reserves()
        forex_val = forex_df["forex_usd_bn"].iloc[-1] if not forex_df.empty else 640.5

        iip_df = self.macro_pipeline.fetch_iip_data()
        iip_val = iip_df["iip_growth"].iloc[-1] if not iip_df.empty else 3.8

        cpi_df = self.macro_pipeline.fetch_inflation_cpi()
        # Fallback if CPI DataFrame doesn't have standard column
        if not cpi_df.empty and "cpi_yoy" in cpi_df.columns:
            cpi_val = cpi_df["cpi_yoy"].iloc[-1]
        else:
            cpi_val = 5.0

        # Combine macro parameters into a single multiplier:
        # Bullish macro: low repo rate, strong IIP, low CPI, growing forex reserves
        macro_score = 0.0

        # Repo Rate component (Neutral 6.5%)
        macro_score += (6.5 - repo_rate) * 0.2

        # CPI Inflation component (Target 4%)
        macro_score += (4.0 - cpi_val) * 0.1

        # IIP Growth component (Target >4%)
        macro_score += (iip_val - 4.0) * 0.1

        # Forex reserves component (Bullish if high, baseline 600 bn)
        macro_score += (forex_val - 600.0) * 0.005

        logger.info(
            f"India Macro Sentiment calculated: {macro_score:.3f} (Repo: {repo_rate}%, CPI: {cpi_val}%, IIP: {iip_val}%, Forex: {forex_val}B)"
        )

        for symbol in symbols:
            # We fetch options chain for the symbol (works mostly for F&O stocks like RELIANCE)
            try:
                opt_chain = self.options_pipeline.fetch_option_chain(symbol)

                # Calculate Put-Call Ratio (PCR) from Open Interest
                total_ce_oi = 0
                total_pe_oi = 0

                if "records" in opt_chain and "data" in opt_chain["records"]:
                    for item in opt_chain["records"]["data"]:
                        if "CE" in item:
                            total_ce_oi += item["CE"].get("openInterest", 0)
                        if "PE" in item:
                            total_pe_oi += item["PE"].get("openInterest", 0)

                pcr = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 1.0

                # Sentiment logic:
                # PCR > 1.2 is bullish (puts being sold)
                # PCR < 0.8 is bearish
                sentiment = pcr - 1.0 + macro_score

                signals[symbol] = sentiment
            except Exception as e:
                logger.debug(f"Could not compute India Macro Alpha for {symbol}: {e}")
                signals[symbol] = 0.0

        return pd.Series(signals)
