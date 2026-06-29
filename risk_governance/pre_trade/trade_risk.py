from datetime import datetime, time

import numpy as np
import pandas as pd


class TradeRiskEngine:
    """
    Trade-Level Risk Controls
    Enforces strict mathematical logic on Stop Losses, Targets, and Holding Times.
    """

    def __init__(self, estimated_slippage_pct: float = 0.001):  # 0.1% slippage
        self.estimated_slippage_pct = estimated_slippage_pct

    def estimate_dynamic_slippage(
        self, order_size_rupees: float, volume_30s_rupees: float, volatility_factor: float = 1.0
    ) -> float:
        """
        Calculates estimated market impact slippage:
        slippage = (order_size_in_rupees / 30_second_traded_volume_in_rupees) * volatility_factor
        """
        if volume_30s_rupees <= 0:
            return self.estimated_slippage_pct

        market_impact = (order_size_rupees / volume_30s_rupees) * 0.02 * volatility_factor
        # Cap/collar between 0.05% and 2.0%
        return max(0.0005, min(market_impact, 0.02))

    def calculate_dynamic_stops_and_targets(
        self,
        df_1min: pd.DataFrame,
        entry_price: float,
        is_buy: bool,
        order_size_rupees: float = 100000.0,
        volume_30s_rupees: float = 10000000.0,
    ):
        """
        Calculates volatility-adjusted Stop Loss (1 std dev of 20-day intraday moves)
        and Dynamic Target (2 * SL_distance + 2 * slippage).

        Args:
            df_1min: DataFrame containing at least 20 days of 1-minute bars.
                     Must have 'Close' column.
        """
        if df_1min is None or len(df_1min) < 30:
            # Fallback if we don't have enough data
            sl_distance_pct = 0.01  # 1% static fallback
        else:
            # Calculate rolling 1-minute returns
            returns = df_1min["close"].pct_change().dropna()

            # Intraday volatility (standard deviation of 1-min returns over the period)
            # Scale it to a typical trade duration (e.g., 90 minutes)
            # 1 std dev of 90-min move = 1-min std dev * sqrt(90)
            sl_distance_pct = returns.std() * np.sqrt(90)

            # Cap and Collar the SL to prevent ridiculous extremes
            sl_distance_pct = max(0.003, min(sl_distance_pct, 0.03))  # between 0.3% and 3.0%

        sl_distance_abs = entry_price * sl_distance_pct

        # Estimate dynamic slippage based on size and current volume
        est_slippage_pct = self.estimate_dynamic_slippage(order_size_rupees, volume_30s_rupees)
        slippage_abs = entry_price * est_slippage_pct

        # Target must be at least 2:1 R:R plus slippage accounting
        min_target_distance_abs = (2 * sl_distance_abs) + (2 * slippage_abs)

        if is_buy:
            stop_loss = entry_price - sl_distance_abs
            target = entry_price + min_target_distance_abs
        else:
            stop_loss = entry_price + sl_distance_abs
            target = entry_price - min_target_distance_abs

        return {
            "stop_loss": round(stop_loss, 2),
            "target": round(target, 2),
            "sl_distance_pct": round(sl_distance_pct * 100, 3),
            "estimated_slippage_pct": round(est_slippage_pct * 100, 3),
        }

    def should_time_stop(self, entry_time: datetime, current_time: datetime) -> bool:
        """
        Time Stop logic:
        1. Exit if holding for > 90 minutes.
        2. Exit if current time is >= 2:45 PM (14:45).
        """
        # Force close at 2:45 PM
        cutoff_time = time(14, 45)
        if current_time.time() >= cutoff_time:
            return True

        # 90 minute hold time limit
        duration_minutes = (current_time - entry_time).total_seconds() / 60.0
        return duration_minutes >= 90.0
