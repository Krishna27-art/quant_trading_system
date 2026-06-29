import numpy as np
import pandas as pd


class MultiObjectiveLabeler:
    """
    Implements Triple Barrier Method and multi-objective extraction (MFE/MAE).
    This computes the actual future outcome of a candidate trade entry.
    """

    @staticmethod
    def get_barriers(
        prices: pd.Series, events: pd.DataFrame, pt_sl: tuple[float, float], min_ret: float
    ) -> pd.DataFrame:
        """
        Calculates the exact timestamps when a target or stop loss is hit.
        prices: A pandas Series of closing prices (or high/lows for more accuracy).
        events: A dataframe indexed by candidate entry times, with column 't1' indicating vertical barrier (expiry time).
        pt_sl: (profit taking multiple, stop loss multiple) applied to min_ret.
        """
        out = events[["t1"]].copy()

        # Pre-initialize columns with correct timezone-aware NaT to prevent mixed dtype warnings/errors
        out["stop_loss_time"] = pd.Series(pd.NaT, index=events.index, dtype=events["t1"].dtype)
        out["target_time"] = pd.Series(pd.NaT, index=events.index, dtype=events["t1"].dtype)

        if pt_sl[0] > 0:
            pt = pt_sl[0] * min_ret
        else:
            pt = pd.Series(index=events.index, dtype=float)  # NaN

        if pt_sl[1] > 0:
            sl = -pt_sl[1] * min_ret
        else:
            sl = pd.Series(index=events.index, dtype=float)  # NaN

        for loc, t1 in events["t1"].fillna(prices.index[-1]).items():
            path_prices = prices[loc:t1]
            if path_prices.empty:
                continue

            # Returns relative to entry
            path_returns = (path_prices / prices[loc]) - 1

            # Find earliest hit
            sl_hits = path_returns[path_returns < sl]
            out.loc[loc, "stop_loss_time"] = sl_hits.index.min() if not sl_hits.empty else pd.NaT

            pt_hits = path_returns[path_returns > pt]
            out.loc[loc, "target_time"] = pt_hits.index.min() if not pt_hits.empty else pd.NaT

            # Extract MFE / MAE
            out.loc[loc, "actual_mfe"] = path_returns.max()
            out.loc[loc, "actual_mae"] = path_returns.min()
            out.loc[loc, "actual_return"] = path_returns.iloc[-1]

        return out

    @staticmethod
    def get_labels(barrier_events: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns the barrier hit label (-1, 0, 1) based on the first touch.
        """
        events = barrier_events.copy()
        events["first_touch"] = events[["stop_loss_time", "target_time", "t1"]].min(axis=1)

        labels = pd.Series(0, index=events.index)  # Default to Time Expiry (0)

        # If stop loss is hit first
        mask_sl = events["stop_loss_time"] == events["first_touch"]
        labels[mask_sl] = -1

        # If target is hit first
        mask_target = events["target_time"] == events["first_touch"]
        labels[mask_target] = 1

        events["direction_label"] = labels
        # Use bar count (integer index positions) instead of calendar days
        # to correctly measure duration regardless of index type
        price_index = barrier_events.index
        first_touch_idx = events["first_touch"].map(
            lambda x: price_index.get_loc(x) if pd.notna(x) and x in price_index else np.nan
        )
        entry_idx = pd.Series(
            [price_index.get_loc(idx) for idx in events.index],
            index=events.index,
        )
        events["actual_duration_bars"] = first_touch_idx - entry_idx

        return events
