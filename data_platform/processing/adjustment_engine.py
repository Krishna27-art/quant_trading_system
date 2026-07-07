import pandas as pd

from utils.clickhouse_client import get_clickhouse_client
from utils.logger import get_logger

logger = get_logger("adjustment_engine")

# Mapping of corporate action types that affect price continuity.
# SPLIT and BONUS both require backward price adjustment; DIVIDEND does not
# adjust the price series (only the dividend_yield column is populated).
_PRICE_ADJUSTMENT_ACTIONS = {"SPLIT", "BONUS"}


def _compute_multiplier(action_type: str, ratio: float | None) -> float:
    """
    Convert a parsed corporate-action row into a backward-adjustment multiplier.

    For a stock split/bonus of N:M (ratio stored as N/M):
      - Post-event shares increase by factor N/M.
      - Pre-event prices must be divided by N/M (i.e. multiplied by M/N)
        to make the series continuous.

    Examples:
      2:1 split  -> ratio=2.0  -> multiplier = 1/2.0 = 0.5
      3:2 split  -> ratio=1.5  -> multiplier = 1/1.5 = 0.667
      1:1 bonus  -> ratio=1.0  -> multiplier = 1/2.0 = 0.5  (shares double)
    """
    if ratio is None or ratio <= 0:
        logger.warning(f"Invalid ratio {ratio!r} for {action_type}; skipping adjustment")
        return 1.0
    if action_type == "SPLIT":
        # ratio = new_shares / old_shares
        return 1.0 / ratio
    if action_type == "BONUS":
        # ratio = bonus_shares_issued / existing_shares
        # Total shares become (1 + ratio) × old, so price scales by 1/(1+ratio)
        return 1.0 / (1.0 + ratio)
    return 1.0


class CorporateActionsAdjustmentEngine:
    """
    Applies backward adjustments to OHLCV data to neutralise the effects of
    stock splits and bonus issues.

    Data flow
    ---------
    corporate_actions table  (populated by CorporateActionsPipeline)
        └─> compute backward multipliers inline (this class)
            └─> write adjusted_equity_history  (read by CanonicalFeatureBuilder)

    Previously the engine read from ``events_multiplier_table``, a table that
    nothing in the codebase ever wrote to.  Every symbol therefore received an
    adj_factor of 1.0 and features were silently computed on raw prices.
    """

    def __init__(self):
        self.ch_client = get_clickhouse_client()

    def generate_adjusted_prices(self, symbol: str) -> None:
        """
        Calculates backwards-adjusted prices for a symbol and writes to
        adjusted_equity_history.
        """
        logger.info(f"Generating adjusted prices for {symbol}")

        try:
            with self.ch_client.connection() as client:
                # 1. Fetch raw prices
                prices_res = client.query(
                    f"SELECT date, open, high, low, close, volume "
                    f"FROM equity_history WHERE symbol = '{symbol}' ORDER BY date ASC"
                )
                if not prices_res.result_rows:
                    logger.info(f"No price rows found for {symbol}; skipping.")
                    return

                df_prices = pd.DataFrame(
                    prices_res.result_rows,
                    columns=["date", "open", "high", "low", "close", "volume"],
                )

                # 2. Build multipliers directly from corporate_actions.
                #    FIX: previously queried events_multiplier_table which was
                #    never written to, so adj_factor was always 1.0.
                ca_res = client.query(
                    f"SELECT ex_date, action_type, ratio "
                    f"FROM corporate_actions "
                    f"WHERE symbol = '{symbol}' "
                    f"  AND action_type IN ('SPLIT', 'BONUS') "
                    f"ORDER BY ex_date ASC"
                )
                df_ca = pd.DataFrame(
                    ca_res.result_rows if ca_res.result_rows else [],
                    columns=["ex_date", "action_type", "ratio"],
                )

                # Compute the per-event backward multiplier
                if not df_ca.empty:
                    df_ca["multiplier"] = df_ca.apply(
                        lambda r: _compute_multiplier(r["action_type"], r["ratio"]),
                        axis=1,
                    )
                    logger.info(
                        f"{len(df_ca)} corporate action(s) found for {symbol}: "
                        + ", ".join(
                            f"{r.action_type}@{r.ex_date}(x{r.multiplier:.4f})"
                            for r in df_ca.itertuples()
                        )
                    )
                else:
                    logger.info(f"No split/bonus events found for {symbol}; adj_factor will be 1.0.")

                # 3. Fetch dividends for dividend_yield column
                df_div = pd.DataFrame()
                try:
                    div_res = client.query(
                        f"SELECT ex_date, ratio AS dividend_amount "
                        f"FROM corporate_actions "
                        f"WHERE symbol = '{symbol}' AND action_type = 'DIVIDEND' "
                        f"ORDER BY ex_date ASC"
                    )
                    if div_res.result_rows:
                        df_div = pd.DataFrame(
                            div_res.result_rows, columns=["ex_date", "dividend_amount"]
                        )
                except Exception as ex:
                    logger.warning(f"Could not fetch dividends for {symbol}: {ex}")

                # 4. Apply backward adjustments (iterate newest event first)
                df_prices["adj_factor"] = 1.0
                if not df_ca.empty:
                    df_mult = df_ca[["ex_date", "multiplier"]].sort_values(
                        "ex_date", ascending=False
                    )
                    for _, row in df_mult.iterrows():
                        ex_date = row["ex_date"]
                        mult = row["multiplier"]
                        # All bars BEFORE the ex-date are adjusted backward
                        mask = df_prices["date"] < ex_date
                        df_prices.loc[mask, "adj_factor"] *= mult

                # 5. Calculate adjusted columns
                df_prices["adj_open"]   = df_prices["open"]   * df_prices["adj_factor"]
                df_prices["adj_high"]   = df_prices["high"]   * df_prices["adj_factor"]
                df_prices["adj_low"]    = df_prices["low"]    * df_prices["adj_factor"]
                df_prices["adj_close"]  = df_prices["close"]  * df_prices["adj_factor"]
                df_prices["adj_volume"] = df_prices["volume"] / df_prices["adj_factor"]  # inverse

                # 6. Dividend yield on ex-date bars
                df_prices["dividend_yield"] = 0.0
                if not df_div.empty:
                    df_prices["date_dt"] = pd.to_datetime(df_prices["date"]).dt.date
                    df_div["ex_date_dt"] = pd.to_datetime(df_div["ex_date"]).dt.date

                    for _, row in df_div.iterrows():
                        ex_date = row["ex_date_dt"]
                        div_amt = row["dividend_amount"]
                        if div_amt and div_amt > 0:
                            mask = df_prices["date_dt"] == ex_date
                            if mask.any():
                                close_val = df_prices.loc[mask, "close"].values[0]
                                if close_val > 0:
                                    df_prices.loc[mask, "dividend_yield"] = float(
                                        div_amt / close_val
                                    )
                    df_prices = df_prices.drop(columns=["date_dt"])

                # 7. Write to ClickHouse
                df_insert = df_prices[
                    [
                        "date",
                        "adj_open",
                        "adj_high",
                        "adj_low",
                        "adj_close",
                        "adj_volume",
                        "adj_factor",
                        "dividend_yield",
                    ]
                ].copy()
                df_insert["symbol"] = symbol

                client.insert_df("adjusted_equity_history", df_insert)
                logger.info(
                    f"Successfully written {len(df_insert)} adjusted rows for {symbol}"
                )

        except Exception as e:
            logger.error(f"Failed to generate adjusted prices for {symbol}: {e}")

