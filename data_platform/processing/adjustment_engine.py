import pandas as pd

from utils.clickhouse_client import get_clickhouse_client
from utils.logger import get_logger

logger = get_logger("adjustment_engine")


class CorporateActionsAdjustmentEngine:
    """
    Applies backward adjustments to OHLCV data to neutralize
    the effects of stock splits and bonuses.
    """

    def __init__(self):
        self.ch_client = get_clickhouse_client()

    def generate_adjusted_prices(self, symbol: str) -> None:
        """
        Calculates backwards-adjusted prices for a symbol and writes to adjusted_equity_history.
        """
        logger.info(f"Generating adjusted prices for {symbol}")

        try:
            with self.ch_client.connection() as client:
                # 1. Fetch raw prices
                prices_res = client.query(
                    f"SELECT date, open, high, low, close, volume FROM equity_history WHERE symbol = '{symbol}' ORDER BY date ASC"
                )
                if not prices_res.result_rows:
                    return

                df_prices = pd.DataFrame(
                    prices_res.result_rows,
                    columns=["date", "open", "high", "low", "close", "volume"],
                )

                # 2. Fetch multipliers
                mult_res = client.query(
                    f"SELECT ex_date, multiplier FROM events_multiplier_table WHERE symbol = '{symbol}' ORDER BY ex_date ASC"
                )
                df_mult = pd.DataFrame(mult_res.result_rows, columns=["ex_date", "multiplier"])

                # Fetch dividends to calculate dynamic dividend yield
                df_div = pd.DataFrame()
                try:
                    div_res = client.query(
                        f"SELECT ex_date, dividend_amount FROM corporate_actions WHERE symbol = '{symbol}' AND event_type = 'DIVIDEND' ORDER BY ex_date ASC"
                    )
                    if div_res.result_rows:
                        df_div = pd.DataFrame(
                            div_res.result_rows, columns=["ex_date", "dividend_amount"]
                        )
                except Exception as ex:
                    logger.warning(f"Could not fetch dividends from corporate_actions: {str(ex)}")

                df_prices["adj_factor"] = 1.0

                # 3. Apply backward adjustments
                if not df_mult.empty:
                    # Sort descending to apply backward
                    df_mult = df_mult.sort_values(by="ex_date", ascending=False)
                    for _, row in df_mult.iterrows():
                        ex_date = row["ex_date"]
                        mult = row["multiplier"]
                        # Multiply all prices BEFORE the ex_date by the multiplier
                        mask = df_prices["date"] < ex_date
                        df_prices.loc[mask, "adj_factor"] *= mult

                # 4. Calculate adjusted columns
                df_prices["adj_open"] = df_prices["open"] * df_prices["adj_factor"]
                df_prices["adj_high"] = df_prices["high"] * df_prices["adj_factor"]
                df_prices["adj_low"] = df_prices["low"] * df_prices["adj_factor"]
                df_prices["adj_close"] = df_prices["close"] * df_prices["adj_factor"]
                df_prices["adj_volume"] = (
                    df_prices["volume"] / df_prices["adj_factor"]
                )  # Volume scales inversely

                # Calculate dynamic dividend yield on ex-dividend dates
                df_prices["dividend_yield"] = 0.0
                if not df_div.empty:
                    df_prices["date_dt"] = pd.to_datetime(df_prices["date"]).dt.date
                    df_div["ex_date_dt"] = pd.to_datetime(df_div["ex_date"]).dt.date

                    for _, row in df_div.iterrows():
                        ex_date = row["ex_date_dt"]
                        div_amt = row["dividend_amount"]
                        if div_amt > 0:
                            mask = df_prices["date_dt"] == ex_date
                            if mask.any():
                                close_val = df_prices.loc[mask, "close"].values[0]
                                if close_val > 0:
                                    df_prices.loc[mask, "dividend_yield"] = float(
                                        div_amt / close_val
                                    )
                    df_prices = df_prices.drop(columns=["date_dt"])

                # 5. Insert back to ClickHouse
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
                logger.info(f"Successfully processed {len(df_insert)} adjusted rows for {symbol}")

        except Exception as e:
            logger.error(f"Failed to generate adjusted prices for {symbol}: {str(e)}")
