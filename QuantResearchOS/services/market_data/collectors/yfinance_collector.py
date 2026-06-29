import time

import pandas as pd
import yfinance as yf

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1


def fetch_historical_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    interval: str = "1d",
    period: str = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker from Yahoo Finance.

    interval: '1d', '1wk', '1mo', '1m', '5m', '15m', '30m', '1h'
    Retries up to 3 times with 1s delay on failure.
    """
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stock = yf.Ticker(ticker)
            if period:
                df = stock.history(period=period, interval=interval)
            else:
                df = stock.history(start=start_date, end=end_date, interval=interval)
            break
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    else:
        raise RuntimeError(
            f"Failed to fetch data for {ticker} after {MAX_RETRIES} retries"
        ) from last_exception

    if df.empty:
        return pd.DataFrame()

    df.reset_index(inplace=True)
    # yfinance sometimes returns "Date" and sometimes "Datetime" depending on interval
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "timestamp"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "timestamp"}, inplace=True)

    # Standardize columns
    df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    # We only need OHLCV
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    data = fetch_historical_data("RELIANCE.NS", "2023-01-01", "2023-12-31")
    print(data.head())
