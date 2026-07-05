import pandas as pd
import yfinance as yf


def extract_macro_features() -> dict:
    """
    Fetch macro and cross-asset features:
    - India VIX (Regime indicator)
    - USD/INR (Currency correlation for IT)
    - Global overnight (Dow Jones, Nasdaq) gap proxies
    """
    macro_tickers = {"vix": "^INDIAVIX", "usd_inr": "INR=X", "dow": "^DJI", "nasdaq": "^IXIC"}

    features = {
        "vix_level": 15.0,  # default safe assumption
        "usd_inr_chg": 0.0,
        "dow_chg": 0.0,
        "nasdaq_chg": 0.0,
        # Regime encoding MUST match RegimeClassifier and training pipeline:
        # 0 = Normal (low VIX, stable)
        # 1 = Caution (VIX 20-25)
        # 2 = Risk-Off (VIX > 25)
        "market_regime": 0,
    }

    try:
        data = yf.download(list(macro_tickers.values()), period="5d", progress=False)
        if data.empty:
            return features

        if isinstance(data.columns, pd.MultiIndex):
            # Yahoo finance returns MultiIndex when downloading multiple tickers
            close_data = data["Close"]
        else:
            close_data = data

        # VIX
        if "^INDIAVIX" in close_data.columns:
            vix_val = close_data["^INDIAVIX"].dropna().iloc[-1]
            features["vix_level"] = round(float(vix_val), 2)
            if vix_val > 25:
                features["market_regime"] = 2   # Risk-Off
            elif vix_val > 20:
                features["market_regime"] = 1   # Caution

        # USD/INR
        if "INR=X" in close_data.columns:
            inr = close_data["INR=X"].dropna()
            if len(inr) >= 2:
                features["usd_inr_chg"] = round(
                    float(((inr.iloc[-1] - inr.iloc[-2]) / inr.iloc[-2]) * 100), 3
                )

        # DOW
        if "^DJI" in close_data.columns:
            dow = close_data["^DJI"].dropna()
            if len(dow) >= 2:
                features["dow_chg"] = round(
                    float(((dow.iloc[-1] - dow.iloc[-2]) / dow.iloc[-2]) * 100), 3
                )

        # NASDAQ
        if "^IXIC" in close_data.columns:
            nasdaq = close_data["^IXIC"].dropna()
            if len(nasdaq) >= 2:
                features["nasdaq_chg"] = round(
                    float(((nasdaq.iloc[-1] - nasdaq.iloc[-2]) / nasdaq.iloc[-2]) * 100), 3
                )

    except Exception as e:
        print(f"Error fetching macro features: {e}")

    # Provide direct alias for canonical feature builder
    features["vix"] = features.get("vix_level", 15.0)
    features["nifty_pcr"] = features.get("nifty_pcr", 1.0)
    return features


def extract_historical_macro(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Extract historical macroeconomic time-series aligned to given timestamps.
    Used during model training to eradicate covariate shift.
    """
    out = pd.DataFrame(index=timestamps)
    if len(timestamps) == 0:
        out["vix"] = 15.0
        out["nifty_pcr"] = 1.0
        return out

    try:
        start_dt = pd.to_datetime(timestamps.min()).strftime("%Y-%m-%d")
        end_dt = (pd.to_datetime(timestamps.max()) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

        vix_df = yf.download("^INDIAVIX", start=start_dt, end=end_dt, progress=False)
        if not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                close = vix_df["Close"]
            elif "Close" in vix_df.columns:
                close = vix_df["Close"]
            else:
                close = vix_df
            if isinstance(close, pd.DataFrame):
                close = close[close.columns[0]]
            vix_series = close.reindex(timestamps, method="ffill").bfill()
            out["vix"] = vix_series.fillna(15.0)
        else:
            out["vix"] = 15.0
    except Exception as e:
        print(f"Notice: Could not fetch historical VIX ({e}). Falling back to baseline.")
        out["vix"] = 15.0

    out["nifty_pcr"] = 1.0  # Historical PCR baseline
    return out

