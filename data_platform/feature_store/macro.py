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
        "market_regime": 1,  # 1=Normal, 0=Caution, -1=Risk Off
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
                features["market_regime"] = -1
            elif vix_val > 20:
                features["market_regime"] = 0

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

    return features
