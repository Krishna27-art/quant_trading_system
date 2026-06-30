"""
PATCH: scripts/generate_live_predictions.py
============================================
Apply these changes to wire the actual trained models into the live path.
Everywhere a win_prob was hardcoded arithmetic, it now comes from the model.

Only the changed sections are shown.  Everything else in the file stays as-is.
"""

# ── 1. ADD these imports at the top (after existing imports) ────────────────

from prediction_intelligence.base_logistic import (
    ModelRegistry,
    build_features,
    INTRADAY_FEATURES,
    SWING_FEATURES,
    LONGTERM_FEATURES,
)

_registry = ModelRegistry()   # singleton — loads each model file once per process


# ── 2. REPLACE generate_intraday_predictions() ─────────────────────────────

def generate_intraday_predictions(db_session) -> list[dict]:
    """Generates INTRADAY signals based on 1-min candles."""
    logger.info("Generating INTRADAY predictions...")
    predictions = []

    macro = extract_macro_features()
    vix   = macro.get("vix_level", 15.0)

    model = _registry.get("LGBM_INTRADAY_v1", "INTRADAY")

    for sym in SYMBOLS:
        try:
            df = download_historical_candles(sym, interval="1minute")
            if df.empty:
                df = yf.download(f"{sym}.NS", period="5d", interval="1m", progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.reset_index().rename(columns={
                        "Datetime": "timestamp", "Open": "open",
                        "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                    })

            if df.empty or len(df) < 20:
                continue

            # Build features the same way the training script did
            feats = build_features(df, "INTRADAY", extra={"vix": vix})
            X_live = feats[INTRADAY_FEATURES].tail(1).reset_index(drop=True)

            if model.is_ready() and not X_live.isnull().all(axis=1).any():
                win_prob = float(model.predict_proba(X_live)[0])
            else:
                # Fallback: simple VWAP rule (same as original) until model trained
                vwap_dist = float(X_live["vwap_dist"].iloc[0]) if "vwap_dist" in X_live.columns else 0.0
                win_prob  = max(0.1, min(0.9, 0.5 + (0.05 if vwap_dist > 0 else -0.05) - (0.02 if vix > 20 else 0)))
                logger.warning(f"{sym} INTRADAY: model not ready, using rule-based fallback")

            entry     = float(df["close"].iloc[-1])
            direction = "BUY" if win_prob > 0.5 else "SELL"
            target    = entry * 1.015 if direction == "BUY" else entry * 0.985
            sl        = entry * 0.9925 if direction == "BUY" else entry * 1.0075

            predictions.append({
                "symbol":               sym,
                "timeframe":            "INTRADAY",
                "direction":            direction,
                "entry_price":          round(entry, 2),
                "stop_loss":            round(sl, 2),
                "target_price":         round(target, 2),
                "predicted_probability": round(win_prob, 2),
                "model_version":        "LGBM_INTRADAY_v1",
                "features_json":        X_live.iloc[0].to_json(),
            })

        except Exception as e:
            logger.error(f"Error generating intraday for {sym}: {e}")

    return predictions


# ── 3. REPLACE generate_swing_predictions() ────────────────────────────────

def generate_swing_predictions(db_session) -> list[dict]:
    """Generates SWING signals based on daily candles."""
    logger.info("Generating SWING predictions...")
    predictions = []

    pcr   = fetch_option_chain_pcr("Nifty 50")
    macro = extract_macro_features()
    vix   = macro.get("vix_level", 15.0)

    model = _registry.get("XGB_SWING_v1", "SWING")

    for sym in SYMBOLS:
        try:
            df = download_historical_candles(sym, interval="day")
            if df.empty:
                df = yf.download(f"{sym}.NS", period="1y", progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.reset_index().rename(columns={
                        "Date": "timestamp", "Open": "open",
                        "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                    })

            if df.empty or len(df) < 50:
                continue

            feats  = build_features(df, "SWING", extra={"vix": vix, "nifty_pcr": pcr})
            X_live = feats[SWING_FEATURES].tail(1).reset_index(drop=True)

            if model.is_ready() and not X_live.isnull().all(axis=1).any():
                win_prob = float(model.predict_proba(X_live)[0])
            else:
                z_val    = float(X_live["z_score_20d"].iloc[0]) if "z_score_20d" in X_live.columns else 0.0
                win_prob = max(0.1, min(0.9,
                    0.5
                    + (0.08 if z_val < -1.5 else -0.05 if z_val > 1.5 else 0.0)
                    + (0.03 if pcr > 1.0 else -0.02)
                ))
                logger.warning(f"{sym} SWING: model not ready, using rule-based fallback")

            entry     = float(df["close"].iloc[-1])
            direction = "BUY" if win_prob > 0.5 else "SELL"
            target    = entry * 1.03 if direction == "BUY" else entry * 0.97
            sl        = entry * 0.985 if direction == "BUY" else entry * 1.015

            predictions.append({
                "symbol":               sym,
                "timeframe":            "SWING",
                "direction":            direction,
                "entry_price":          round(entry, 2),
                "stop_loss":            round(sl, 2),
                "target_price":         round(target, 2),
                "predicted_probability": round(win_prob, 2),
                "model_version":        "XGB_SWING_v1",
                "features_json":        X_live.iloc[0].to_json(),
            })

        except Exception as e:
            logger.error(f"Error generating swing for {sym}: {e}")

    return predictions


# ── 4. REPLACE generate_longterm_predictions() ─────────────────────────────

def generate_longterm_predictions(db_session) -> list[dict]:
    """Generates LONGTERM signals based on weekly candles + fundamentals."""
    logger.info("Generating LONGTERM predictions...")
    predictions = []

    macro = extract_macro_features()
    vix   = macro.get("vix_level", 15.0)

    model = _registry.get("LOGREG_LONGTERM_v1", "LONGTERM")   # ← THE FIX

    for sym in SYMBOLS:
        try:
            df = download_historical_candles(sym, interval="week")
            if df.empty:
                df = yf.download(f"{sym}.NS", period="2y", interval="1wk", progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.reset_index().rename(columns={
                        "Date": "timestamp", "Open": "open",
                        "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume",
                    })

            if df.empty or len(df) < 10:
                continue

            # Fetch fundamentals
            pe_ratio      = 20.0
            debt_to_equity = 0.5
            try:
                ticker = yf.Ticker(f"{sym}.NS")
                info   = ticker.info
                pe_ratio       = float(info.get("forwardPE", 20.0))
                debt_to_equity = float(info.get("debtToEquity", 50.0)) / 100.0
            except Exception as fe:
                logger.warning(f"Failed to fetch yfinance financials for {sym}: {fe}")

            feats  = build_features(df, "LONGTERM", extra={
                "vix": vix, "pe_ratio": pe_ratio, "debt_to_equity": debt_to_equity,
            })
            X_live = feats[LONGTERM_FEATURES].tail(1).reset_index(drop=True)

            if model.is_ready() and not X_live.isnull().all(axis=1).any():
                win_prob = float(model.predict_proba(X_live)[0])
            else:
                win_prob = max(0.1, min(0.9,
                    0.5
                    + (0.06 if pe_ratio < 25 else -0.04)
                    + (0.04 if debt_to_equity < 0.8 else -0.05)
                ))
                logger.warning(f"{sym} LONGTERM: model not ready, using rule-based fallback")

            entry     = float(df["close"].iloc[-1])
            direction = "BUY" if win_prob > 0.5 else "SELL"
            target    = entry * 1.20 if direction == "BUY" else entry * 0.80
            sl        = entry * 0.90 if direction == "BUY" else entry * 1.10

            predictions.append({
                "symbol":               sym,
                "timeframe":            "LONGTERM",
                "direction":            direction,
                "entry_price":          round(entry, 2),
                "stop_loss":            round(sl, 2),
                "target_price":         round(target, 2),
                "predicted_probability": round(win_prob, 2),
                "model_version":        "LOGREG_LONGTERM_v1",
                "features_json":        X_live.iloc[0].to_json(),
            })

        except Exception as e:
            logger.error(f"Error generating long-term for {sym}: {e}")

    return predictions