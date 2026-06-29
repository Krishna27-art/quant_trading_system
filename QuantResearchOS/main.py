"""
QuantResearchOS — Institutional Quantitative Research Platform
Main entry point and pipeline orchestrator.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("QuantResearchOS")


def run_data_pipeline():
    """Phase 1: Collect → Validate → Store market data."""
    from services.feature_store.offline_store import OfflineFeatureStore
    from services.market_data.collectors.yfinance_collector import fetch_historical_data
    from services.market_data.validation.validation_engine import ValidationEngine

    logger.info("=== DATA PIPELINE START ===")

    # 1. Collect
    symbols = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS"]
    validator = ValidationEngine()
    store = OfflineFeatureStore()

    for symbol in symbols:
        logger.info(f"Collecting {symbol}...")
        df = fetch_historical_data(symbol, period="1y")

        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol}")
            continue

        # 2. Validate
        result = validator.validate_ohlcv(df)
        logger.info(
            f"{symbol} — Quality Score: {result['score']:.1f}/100 | Issues: {result['issues']}"
        )

        if result["score"] < 50:
            logger.error(f"{symbol} REJECTED — quality too low")
            continue

        # 3. Store
        clean_symbol = symbol.replace(".NS", "")
        store.save_features(df, "equity", clean_symbol, version=1)
        logger.info(f"{symbol} saved to offline store")

    logger.info("=== DATA PIPELINE COMPLETE ===")


def run_label_pipeline():
    """Phase 2: Generate Triple Barrier labels for stored data."""
    import pandas as pd
    from services.feature_store.offline_store import OfflineFeatureStore
    from services.label_engine.triple_barrier import MultiObjectiveLabeler

    logger.info("=== LABEL PIPELINE START ===")
    symbols = ["RELIANCE", "INFY", "TCS", "HDFCBANK"]
    store = OfflineFeatureStore()

    for symbol in symbols:
        try:
            df = store.load_features("equity", symbol, version=1)
            # Ensure timestamp is datetime and index is timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            # Select candidate entry times (e.g. daily close prices)
            # Vertical barrier (expiry time) is 10 trading sessions ahead
            events = pd.DataFrame(index=df.index)
            # t1 is the index 10 steps ahead (approx 10 trading sessions)
            events["t1"] = pd.Series(df.index, index=df.index).shift(-10)
            events.dropna(subset=["t1"], inplace=True)

            # pt_sl = (1.5, 1.0), min_ret = 1%
            barriers = MultiObjectiveLabeler.get_barriers(
                df["close"], events, pt_sl=(1.5, 1.0), min_ret=0.01
            )
            labels_df = MultiObjectiveLabeler.get_labels(barriers)

            # Save labeled dataset back
            store.save_features(labels_df, "labels", symbol, version=1)
            logger.info(f"Generated and saved triple barrier labels for {symbol}")
        except Exception as e:
            logger.error(f"Failed to generate labels for {symbol}: {e}")

    logger.info("=== LABEL PIPELINE COMPLETE ===")


def run_training_pipeline():
    """Phase 3: Train base models and meta-ensemble."""
    import numpy as np
    import pandas as pd
    from ml.models.tree.base_trainer import BaseTreeTrainer
    from services.feature_store.offline_store import OfflineFeatureStore

    logger.info("=== TRAINING PIPELINE START ===")
    symbols = ["RELIANCE", "INFY", "TCS", "HDFCBANK"]
    store = OfflineFeatureStore()

    for symbol in symbols:
        try:
            # Load features and labels
            features_df = store.load_features("equity", symbol, version=1)
            labels_df = store.load_features("labels", symbol, version=1)

            # Align features and labels by timestamp
            features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
            features_df.set_index("timestamp", inplace=True)

            # Join features and labels
            # Simple feature: 5-day SMA and 10-day SMA, daily return
            features_df["sma_5"] = features_df["close"].rolling(5).mean()
            features_df["sma_10"] = features_df["close"].rolling(10).mean()
            features_df["daily_return"] = features_df["close"].pct_change()

            # Combine
            dataset = features_df.join(labels_df[["direction_label", "actual_return"]]).dropna()

            if dataset.empty:
                logger.warning(f"No aligned dataset for {symbol}")
                continue

            X = dataset[["sma_5", "sma_10", "daily_return"]]
            y = dataset["direction_label"]

            # Train classification model
            trainer = BaseTreeTrainer(model_type="xgboost")
            # Map labels to consecutive integers dynamically
            unique_classes = sorted(y.unique())
            class_mapping = {val: idx for idx, val in enumerate(unique_classes)}
            y_encoded = y.map(class_mapping)

            trainer.train(
                X,
                y_encoded,
                params={"max_depth": 3, "n_estimators": 50, "eval_metric": "mlogloss"},
                task="classification",
            )

            # Predict
            preds = trainer.predict(X)
            logger.info(
                f"Successfully trained tree model for {symbol}. Predictions range: {np.unique(preds)}"
            )
        except Exception as e:
            logger.error(f"Failed to run training pipeline for {symbol}: {e}")

    logger.info("=== TRAINING PIPELINE COMPLETE ===")


def main():
    """Main orchestrator."""
    logger.info("QuantResearchOS v0.1.0 — Starting...")

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "data":
            run_data_pipeline()
        elif command == "labels":
            run_label_pipeline()
        elif command == "train":
            run_training_pipeline()
        elif command == "dashboard":
            import subprocess

            subprocess.run(["streamlit", "run", "apps/dashboard/app.py"])
        elif command == "status":
            logger.info("System Status:")
            logger.info("  Data Pipeline:     READY")
            logger.info("  Label Pipeline:    SKELETON")
            logger.info("  Training Pipeline: SKELETON")
            logger.info("  Dashboard:         READY (dummy data)")
        else:
            logger.error(f"Unknown command: {command}")
            print_usage()
    else:
        print_usage()


def print_usage():
    print("""
Usage: python main.py <command>

Commands:
  data        Run data collection pipeline (yfinance → validate → store)
  labels      Run label generation pipeline (Triple Barrier)
  train       Run model training pipeline
  dashboard   Launch Streamlit research dashboard
  status      Show system status
    """)


if __name__ == "__main__":
    main()
