"""
Hidden Markov Model (HMM) for Micro-Regime Detection.
Detects state transitions (e.g., trending, mean-reverting, chaotic) on a tick-by-tick basis.
"""

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class MicroRegimeHMM:
    def __init__(self, n_components: int = 3):
        """
        n_components represents the number of hidden market states.
        For example: 0 = Low Vol Mean Reverting, 1 = High Vol Trending, 2 = Flash Crash/Shock
        """
        self.n_components = n_components

        # Neutral defaults until a real model is fitted.
        self.transition_matrix = np.ones((n_components, n_components)) / n_components
        self.emission_means = np.zeros(n_components)

        self.current_state = 0

        # Auto-fit/load on initialization
        self.load_or_fit()

    def load_or_fit(self) -> None:
        """Load HMM parameters from disk if present, otherwise auto-fit on 60 days of DuckDB history."""
        import pickle
        from pathlib import Path

        weights_path = Path(__file__).parent.parent / "hmm_regime.pkl"

        def verify_checksum(filepath: Path) -> bool:
            import hashlib

            checksum_path = filepath.with_suffix(filepath.suffix + ".sha256")
            if not checksum_path.exists():
                return False
            try:
                with open(filepath, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                with open(checksum_path) as f:
                    expected_hash = f.read().strip()
                return file_hash == expected_hash
            except Exception:
                return False

        def save_checksum(filepath: Path) -> None:
            import hashlib

            try:
                with open(filepath, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                checksum_path = filepath.with_suffix(filepath.suffix + ".sha256")
                with open(checksum_path, "w") as f:
                    f.write(file_hash)
            except Exception as e:
                logger.error(f"Failed to save checksum for {filepath}: {e}")

        if weights_path.exists():
            if not verify_checksum(weights_path):
                logger.warning("HMM weights checksum validation failed! Re-fitting model.")
            else:
                try:
                    with open(weights_path, "rb") as f:
                        data = pickle.load(f)
                        self.transition_matrix = data["transmat"]
                        self.emission_means = data["means"]
                        from hmmlearn.hmm import GaussianHMM

                        self.model = GaussianHMM(
                            n_components=self.n_components, covariance_type="diag"
                        )
                        self.model.transmat_ = self.transition_matrix
                        self.model.means_ = data["means_raw"]
                        self.model.covars_ = data["covars"]
                        self.model.startprob_ = data["startprob"]
                    logger.info("Successfully loaded HMM weights from disk with valid checksum.")
                    return
                except Exception as e:
                    logger.error(f"Failed to load HMM weights, fitting fresh: {e}")

        # Fetch historical data from DuckDB. Do not fit on synthetic data.
        logger.info("Fitting HMM from scratch on 60 days of historical data...")
        try:
            import duckdb

            from config.settings import DB_PATH

            if not DB_PATH.exists():
                logger.warning(f"DuckDB path {DB_PATH} does not exist. HMM remains neutral.")
                return
            else:
                conn = duckdb.connect(str(DB_PATH))
                df = conn.execute("""
                    SELECT close, volume FROM equity_history
                    WHERE symbol = 'NIFTY' OR symbol = 'RELIANCE'
                    ORDER BY date DESC LIMIT 200
                """).fetchdf()
                conn.close()

                if df.empty or len(df) < 60:
                    logger.warning(
                        "Historical data is insufficient for HMM fit. HMM remains neutral."
                    )
                    return
                else:
                    df = df.iloc[::-1].reset_index(drop=True)
                    if not {"spread", "ofi", "volume"}.issubset(df.columns):
                        logger.warning(
                            "HMM training data lacks spread/ofi/volume. HMM remains neutral."
                        )
                        return
                    features = df[["spread", "ofi", "volume"]].values

            from hmmlearn.hmm import GaussianHMM

            self.model = GaussianHMM(
                n_components=self.n_components, covariance_type="diag", n_iter=100
            )
            self.model.fit(features)
            self.transition_matrix = self.model.transmat_
            self.emission_means = self.model.means_.flatten()

            with open(weights_path, "wb") as f:
                pickle.dump(
                    {
                        "transmat": self.transition_matrix,
                        "means": self.emission_means,
                        "means_raw": self.model.means_,
                        "covars": self.model.covars_,
                        "startprob": self.model.startprob_,
                    },
                    f,
                )
            save_checksum(weights_path)
            logger.info("HMM fitted and saved to disk with checksum.")
        except Exception as e:
            logger.critical(f"Failed to fit HMM: {e}")

    def fit(self, historical_features: np.ndarray):
        """
        Fits the HMM using Baum-Welch (EM algorithm).
        """
        try:
            from hmmlearn.hmm import GaussianHMM

            self.model = GaussianHMM(
                n_components=self.n_components, covariance_type="diag", n_iter=100
            )
            self.model.fit(historical_features)
            self.transition_matrix = self.model.transmat_
            self.emission_means = self.model.means_.flatten()
            logger.info("HMM fitted successfully.")
        except ImportError:
            logger.warning("hmmlearn not installed. Using static HMM stub.")

    def predict_regime(self, recent_ticks: np.ndarray) -> int:
        """
        Uses the Viterbi algorithm to determine the most likely current hidden state.
        recent_ticks: array of recent features (e.g., [spread, order_imbalance, volume])
        """
        if hasattr(self, "model"):
            states = self.model.predict(recent_ticks)
            self.current_state = states[-1]
            return self.current_state

        self.current_state = 0
        return self.current_state
