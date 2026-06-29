import pandas as pd
from sklearn.mixture import GaussianMixture


class RegimeClassifier:
    """
    Classifies the market into 4 regimes using unsupervised clustering (GMM)
    or strict rules based on Macro features.

    Regimes:
    0: Trending Up (Low Volatility, Positive Momentum)
    1: Trending Down (High Volatility, Negative Momentum)
    2: High Volatility Choppy (High VIX, Mean Reverting)
    3: Low Volatility Range-Bound (Low VIX, Flat Momentum)
    """

    def __init__(self):
        # We use a GMM for unsupervised clustering on historical VIX and Momentum
        self.gmm = GaussianMixture(n_components=4, random_state=42)
        self.is_trained = False

    def train(self, features_df: pd.DataFrame):
        """
        Expects a DataFrame with 'vix' and 'nifty_momentum' columns.
        """
        if "vix" not in features_df.columns or "nifty_momentum" not in features_df.columns:
            raise ValueError("Regime classifier requires 'vix' and 'nifty_momentum' features.")

        X = features_df[["vix", "nifty_momentum"]].fillna(method="ffill").dropna()
        if len(X) > 10:
            self.gmm.fit(X)
            self.is_trained = True

    def predict(self, current_vix: float, current_momentum: float) -> int:
        """
        Predict the regime for today.
        If not trained on historical data, falls back to rule-based regime classification.
        """
        if self.is_trained:
            X_pred = pd.DataFrame({"vix": [current_vix], "nifty_momentum": [current_momentum]})
            return int(self.gmm.predict(X_pred)[0])

        # Fallback Rule-Based Classification (useful for MVP without historical data)
        if current_vix > 20.0:
            if current_momentum < -0.01:
                return 1  # Trending Down (Panic)
            else:
                return 2  # High Vol Choppy
        else:
            if current_momentum > 0.005:
                return 0  # Trending Up
            else:
                return 3  # Low Vol Range-Bound
