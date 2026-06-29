import numpy as np
import pandas as pd


class BaseLSTM:
    """
    Layer 2: Base LSTM Model (Placeholder for MVP)
    Intended to be implemented using PyTorch to capture sequential time-series patterns
    from sequences of 30-minute bars over the past 5 sessions.

    Since training an LSTM requires gigabytes of tick data and a GPU,
    this class simply outputs 0.5 (neutral probability) until the real dataset is mounted.
    """

    def __init__(self):
        self.is_trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, feature_cols: list[str]):
        # PyTorch DataLoaders and nn.LSTM training loop goes here
        self.is_trained = True
        pass

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        # Return a neutral 50% probability array matching the length of X_test
        return np.full(len(X_test), 0.5)
