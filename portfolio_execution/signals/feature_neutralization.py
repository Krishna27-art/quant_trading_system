"""
Feature Neutralization

Neutralizes alpha signals against common risk factors (Market Beta, Size, Sector)
to ensure signals represent pure, uncorrelated alpha. Uses Vectorized Linear Least Squares (LSTSQ),
Ridge regression, and PCA for multicollinearity detection and handling.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureNeutralizer:
    """
    Orthogonalizes signals against specified risk factors using cross-sectional regression.
    """

    @staticmethod
    def neutralize_beta(
        signal: pd.Series, market_beta: pd.Series, weights: pd.Series | None = None
    ) -> pd.Series:
        """
        Remove market beta exposure from a signal.
        """
        df = pd.DataFrame({"signal": signal, "beta": market_beta}).dropna()
        if df.empty:
            return signal

        return FeatureNeutralizer._regress_out(df["signal"], df[["beta"]], weights=weights)

    @staticmethod
    def neutralize_size(
        signal: pd.Series, log_mcap: pd.Series, weights: pd.Series | None = None
    ) -> pd.Series:
        """
        Remove size factor exposure (log market cap) from a signal.
        """
        df = pd.DataFrame({"signal": signal, "size": log_mcap}).dropna()
        if df.empty:
            return signal

        return FeatureNeutralizer._regress_out(df["signal"], df[["size"]], weights=weights)

    @staticmethod
    def neutralize_sector(signal: pd.Series, sector_map: pd.Series) -> pd.Series:
        """
        Remove sector-specific biases by demeaning within each sector.
        """
        df = pd.DataFrame({"signal": signal, "sector": sector_map}).dropna()
        if df.empty:
            return signal

        neutralized = df["signal"] - df.groupby("sector")["signal"].transform("mean")
        return neutralized

    @staticmethod
    def neutralize_all(
        signal: pd.Series,
        factors_df: pd.DataFrame,
        weights: pd.Series | None = None,
        use_ridge: bool = False,
        ridge_alpha: float = 1.0,
    ) -> pd.Series:
        """
        Remove exposure to multiple continuous factors via multiple regression.
        Uses Ridge regression if use_ridge is True to handle multicollinearity.
        """
        df = pd.concat([signal, factors_df], axis=1).dropna()
        if df.empty:
            return signal

        sig_col = df.columns[0]
        factor_cols = df.columns[1:]

        if use_ridge:
            return FeatureNeutralizer._regress_out_ridge(
                df[sig_col], df[factor_cols], alpha=ridge_alpha, weights=weights
            )
        return FeatureNeutralizer._regress_out(df[sig_col], df[factor_cols], weights=weights)

    @staticmethod
    def _regress_out(y: pd.Series, X: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
        """
        Core regression function using np.linalg.lstsq. Returns the residuals of y ~ X.
        """
        X_mat = np.column_stack((np.ones(len(X)), X.values))
        y_vec = y.values

        if weights is not None:
            w = weights.reindex(y.index).fillna(1.0).values
            W_sqrt = np.sqrt(w)
            X_mat_w = X_mat * W_sqrt[:, np.newaxis]
            y_vec_w = y_vec * W_sqrt
            coef, _, _, _ = np.linalg.lstsq(X_mat_w, y_vec_w, rcond=None)
        else:
            coef, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)

        pred = X_mat @ coef
        return pd.Series(y.values - pred, index=y.index)

    @staticmethod
    def _regress_out_ridge(
        y: pd.Series, X: pd.DataFrame, alpha: float = 1.0, weights: pd.Series | None = None
    ) -> pd.Series:
        """
        Regression using Ridge (L2 penalty) for handling collinear features safely.
        """
        if weights is not None:
            w = weights.reindex(y.index).fillna(1.0).values
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X.values, y.values, sample_weight=w)
        else:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X.values, y.values)

        pred = model.predict(X.values)
        return pd.Series(y.values - pred, index=y.index)

    @staticmethod
    def detect_multicollinearity(features_df: pd.DataFrame) -> pd.Series:
        """
        Calculate Variance Inflation Factor (VIF) for all features to detect multicollinearity.
        Uses vectorized inverse correlation matrix. VIF > 5-10 indicates high multicollinearity.
        """
        df = features_df.dropna()
        if df.empty or df.shape[1] < 2:
            return pd.Series(dtype=float)

        X_scaled = (df - df.mean()) / df.std()
        corr_matrix = np.corrcoef(X_scaled.values, rowvar=False)
        corr_matrix += np.eye(corr_matrix.shape[0]) * 1e-6  # ridge to prevent singular matrix

        try:
            inv_corr = np.linalg.inv(corr_matrix)
            vifs = np.diag(inv_corr)
        except np.linalg.LinAlgError:
            vifs = np.full(X_scaled.shape[1], np.inf)

        return pd.Series(vifs, index=df.columns)

    @staticmethod
    def remove_collinear_features(
        features_df: pd.DataFrame, vif_threshold: float = 5.0
    ) -> pd.DataFrame:
        """
        Iteratively removes features with VIF > threshold.
        """
        df = features_df.copy()

        while True:
            vifs = FeatureNeutralizer.detect_multicollinearity(df)
            if vifs.empty or vifs.max() <= vif_threshold:
                break

            max_vif_feat = vifs.idxmax()
            logger.info(f"Removing collinear feature {max_vif_feat} (VIF={vifs.max():.2f})")
            df = df.drop(columns=[max_vif_feat])

            if df.shape[1] < 2:
                break

        return df

    @staticmethod
    def reduce_multicollinearity_pca(
        features_df: pd.DataFrame, variance_threshold: float = 0.95, fit_end_date=None
    ) -> pd.DataFrame:
        """
        Use PCA to reduce features and eliminate multicollinearity,
        keeping components that explain variance_threshold proportion of variance.
        """
        df = features_df.dropna()
        if df.empty or df.shape[1] < 2:
            return features_df

        scaler = StandardScaler()
        pca = PCA(n_components=variance_threshold)

        if fit_end_date:
            train = df.loc[:fit_end_date]
        else:
            n = len(df)
            train = df.iloc[: int(n * 0.6)]

        if train.empty:
            train = df

        scaler.fit(train)
        pca.fit(scaler.transform(train))

        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)

        pca_cols = [f"PCA_{i + 1}" for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, index=df.index, columns=pca_cols)
