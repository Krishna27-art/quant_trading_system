import pandas as pd


class PITImputer:
    """
    Point-in-Time (PIT) Imputer that prevents lookahead bias:
    1. Excludes assets with >30% missing values in the training window.
    2. Imputes price-derived features using forward-fill capped at 5 bars.
    3. Imputes volume features using cross-sectional median.
    4. Imputes ratio features using cross-sectional sector-level median.
    """

    def __init__(self, max_missing_pct: float = 0.3, ffill_limit: int = 5):
        self.max_missing_pct = max_missing_pct
        self.ffill_limit = ffill_limit
        self.valid_assets: list[str] = []
        self.sector_medians: dict[str, float] = {}
        self.global_medians: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, sector_map: dict[str, str] | None = None) -> "PITImputer":
        """
        Fit imputer on training data.
        df must contain columns: ['timestamp', 'asset', 'feature_type', 'value']
        or be a wide format DataFrame indexed by timestamp with MultiIndex (asset, feature) columns.
        We expect wide format for standard pipeline processing:
        Index: timestamp
        Columns: MultiIndex(asset, feature) or columns representing features.
        Let's support both wide format (with columns as features or asset-features) or standard DataFrame.
        Assuming standard wide format: columns are features, asset is a column, timestamp is index or column.
        Let's handle standard tabular panel data:
        df has columns: ['timestamp', 'asset', 'sector'] + features
        """
        features = [col for col in df.columns if col not in ["timestamp", "asset", "sector"]]

        # 1. Exclude assets with >30% missing values
        asset_missing = df.groupby("asset")[features].apply(lambda x: x.isna().mean().mean())
        self.valid_assets = asset_missing[asset_missing <= self.max_missing_pct].index.tolist()

        # Filter valid assets
        df_filtered = df[df["asset"].isin(self.valid_assets)]

        # 2. Fit global and sector-level medians on training window
        for feat in features:
            self.global_medians[feat] = df_filtered[feat].median()
            if "sector" in df_filtered.columns:
                sector_groups = df_filtered.groupby("sector")[feat].median()
                for sector, val in sector_groups.items():
                    self.sector_medians[f"{sector}_{feat}"] = val

        return self

    def transform(self, df: pd.DataFrame, sector_map: dict[str, str] | None = None) -> pd.DataFrame:
        """
        Apply point-in-time imputation to the input DataFrame.
        """
        df = df.copy()

        # Filter to valid assets during training (if fit has run)
        if self.valid_assets:
            df = df[df["asset"].isin(self.valid_assets)]

        features = [col for col in df.columns if col not in ["timestamp", "asset", "sector"]]

        # Group by asset to apply forward fill on price features and group-specific ffills
        # We classify features based on naming:
        # Price features: contains 'price', 'close', 'open', 'high', 'low', 'adj'
        # Volume features: contains 'volume', 'qty'
        # Ratio features: contains 'ratio', 'pe', 'pb', 'ps', 'margin', 'pct'

        result_dfs = []
        for _asset, group in df.groupby("asset"):
            group = (
                group.sort_values("timestamp")
                if "timestamp" in group.columns
                else group.sort_index()
            )

            for col in features:
                col_lower = col.lower()
                is_price = any(
                    p in col_lower for p in ["price", "close", "open", "high", "low", "adj"]
                )
                is_volume = any(v in col_lower for v in ["volume", "qty"])

                if is_price:
                    # Forward-fill capped at ffill_limit
                    group[col] = group[col].ffill(limit=self.ffill_limit)
                    # If still NaN, it will be dropped or filled with global median
                    group = group.dropna(subset=[col])
                elif is_volume:
                    # Fill with cross-sectional median (if available in global_medians)
                    median_val = self.global_medians.get(col, 0.0)
                    group[col] = group[col].fillna(median_val)
                else:
                    # Ratio / other feature: Sector-level median fill, then global median fill
                    sector = group["sector"].iloc[0] if "sector" in group.columns else None
                    fill_val = self.sector_medians.get(f"{sector}_{col}") if sector else None
                    if fill_val is None or pd.isna(fill_val):
                        fill_val = self.global_medians.get(col, 0.0)
                    group[col] = group[col].fillna(fill_val)

            result_dfs.append(group)

        if not result_dfs:
            return pd.DataFrame(columns=df.columns)

        return pd.concat(result_dfs).sort_index()
