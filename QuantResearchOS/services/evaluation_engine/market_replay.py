from collections.abc import Generator

import pandas as pd


class MarketReplayEngine:
    """
    Simulates the exact state of the market at historical points to prevent lookahead bias.
    Streams data to the models and feature pipelines identically to how a live data feed would.
    Supports both single-asset (DataFrame) and multi-asset (dict of DataFrames) modes.
    """

    def __init__(self, historical_data: pd.DataFrame | dict[str, pd.DataFrame]):
        if isinstance(historical_data, dict):
            # Multi-asset: dict keyed by symbol
            self.multi_asset = True
            self.assets: dict[str, pd.DataFrame] = {
                symbol: df.sort_index() for symbol, df in historical_data.items()
            }
            self.data = None
        else:
            # Single-asset: plain DataFrame
            self.multi_asset = False
            self.data = historical_data.sort_index()
            self.assets = None

    def stream_candles(self, symbol: str = None) -> Generator[tuple, None, None]:
        """
        Yields one candle at a time as a named tuple (faster than iterrows).
        For multi-asset mode, provide the symbol name.
        """
        data = self._resolve_data(symbol)
        yield from data.itertuples()

    def stream_windows(
        self, window_size: int, symbol: str = None
    ) -> Generator[pd.DataFrame, None, None]:
        """Yields a rolling window of candles (useful for feature generation)."""
        data = self._resolve_data(symbol)
        for i in range(window_size, len(data) + 1):
            yield data.iloc[i - window_size : i]

    def stream_multi_asset_candles(self) -> Generator[dict[str, tuple], None, None]:
        """
        Yields synchronised candles across all assets at each timestamp.
        Only timestamps present in ALL assets are streamed.
        """
        if not self.multi_asset:
            raise ValueError(
                "stream_multi_asset_candles requires multi-asset data (dict of DataFrames)."
            )

        # Build a common index across all assets
        common_index = None
        for df in self.assets.values():
            common_index = df.index if common_index is None else common_index.intersection(df.index)

        for ts in common_index:
            snapshot = {}
            for symbol, df in self.assets.items():
                if ts in df.index:
                    snapshot[symbol] = df.loc[ts]
            yield snapshot

    def get_symbols(self):
        """Returns list of available symbols (multi-asset mode only)."""
        if not self.multi_asset:
            raise ValueError("get_symbols is only available in multi-asset mode.")
        return list(self.assets.keys())

    def _resolve_data(self, symbol: str = None) -> pd.DataFrame:
        """Resolves the correct DataFrame for the given symbol."""
        if self.multi_asset:
            if symbol is None:
                raise ValueError("Symbol must be specified in multi-asset mode.")
            if symbol not in self.assets:
                raise KeyError(
                    f"Symbol '{symbol}' not found. Available: {list(self.assets.keys())}"
                )
            return self.assets[symbol]
        return self.data
