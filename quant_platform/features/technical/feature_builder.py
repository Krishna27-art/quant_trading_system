"""
Institutional Technical & Quantitative Feature Builder

Transforms raw OHLCV data into stationary, point-in-time features.
Prevents look-ahead bias by strict lagging and recursive rolling windows.
"""

import math
import logging
from typing import List, Dict, Any

logger = logging.getLogger("FeatureBuilder")

class FeatureBuilder:
    def __init__(self, windows: List[int] = [5, 15, 60]):
        self.windows = windows

    def build_features(self, ohlcv_series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Accepts chronological list of dicts with keys: symbol, timestamp, open, high, low, close, volume.
        Returns list of feature dictionaries with computed indicators.
        """
        if not ohlcv_series:
            return []

        # Sort chronologically by timestamp just in case
        sorted_series = sorted(ohlcv_series, key=lambda x: x["timestamp"])
        features_list = []

        # Maintain running history per symbol
        symbol_history: Dict[str, List[float]] = {}
        symbol_volumes: Dict[str, List[float]] = {}

        for bar in sorted_series:
            sym = bar["symbol"]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            vol = float(bar.get("volume", 0.0))

            if sym not in symbol_history:
                symbol_history[sym] = []
                symbol_volumes[sym] = []

            hist = symbol_history[sym]
            v_hist = symbol_volumes[sym]

            # Compute features using strictly prior and current bar data
            feat = {
                "symbol": sym,
                "timestamp": bar["timestamp"],
                "close": close,
                "high": high,
                "low": low,
                "volume": vol,
            }

            # 1. Log return from previous bar
            if len(hist) >= 1:
                feat["ret_1"] = math.log(close / hist[-1]) if hist[-1] > 0 else 0.0
            else:
                feat["ret_1"] = 0.0

            # 2. Rolling Momentum & Volatility across windows
            for w in self.windows:
                if len(hist) >= w:
                    window_slice = hist[-w:]
                    # Momentum (Log return over window)
                    feat[f"ret_{w}"] = math.log(close / window_slice[0]) if window_slice[0] > 0 else 0.0
                    
                    # Realized Volatility (std dev of returns)
                    mean_val = sum(window_slice) / len(window_slice)
                    variance = sum((x - mean_val) ** 2 for x in window_slice) / len(window_slice)
                    feat[f"vol_{w}"] = math.sqrt(variance) / mean_val if mean_val > 0 else 0.0
                    
                    # Volume relative to rolling average
                    v_slice = v_hist[-w:]
                    mean_vol = sum(v_slice) / len(v_slice) if v_slice else 1.0
                    feat[f"vol_ratio_{w}"] = (vol / mean_vol) if mean_vol > 0 else 1.0
                else:
                    feat[f"ret_{w}"] = feat["ret_1"]
                    feat[f"vol_{w}"] = 0.01  # baseline noise
                    feat[f"vol_ratio_{w}"] = 1.0

            # 3. Microstructure spread approximation (High-Low range normalized)
            feat["hl_spread"] = (high - low) / close if close > 0 else 0.0

            # 4. Institutional Flow Proxy (Volume * signed return)
            feat["flow_proxy"] = feat["ret_1"] * vol

            features_list.append(feat)

            # Update historical buffers
            hist.append(close)
            v_hist.append(vol)

        logger.debug(f"Computed features for {len(features_list)} bars across {len(symbol_history)} symbols.")
        return features_list

    def get_feature_names(self) -> List[str]:
        names = ["ret_1", "hl_spread", "flow_proxy"]
        for w in self.windows:
            names.extend([f"ret_{w}", f"vol_{w}", f"vol_ratio_{w}"])
        return names

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = FeatureBuilder(windows=[3, 5])
    sample_data = [
        {"symbol": "RELIANCE", "timestamp": "2026-07-01T09:15:00Z", "open": 2400, "high": 2410, "low": 2395, "close": 2405, "volume": 10000},
        {"symbol": "RELIANCE", "timestamp": "2026-07-01T09:30:00Z", "open": 2405, "high": 2425, "low": 2402, "close": 2420, "volume": 15000},
        {"symbol": "RELIANCE", "timestamp": "2026-07-01T09:45:00Z", "open": 2420, "high": 2422, "low": 2410, "close": 2415, "volume": 12000},
        {"symbol": "RELIANCE", "timestamp": "2026-07-01T10:00:00Z", "open": 2415, "high": 2440, "low": 2412, "close": 2435, "volume": 25000},
    ]
    feats = builder.build_features(sample_data)
    for f in feats:
        print(f"[{f['timestamp']}] Close: {f['close']} | ret_1: {f['ret_1']:.4f} | ret_3: {f['ret_3']:.4f} | vol_ratio_3: {f['vol_ratio_3']:.2f}")
