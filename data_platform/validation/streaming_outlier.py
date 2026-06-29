class StreamingOutlierValidator:
    def __init__(self, max_move_pct=0.08, corp_action_threshold=0.18, zero_vol_grace_seconds=5):
        self.max_move = max_move_pct
        self.ca_thresh = corp_action_threshold
        self.last_good = {}  # symbol -> (price, ts, atr)
        self.ca_log = {}  # symbol -> (ts, ca_type)
        self.zero_vol_grace = zero_vol_grace_seconds

    def is_valid(self, symbol: str, price: float, volume: int, ts: float) -> tuple:
        """Returns (is_valid: bool, reason: str)"""

        # Basic sanity
        if price is None or price <= 0:
            return False, "zero_or_negative_price"

        if volume == 0:
            # Allow zero volume briefly at open
            if symbol in self.last_good:
                _, last_ts, _ = self.last_good[symbol]
                if ts - last_ts > self.zero_vol_grace:
                    return False, "zero_volume"

        if symbol not in self.last_good:
            # First tick for this symbol — accept and store
            self.last_good[symbol] = (price, ts, price * 0.005)
            return True, "first_tick"

        last_price, last_ts, atr = self.last_good[symbol]

        # Stale tick
        if ts < last_ts:
            return False, "stale_timestamp"

        pct_move = abs(price / last_price - 1)

        # Check for corporate action — detect and RESET, never brick
        if pct_move > self.ca_thresh:
            ratio = price / last_price
            ca_type = "UNKNOWN_CA"

            if 0.48 <= ratio <= 0.52:
                ca_type = "SPLIT_2FOR1"
            elif 0.32 <= ratio <= 0.35:
                ca_type = "SPLIT_3FOR1"
            elif 0.24 <= ratio <= 0.26:
                ca_type = "SPLIT_4FOR1"
            elif 1.95 <= ratio <= 2.05:
                ca_type = "BONUS_1FOR1"
            elif 1.45 <= ratio <= 1.55:
                ca_type = "BONUS_1FOR2"

            self.ca_log[symbol] = (ts, ca_type)
            # CRITICAL: Reset reference price, do NOT reject
            new_atr = abs(price - last_price) * 0.1
            self.last_good[symbol] = (price, ts, new_atr)
            return True, f"corp_action_{ca_type}"

        # Normal bad tick
        if pct_move > self.max_move:
            return False, f"excessive_move_{pct_move:.3f}"

        # Update ATR estimate
        new_atr = atr * 0.95 + abs(price - last_price) * 0.05
        self.last_good[symbol] = (price, ts, new_atr)
        return True, "ok"
