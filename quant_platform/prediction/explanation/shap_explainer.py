"""
Institutional SHAP Explainer & Reason Generator

Provides transparent feature attribution for every predicted alpha contract.
Eliminates black-box distrust by identifying specific technical, options, or macro drivers.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger("SHAPExplainer")

class SHAPExplainer:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def explain_prediction(self, feature_values: Dict[str, Any], model_importances: Dict[str, float], top_k: int = 3) -> List[str]:
        """
        Generates institutional human-readable reasons explaining why a signal was triggered.
        Approximates SHAP attribution by multiplying normalized feature importance by feature z-score or magnitude.
        """
        reasons = []
        scored_features = []

        for fname in self.feature_names:
            val = float(feature_values.get(fname, 0.0))
            imp = float(model_importances.get(fname, 0.0))
            
            # Calculate attribution score
            score = abs(val) * imp
            scored_features.append((fname, val, imp, score))

        # Sort by attribution score descending
        scored_features.sort(key=lambda x: x[3], reverse=True)

        for fname, val, imp, score in scored_features[:top_k]:
            if fname == "ret_1":
                direction = "Bullish" if val >= 0 else "Bearish"
                reasons.append(f"Top Driver: 1-Bar Return Momentum is {direction} ({val*100:.2f}%) | Importance: {imp*100:.1f}%")
            elif fname.startswith("ret_"):
                window = fname.split("_")[1]
                direction = "Breakout" if val >= 0 else "Breakdown"
                reasons.append(f"Trend Confluence: {window}-Bar Rolling {direction} ({val*100:.2f}%) | Importance: {imp*100:.1f}%")
            elif fname.startswith("vol_ratio_"):
                window = fname.split("_")[2]
                reasons.append(f"Volume Anomaly: Volume is {val:.1f}x normal {window}-bar average | Importance: {imp*100:.1f}%")
            elif fname == "flow_proxy":
                flow_dir = "Inflow" if val >= 0 else "Outflow"
                reasons.append(f"Institutional Flow: Net signed {flow_dir} detected ({val:.1f} units) | Importance: {imp*100:.1f}%")
            elif fname == "hl_spread":
                reasons.append(f"Microstructure: High-Low Volatility Spread is {val*100:.2f}% | Importance: {imp*100:.1f}%")
            else:
                reasons.append(f"Feature Attribution: {fname} = {val:.4f} (Weight: {imp*100:.1f}%)")

        if not reasons:
            reasons.append("Baseline Model Prior (No strong individual feature deviation)")

        return reasons

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    explainer = SHAPExplainer(["ret_1", "vol_ratio_3", "flow_proxy"])
    sample_bar = {"ret_1": 0.012, "vol_ratio_3": 2.4, "flow_proxy": 500.0}
    importances = {"ret_1": 0.45, "vol_ratio_3": 0.35, "flow_proxy": 0.20}
    explanations = explainer.explain_prediction(sample_bar, importances)
    for exp in explanations:
        print(" ->", exp)
