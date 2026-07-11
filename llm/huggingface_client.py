import json
import os
import urllib.request
import urllib.error
import asyncio
from llm.base import BaseLLMClient
from utils.logger import get_logger

logger = get_logger("huggingface_client")


class HuggingFaceClient(BaseLLMClient):
    """
    Hugging Face Inference API Client for models like Qwen/Qwen3-32B.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-32B"):
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_ACCESS_TOCKEN") or "MOCK_KEY_FOR_TESTING"
        self.model_id = model_id
        self.api_url = "https://router.huggingface.co/hf-inference/v1/chat/completions"

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        env = os.getenv("ENV", "LOCAL")
        if env.upper() in ("LIVE", "PAPER"):
            raise RuntimeError("Fatal: LLM mock response triggered in live/paper environment!")
            
        if "losing_factors" in system_prompt or "post-mortem" in system_prompt.lower():
            return json.dumps({
                "losing_factors": [
                    "High VIX (>20) intraday spikes causing wide ATR swings",
                    "Large entry VWAP distance (>0.8%) causing mean reversion drag"
                ],
                "winning_factors": [
                    "Strong 5-minute price momentum matching the daily trend direction",
                    "High volume confirmation ratio (>1.5x of 20-period average)"
                ],
                "analysis": "Mock analysis of winners and losers.",
                "actionable_warnings": [
                    "High stop-loss rate observed for intraday momentum strategies in high VIX (>20) regimes."
                ],
                "suggested_threshold_adjustments": "Recommend increasing the confidence threshold constraint from 0.55 to 0.62 for INTRADAY trades."
            })
        elif "Pre-Market" in system_prompt or "Indian equities day trading analyst" in system_prompt or "institutional quantitative strategist" in system_prompt:
            return json.dumps({
                "market_regime": "Bullish Trend",
                "risk_level": "Moderate",
                "confidence": 0.82,
                "sector_rotation": ["IT", "FINANCIALS"],
                "top_themes": ["Earnings Beats", "FII Inflows"],
                "watchlist": ["TCS", "RELIANCE"],
                "warnings": ["RBI Meeting volatility", "High VIX intraday spikes"]
            })
        elif "Rate the following recent news headlines" in system_prompt:
            return json.dumps(
                {"score": 0.5, "confidence": 0.9, "key_factor": "Positive earnings report"}
            )
        elif "Analyze these day trades" in system_prompt:
            return "### Post Trade Analysis\nWinners were mostly in the IT sector following positive global cues."
        return "{}"

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        if self.hf_token == "MOCK_KEY_FOR_TESTING" or not self.hf_token.startswith("hf_"):
            logger.warning("Using MOCK HuggingFace response because no valid HF token is found.")
            return self._mock_response(system_prompt, user_prompt)

        # Build prompt format using chat completions structure
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.2
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.hf_token}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_bytes = response.read()
                res_json = json.loads(res_bytes.decode("utf-8"))
                
                # Check response format
                if "choices" in res_json and len(res_json["choices"]) > 0:
                    text = res_json["choices"][0].get("message", {}).get("content", "")
                else:
                    text = str(res_json)

                # Clean any markdown tags or JSON code block wrapping if model output is formatted
                text_clean = text.strip()
                if text_clean.startswith("```json"):
                    text_clean = text_clean.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                elif text_clean.startswith("```"):
                    text_clean = text_clean.split("```", 1)[1].rsplit("```", 1)[0].strip()
                return text_clean

        except Exception as e:
            logger.error(f"HuggingFace Inference API call failed: {e}", exc_info=True)
            return self._mock_response(system_prompt, user_prompt)

    async def ask_async(self, system_prompt: str, user_prompt: str) -> str:
        # Wrap the synchronous call in executor to keep it non-blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.ask, system_prompt, user_prompt)
