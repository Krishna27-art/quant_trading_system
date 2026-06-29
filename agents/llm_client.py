import json
import os

import anthropic

from utils.structured_logger import get_structured_logger

logger = get_structured_logger("llm_client")


class AnthropicClient:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "MOCK_KEY_FOR_TESTING")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.model = "claude-3-haiku-20240307"

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        if "Pre-Market" in system_prompt or "Indian equities day trading analyst" in system_prompt:
            return json.dumps(
                {
                    "global_cues": "Positive",
                    "sector_outlook": ["IT", "FINANCIALS"],
                    "stocks_to_watch": ["TCS", "HDFC"],
                    "risk_factors": ["High VIX"],
                    "regime": "Trending Up",
                }
            )
        elif "Rate the following recent news headlines" in system_prompt:
            return json.dumps(
                {"score": 0.5, "confidence": 0.9, "key_factor": "Positive earnings report"}
            )
        elif "Analyze these day trades" in system_prompt:
            return "### Post Trade Analysis\nWinners were mostly in the IT sector following positive global cues."
        return "{}"

    def ask_claude(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generic Claude caller. If the API key is not a real key, returns mocked structural JSON.
        """
        if self.api_key == "MOCK_KEY_FOR_TESTING" or not self.api_key.startswith("sk-ant"):
            logger.warning("Using MOCK Anthropic response because no valid API key is found.")
            return self._mock_response(system_prompt, user_prompt)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception:
            logger.error("Claude API call failed", exc_info=True)
            return self._mock_response(system_prompt, user_prompt)  # Fallback to mock

    async def ask_claude_async(self, system_prompt: str, user_prompt: str) -> str:
        """
        Asynchronous Claude caller. If the API key is not a real key, returns mocked structural JSON.
        """
        if self.api_key == "MOCK_KEY_FOR_TESTING" or not self.api_key.startswith("sk-ant"):
            logger.warning("Using MOCK Anthropic response because no valid API key is found.")
            return self._mock_response(system_prompt, user_prompt)

        try:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text
        except Exception:
            logger.error("Claude Async API call failed", exc_info=True)
            return self._mock_response(system_prompt, user_prompt)  # Fallback to mock


# Singleton instance
llm = AnthropicClient()
