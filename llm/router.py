import os
from llm.base import BaseLLMClient
from llm.huggingface_client import HuggingFaceClient
from utils.logger import get_logger

logger = get_logger("llm_router")


class LLMRouter(BaseLLMClient):
    """
    Router that delegates queries to Hugging Face (Qwen/Qwen3-32B).
    Provides backward-compatibility methods for existing Claude integrations.
    """

    def __init__(self):
        model = os.getenv("LLM_MODEL") or "Qwen/Qwen3-32B"
        logger.info(f"LLMRouter initialized with Hugging Face Provider [{model}]")
        self.client = HuggingFaceClient(model_id=model)

    def ask(self, system_prompt: str, user_prompt: str) -> str:
        return self.client.ask(system_prompt, user_prompt)

    async def ask_async(self, system_prompt: str, user_prompt: str) -> str:
        return await self.client.ask_async(system_prompt, user_prompt)

    # ── Backward Compatibility API ──────────────────────────────────────────

    def ask_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Alias for ask() to avoid breaking existing callers."""
        return self.ask(system_prompt, user_prompt)

    async def ask_claude_async(self, system_prompt: str, user_prompt: str) -> str:
        """Alias for ask_async() to avoid breaking existing callers."""
        return await self.ask_async(system_prompt, user_prompt)

# Singleton instance
llm = LLMRouter()
