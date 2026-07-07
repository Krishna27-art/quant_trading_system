from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    """
    Abstract Base Class for LLM Provider Clients.
    """

    @abstractmethod
    def ask(self, system_prompt: str, user_prompt: str) -> str:
        """
        Synchronous model call.
        """
        pass

    @abstractmethod
    async def ask_async(self, system_prompt: str, user_prompt: str) -> str:
        """
        Asynchronous model call.
        """
        pass
