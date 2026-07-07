"""
LLM Client Interface (Deprecated — use the modular 'llm' package directly)

Maintained for backward-compatible imports in legacy scripts.
"""

from llm.router import LLMRouter

# Expose LLMRouter instance as singleton instance 'llm'
llm = LLMRouter()
