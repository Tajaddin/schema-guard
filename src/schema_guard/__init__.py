"""Cost-aware tiered repair for LLM structured outputs.

Run cheap first, escalate to smart only when needed, fall back to partial
extraction as a last resort. Compatible with any LLM client implementing the
:class:`schema_guard.types.LLMClient` protocol.
"""

from schema_guard.clients import AnthropicClient, GroqClient, MockLLMClient, OpenAIClient
from schema_guard.core import SchemaGuard
from schema_guard.types import Attempt, CompletionResult, LLMClient, Result, Tier

__version__ = "0.1.0"

__all__ = [
    "Attempt",
    "CompletionResult",
    "LLMClient",
    "Result",
    "Tier",
    "SchemaGuard",
    "AnthropicClient",
    "OpenAIClient",
    "GroqClient",
    "MockLLMClient",
]
