"""LLM client adapters with token-level cost tracking.

Each client implements :class:`schema_guard.types.LLMClient` and reports
per-call token counts so cost is computed exactly, not estimated.
"""

from __future__ import annotations

import os

from schema_guard.types import CompletionResult


class _PricedClient:
    """Base helper: per-million-token pricing → per-call cost in USD."""

    model: str
    input_cost_per_mtok: float
    output_cost_per_mtok: float

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_cost_per_mtok
            + output_tokens / 1_000_000 * self.output_cost_per_mtok
        )


# ---------- Mock (tests + offline demos) ----------


class MockLLMClient(_PricedClient):
    """Deterministic mock for tests.

    Resolution order on each call:
    1. ``handler(prompt, system)`` if given
    2. First matching needle in ``canned`` (substring on prompt)
    3. Empty string with the configured base response
    """

    def __init__(
        self,
        model: str = "mock-fast",
        input_cost_per_mtok: float = 0.10,
        output_cost_per_mtok: float = 0.40,
        canned: dict[str, str] | None = None,
        handler=None,
        default: str = "{}",
    ) -> None:
        self.model = model
        self.input_cost_per_mtok = input_cost_per_mtok
        self.output_cost_per_mtok = output_cost_per_mtok
        self.canned = canned or {}
        self.handler = handler
        self.default = default
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> CompletionResult:
        self.calls.append((prompt, system))
        if self.handler is not None:
            text = self.handler(prompt, system)
        else:
            text = self.default
            for needle, answer in self.canned.items():
                if needle in prompt:
                    text = answer
                    break
        # Toy token counting: 1 token ≈ 4 chars (close enough for cost math in tests).
        input_tokens = max(1, (len(prompt) + len(system or "")) // 4)
        output_tokens = max(1, len(text) // 4)
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            cost_usd=self._cost(input_tokens, output_tokens),
        )


# ---------- Anthropic ----------


class AnthropicClient(_PricedClient):
    """Wraps the Anthropic SDK with cost-tracked completions."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        input_cost_per_mtok: float | None = None,
        output_cost_per_mtok: float | None = None,
    ) -> None:
        from anthropic import Anthropic  # imported lazily

        self.model = model or os.environ.get("SCHEMA_GUARD_PRIMARY_MODEL", "claude-haiku-4-5-20251001")
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        defaults = _ANTHROPIC_PRICING.get(self.model, _ANTHROPIC_PRICING["claude-haiku-4-5-20251001"])
        self.input_cost_per_mtok = input_cost_per_mtok if input_cost_per_mtok is not None else defaults[0]
        self.output_cost_per_mtok = output_cost_per_mtok if output_cost_per_mtok is not None else defaults[1]

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> CompletionResult:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        text_parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        text = "".join(text_parts)
        input_tokens = getattr(resp.usage, "input_tokens", 0)
        output_tokens = getattr(resp.usage, "output_tokens", 0)
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            cost_usd=self._cost(input_tokens, output_tokens),
        )


_ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    # USD per million tokens (input, output). Public Anthropic pricing as of 2026-01.
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
}


# ---------- OpenAI ----------


class OpenAIClient(_PricedClient):
    """Wraps the OpenAI SDK with cost-tracked completions."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        input_cost_per_mtok: float | None = None,
        output_cost_per_mtok: float | None = None,
    ) -> None:
        from openai import OpenAI  # imported lazily

        self.model = model or os.environ.get("SCHEMA_GUARD_OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        defaults = _OPENAI_PRICING.get(self.model, _OPENAI_PRICING["gpt-4o-mini"])
        self.input_cost_per_mtok = input_cost_per_mtok if input_cost_per_mtok is not None else defaults[0]
        self.output_cost_per_mtok = output_cost_per_mtok if output_cost_per_mtok is not None else defaults[1]

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> CompletionResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        input_tokens = resp.usage.prompt_tokens if resp.usage else 0
        output_tokens = resp.usage.completion_tokens if resp.usage else 0
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            cost_usd=self._cost(input_tokens, output_tokens),
        )


_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


# ---------- Groq ----------


class GroqClient(_PricedClient):
    """Wraps the Groq SDK with cost-tracked completions. Free tier is $0 actual but
    we still report a synthetic cost so cross-provider comparisons make sense."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        input_cost_per_mtok: float = 0.05,
        output_cost_per_mtok: float = 0.10,
    ) -> None:
        from groq import Groq  # imported lazily

        self.model = model or os.environ.get("SCHEMA_GUARD_GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.input_cost_per_mtok = input_cost_per_mtok
        self.output_cost_per_mtok = output_cost_per_mtok

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> CompletionResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            cost_usd=self._cost(input_tokens, output_tokens),
        )
