"""Shared types: LLMClient protocol, Tier enum, Attempt, Result."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, Protocol, TypeVar


class Tier(str, Enum):
    """Each attempt in the escalation chain.

    The tiers are ordered cheapest → most expensive. ``SchemaGuard.call``
    stops at the first tier whose output validates against the schema.
    """

    T1_PRIMARY = "T1_primary"
    T2_PRIMARY_REPAIR = "T2_primary_repair"
    T3_SMART_REGEN = "T3_smart_regen"
    T4_PARTIAL = "T4_partial"


@dataclass
class CompletionResult:
    """One LLM completion with token-level cost data."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float = 0.0


class LLMClient(Protocol):
    """Minimal LLM client interface.

    Implementations must report token counts in the returned
    :class:`CompletionResult` so :class:`schema_guard.core.SchemaGuard` can
    compute real cost. They should also expose ``model`` and per-million-token
    pricing so it can attribute spend correctly.
    """

    model: str
    input_cost_per_mtok: float
    output_cost_per_mtok: float

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1000) -> CompletionResult:
        """Return a completion with token-level usage information."""
        ...


@dataclass
class Attempt:
    """One step in the escalation chain — success or failure."""

    tier: Tier
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    success: bool
    raw_output: str
    error: str | None = None


T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    """Final result of :meth:`SchemaGuard.call`.

    ``value`` is ``None`` only when every tier failed and partial extraction
    also produced nothing usable. ``tier_reached`` is the tier that produced
    the returned value (or the last tier attempted if all failed).
    """

    value: T | None
    tier_reached: Tier
    total_cost_usd: float
    total_latency_ms: int
    attempts: list[Attempt]
    success: bool
    partial: bool = False
    raw_output: str = ""

    @property
    def n_llm_calls(self) -> int:
        """How many LLM calls were made. T4 (partial) is free; doesn't count."""
        return sum(1 for a in self.attempts if a.tier != Tier.T4_PARTIAL)


def make_attempt(
    tier: Tier,
    completion: CompletionResult | None,
    latency_ms: int,
    success: bool,
    raw_output: str,
    error: str | None = None,
) -> Attempt:
    """Build an :class:`Attempt` from either a real completion or a zero-cost step."""
    if completion is None:
        return Attempt(
            tier=tier,
            model="(no model)",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=latency_ms,
            success=success,
            raw_output=raw_output,
            error=error,
        )
    return Attempt(
        tier=tier,
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=completion.cost_usd,
        latency_ms=latency_ms,
        success=success,
        raw_output=raw_output,
        error=error,
    )
