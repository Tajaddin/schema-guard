"""SchemaGuard: tiered, cost-aware repair for LLM structured outputs.

Flow:
  T1 (primary):         call the configured primary model, try strict parse + validate.
  T2 (primary repair):  same primary model, repair prompt over the raw output.
  T3 (smart regen):     escalate to the more capable model with a strict schema prompt.
  T4 (partial):         free regex extraction + default-fill.

Stops at the first tier whose output validates against the Pydantic schema.
"""

from __future__ import annotations

import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from schema_guard.repair import (
    REPAIR_SYSTEM,
    make_repair_prompt,
    make_strict_regen_prompt,
    partial_extract,
    schema_to_json,
    try_json_loads,
)
from schema_guard.types import (
    Attempt,
    LLMClient,
    Result,
    Tier,
    make_attempt,
)

T = TypeVar("T", bound=BaseModel)


class SchemaGuard:
    """Wrap an LLM call with tiered repair.

    Parameters
    ----------
    primary:
        The LLM the user actually wants to use (usually cheap+fast).
    smart_fallback:
        A more capable LLM used only on T3. If omitted, T3 falls back to the
        primary with a stricter prompt (still useful but with less lift).
    enable_partial_fallback:
        Whether to run T4 partial extraction when all LLM tiers fail.
    max_repair_tokens:
        Token cap for the repair / regen prompts. Repair is short by design.
    """

    def __init__(
        self,
        primary: LLMClient,
        smart_fallback: LLMClient | None = None,
        *,
        enable_partial_fallback: bool = True,
        max_repair_tokens: int = 1500,
    ) -> None:
        self.primary = primary
        self.smart_fallback = smart_fallback
        self.enable_partial_fallback = enable_partial_fallback
        self.max_repair_tokens = max_repair_tokens

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def call(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_tokens: int = 1000,
    ) -> Result[T]:
        attempts: list[Attempt] = []
        schema_json = schema_to_json(schema)

        # ---- T1: primary attempt
        t1_attempt, parsed = self._tier_primary(prompt, schema, schema_json, system, max_tokens)
        attempts.append(t1_attempt)
        if t1_attempt.success and parsed is not None:
            return self._build_result(parsed, Tier.T1_PRIMARY, attempts, success=True)

        # ---- T2: primary repair
        t2_attempt, parsed = self._tier_primary_repair(
            schema, schema_json, raw_output=t1_attempt.raw_output, prior_error=t1_attempt.error
        )
        attempts.append(t2_attempt)
        if t2_attempt.success and parsed is not None:
            return self._build_result(parsed, Tier.T2_PRIMARY_REPAIR, attempts, success=True)

        # ---- T3: smart regen
        t3_attempt, parsed = self._tier_smart_regen(
            prompt, schema, schema_json, system, prior_error=t2_attempt.error or t1_attempt.error
        )
        attempts.append(t3_attempt)
        if t3_attempt.success and parsed is not None:
            return self._build_result(parsed, Tier.T3_SMART_REGEN, attempts, success=True)

        # ---- T4: partial extraction (free)
        if self.enable_partial_fallback:
            t4_attempt, parsed = self._tier_partial(schema, attempts)
            attempts.append(t4_attempt)
            if t4_attempt.success and parsed is not None:
                return self._build_result(parsed, Tier.T4_PARTIAL, attempts, success=True, partial=True)

        return self._build_result(None, attempts[-1].tier, attempts, success=False)

    # ------------------------------------------------------------------
    # tier implementations
    # ------------------------------------------------------------------
    def _tier_primary(
        self,
        prompt: str,
        schema: type[T],
        schema_json: str,
        system: str | None,
        max_tokens: int,
    ) -> tuple[Attempt, T | None]:
        start = time.perf_counter()
        primed_system = self._prime_system(system, schema_json)
        completion = self.primary.complete(prompt, system=primed_system, max_tokens=max_tokens)
        elapsed = int((time.perf_counter() - start) * 1000)
        parsed, err = self._parse_and_validate(completion.text, schema)
        return (
            make_attempt(
                Tier.T1_PRIMARY,
                completion,
                elapsed,
                success=parsed is not None,
                raw_output=completion.text,
                error=err,
            ),
            parsed,
        )

    def _tier_primary_repair(
        self,
        schema: type[T],
        schema_json: str,
        raw_output: str,
        prior_error: str | None,
    ) -> tuple[Attempt, T | None]:
        start = time.perf_counter()
        repair_prompt = make_repair_prompt(raw_output, schema_json, prior_error)
        completion = self.primary.complete(
            repair_prompt, system=REPAIR_SYSTEM, max_tokens=self.max_repair_tokens
        )
        elapsed = int((time.perf_counter() - start) * 1000)
        parsed, err = self._parse_and_validate(completion.text, schema)
        return (
            make_attempt(
                Tier.T2_PRIMARY_REPAIR,
                completion,
                elapsed,
                success=parsed is not None,
                raw_output=completion.text,
                error=err,
            ),
            parsed,
        )

    def _tier_smart_regen(
        self,
        prompt: str,
        schema: type[T],
        schema_json: str,
        system: str | None,
        prior_error: str | None,
    ) -> tuple[Attempt, T | None]:
        start = time.perf_counter()
        client = self.smart_fallback or self.primary
        regen_prompt = make_strict_regen_prompt(prompt, schema_json, prior_error)
        primed_system = self._prime_system(system, schema_json, strict=True)
        completion = client.complete(
            regen_prompt, system=primed_system, max_tokens=self.max_repair_tokens
        )
        elapsed = int((time.perf_counter() - start) * 1000)
        parsed, err = self._parse_and_validate(completion.text, schema)
        return (
            make_attempt(
                Tier.T3_SMART_REGEN,
                completion,
                elapsed,
                success=parsed is not None,
                raw_output=completion.text,
                error=err,
            ),
            parsed,
        )

    def _tier_partial(
        self, schema: type[T], prior_attempts: list[Attempt]
    ) -> tuple[Attempt, T | None]:
        start = time.perf_counter()
        # Use the most recent non-empty raw output as the salvage source.
        source = ""
        for a in reversed(prior_attempts):
            if a.raw_output.strip():
                source = a.raw_output
                break
        parsed = partial_extract(source, schema)
        elapsed = int((time.perf_counter() - start) * 1000)
        return (
            make_attempt(
                Tier.T4_PARTIAL,
                completion=None,
                latency_ms=elapsed,
                success=parsed is not None,
                raw_output=source,
                error=None if parsed is not None else "partial extraction failed",
            ),
            parsed,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prime_system(
        user_system: str | None, schema_json: str, *, strict: bool = False
    ) -> str:
        base = (
            user_system.strip() + "\n\n"
            if user_system
            else ""
        )
        clause = (
            "You MUST respond with a single valid JSON object/array that conforms to this schema. "
            "No prose, no markdown fences, no extra commentary."
        )
        if strict:
            clause = (
                "STRICT MODE: respond with a single valid JSON object/array exactly matching this "
                "schema. Any deviation will be rejected. No prose, no markdown fences, no comments."
            )
        return f"{base}{clause}\n\nSchema:\n{schema_json}"

    @staticmethod
    def _parse_and_validate(text: str, schema: type[T]) -> tuple[T | None, str | None]:
        parsed, err = try_json_loads(text)
        if parsed is None:
            return None, err
        try:
            return schema.model_validate(parsed), None
        except ValidationError as exc:
            return None, str(exc)

    @staticmethod
    def _build_result(
        value,
        tier: Tier,
        attempts: list[Attempt],
        *,
        success: bool,
        partial: bool = False,
    ) -> Result:
        total_cost = sum(a.cost_usd for a in attempts)
        total_latency = sum(a.latency_ms for a in attempts)
        raw_output = attempts[-1].raw_output if attempts else ""
        return Result(
            value=value,
            tier_reached=tier,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            attempts=attempts,
            success=success,
            partial=partial,
            raw_output=raw_output,
        )


def _coerce_attempt_list(attempts) -> list[Attempt]:  # pragma: no cover - kept for type-hint readability
    """Identity helper kept to make the function-signature in tests clearer."""
    return list(attempts)
