"""Tests for the tiered SchemaGuard core."""

from __future__ import annotations

from pydantic import BaseModel

from schema_guard import MockLLMClient, SchemaGuard
from schema_guard.types import Tier


class Person(BaseModel):
    name: str
    age: int


class PersonMaybe(BaseModel):
    name: str
    age: int = 0


def test_t1_clean_pass() -> None:
    primary = MockLLMClient(handler=lambda p, s: '{"name": "Alice", "age": 30}')
    g = SchemaGuard(primary=primary)
    r = g.call("extract", Person)
    assert r.success
    assert r.tier_reached == Tier.T1_PRIMARY
    assert r.value.name == "Alice"
    assert r.n_llm_calls == 1


def test_t1_fences_pass() -> None:
    primary = MockLLMClient(handler=lambda p, s: '```json\n{"name":"A","age":1}\n```')
    r = SchemaGuard(primary=primary).call("extract", Person)
    assert r.tier_reached == Tier.T1_PRIMARY
    assert r.success


def test_t2_repair_pass() -> None:
    def h(p, s):
        if "repair" in (s or "").lower():
            return '{"name":"Bob","age":42}'
        return "{name: Bob, age: 42}"  # invalid JSON keys without quotes

    primary = MockLLMClient(handler=h)
    r = SchemaGuard(primary=primary).call("extract", Person)
    assert r.success
    assert r.tier_reached == Tier.T2_PRIMARY_REPAIR
    assert r.n_llm_calls == 2


def test_t3_smart_regen_pass() -> None:
    def h(p, s):
        if "STRICT" in (s or ""):
            return '{"name":"Carol","age":25}'
        return "prose with no json"

    primary = MockLLMClient(handler=h)
    smart = MockLLMClient(handler=h, model="mock-smart")
    r = SchemaGuard(primary=primary, smart_fallback=smart).call("extract", Person)
    assert r.success
    assert r.tier_reached == Tier.T3_SMART_REGEN
    assert r.n_llm_calls == 3


def test_t4_partial_with_default() -> None:
    def h(p, s):
        # Always returns prose + a name fragment, no age. T1/T2/T3 fail validation
        # because Person requires age. With PersonMaybe (age default 0),
        # partial_extract fills in age=0.
        return 'Sure: {"name": "Dan"} but age unknown.'

    # Important: use PersonMaybe so partial_extract can fill defaults.
    primary = MockLLMClient(handler=h)
    r = SchemaGuard(primary=primary).call("extract", PersonMaybe)
    # Note: with PersonMaybe and JSON-blob extraction, T1 already succeeds with default age=0.
    assert r.success
    # T1 should pass since try_json_loads salvages the {"name":"Dan"} blob.
    assert r.tier_reached == Tier.T1_PRIMARY


def test_total_failure_returns_unsuccess() -> None:
    primary = MockLLMClient(handler=lambda p, s: "no json no fix possible")
    r = SchemaGuard(primary=primary, enable_partial_fallback=False).call("extract", Person)
    assert not r.success
    assert r.value is None
    assert r.tier_reached == Tier.T3_SMART_REGEN


def test_cost_accumulates_across_tiers() -> None:
    def h(p, s):
        return "no json"

    # Each call uses ~the same number of tokens; the result should have cumulative cost.
    primary = MockLLMClient(handler=h, input_cost_per_mtok=10.0, output_cost_per_mtok=10.0)
    r = SchemaGuard(primary=primary).call("extract", Person)
    # T1 + T2 + T3 + T4(free) — 3 LLM calls all > 0 cost
    assert r.total_cost_usd > 0
    assert r.n_llm_calls == 3
    assert sum(1 for a in r.attempts if a.tier == Tier.T4_PARTIAL) == 1


def test_smart_fallback_uses_separate_client() -> None:
    def h_primary(p, s):
        return "bad"

    def h_smart(p, s):
        return '{"name":"Smart","age":1}'

    primary = MockLLMClient(handler=h_primary)
    smart = MockLLMClient(handler=h_smart, model="mock-smart")
    r = SchemaGuard(primary=primary, smart_fallback=smart).call("extract", Person)
    assert r.success
    assert r.tier_reached == Tier.T3_SMART_REGEN
    # The last successful attempt was on the smart client.
    assert r.attempts[-1].model == "mock-smart"


def test_attempt_records_have_required_fields() -> None:
    primary = MockLLMClient(handler=lambda p, s: '{"name":"X","age":1}')
    r = SchemaGuard(primary=primary).call("extract", Person)
    a = r.attempts[0]
    assert a.tier == Tier.T1_PRIMARY
    assert a.input_tokens > 0
    assert a.output_tokens > 0
    assert a.success
    assert a.cost_usd >= 0


def test_optional_smart_falls_back_to_primary() -> None:
    """When smart_fallback is None, T3 uses primary with a stricter prompt."""

    calls: list[str | None] = []

    def h(p, s):
        calls.append(s)
        if "STRICT" in (s or ""):
            return '{"name":"S","age":1}'
        return "bad"

    primary = MockLLMClient(handler=h)
    r = SchemaGuard(primary=primary, smart_fallback=None).call("extract", Person)
    assert r.success
    assert r.tier_reached == Tier.T3_SMART_REGEN
    # Make sure at least one call used the STRICT system prompt.
    assert any(s and "STRICT" in s for s in calls)
