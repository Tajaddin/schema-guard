"""Head-to-head structured-output benchmark.

Strategies compared on the same prompts + same Anthropic model:

1. ``raw``           — single primary call, ``json.loads`` + Pydantic validate
2. ``instructor``    — instructor.from_anthropic with response_model + max_retries
3. ``pydantic-ai``   — Agent(model=..., output_type=Schema).run_sync
4. ``schema-guard``  — this project: tiered T1→T2→T3→T4

Metrics:
* compliance rate (% of prompts that produced a schema-valid object)
* total cost USD (token-based, using public Anthropic pricing)
* total wall-clock latency
* per-prompt detail goes into ``bench/compare_results.json``

Outlines is intentionally excluded: it requires logit-level access to a local
model and cannot run against a closed API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_tasks
from pydantic import ValidationError

from schema_guard import AnthropicClient, SchemaGuard
from schema_guard.repair import try_json_loads

# Public Anthropic Haiku 4.5 pricing (USD per million tokens).
HAIKU_IN_COST = 0.80
HAIKU_OUT_COST = 4.00
SONNET_IN_COST = 3.00
SONNET_OUT_COST = 15.00

PRIMARY_MODEL = "claude-haiku-4-5-20251001"
SMART_MODEL = "claude-sonnet-4-6"


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    if "haiku" in model:
        return input_tokens / 1e6 * HAIKU_IN_COST + output_tokens / 1e6 * HAIKU_OUT_COST
    if "sonnet" in model:
        return input_tokens / 1e6 * SONNET_IN_COST + output_tokens / 1e6 * SONNET_OUT_COST
    return 0.0


# ----------------------------- strategies -----------------------------


def run_raw(prompt: str, schema: Any) -> dict:
    """Single primary call. Strict json.loads + Pydantic validate. No retry."""
    from anthropic import Anthropic

    client = Anthropic()
    start = time.perf_counter()
    system_msg = (
        "Respond with a single valid JSON object that conforms to this schema. "
        "No prose, no markdown fences.\n\nSchema:\n" + schema_to_json(schema)
    )
    try:
        resp = client.messages.create(
            model=PRIMARY_MODEL,
            max_tokens=1000,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.perf_counter() - start) * 1000)
        return {"success": False, "value": None, "tier": "raw", "cost_usd": 0.0, "latency_ms": elapsed, "error": str(exc)[:200]}
    raw_text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    elapsed = int((time.perf_counter() - start) * 1000)
    cost = cost_of(PRIMARY_MODEL, resp.usage.input_tokens, resp.usage.output_tokens)
    parsed, err = try_json_loads(raw_text)
    if parsed is None:
        return {"success": False, "value": None, "tier": "raw", "cost_usd": cost, "latency_ms": elapsed, "error": err}
    try:
        obj = schema.model_validate(parsed)
        return {"success": True, "value": obj.model_dump(), "tier": "raw", "cost_usd": cost, "latency_ms": elapsed, "error": None}
    except ValidationError as exc:
        return {"success": False, "value": None, "tier": "raw", "cost_usd": cost, "latency_ms": elapsed, "error": str(exc)[:200]}


def run_instructor(prompt: str, schema: Any, max_retries: int = 3) -> dict:
    """Instructor with default retry (same primary model on each retry)."""
    import instructor
    from anthropic import Anthropic

    raw_client = Anthropic()
    client = instructor.from_anthropic(raw_client)
    start = time.perf_counter()
    # Instructor doesn't expose token counts directly; intercept by attaching a hook.
    usage_log: list[tuple[int, int]] = []

    def _on_response(resp):
        if hasattr(resp, "usage"):
            usage_log.append((resp.usage.input_tokens, resp.usage.output_tokens))

    try:
        client.on("completion:response", _on_response)
        obj, completion = client.messages.create_with_completion(
            model=PRIMARY_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            response_model=schema,
            max_retries=max_retries,
        )
        elapsed = int((time.perf_counter() - start) * 1000)
        if hasattr(completion, "usage"):
            usage_log.append((completion.usage.input_tokens, completion.usage.output_tokens))
        total_in = sum(i for i, _ in usage_log)
        total_out = sum(o for _, o in usage_log)
        cost = cost_of(PRIMARY_MODEL, total_in, total_out)
        return {"success": True, "value": obj.model_dump(), "tier": "instructor", "cost_usd": cost, "latency_ms": elapsed, "error": None}
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.perf_counter() - start) * 1000)
        total_in = sum(i for i, _ in usage_log)
        total_out = sum(o for _, o in usage_log)
        cost = cost_of(PRIMARY_MODEL, total_in, total_out)
        return {"success": False, "value": None, "tier": "instructor", "cost_usd": cost, "latency_ms": elapsed, "error": str(exc)[:200]}


def run_pydantic_ai(prompt: str, schema: Any) -> dict:
    """Pydantic-AI Agent with output_type=schema."""
    from pydantic_ai import Agent

    agent = Agent(f"anthropic:{PRIMARY_MODEL}", output_type=schema)
    start = time.perf_counter()
    try:
        result = agent.run_sync(prompt)
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.perf_counter() - start) * 1000)
        return {"success": False, "value": None, "tier": "pydantic-ai", "cost_usd": 0.0, "latency_ms": elapsed, "error": str(exc)[:200]}
    elapsed = int((time.perf_counter() - start) * 1000)
    usage = getattr(result, "usage", None)
    if callable(usage):
        usage = usage()
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    cost = cost_of(PRIMARY_MODEL, input_tokens, output_tokens)
    try:
        out = result.output
        return {"success": True, "value": out.model_dump(), "tier": "pydantic-ai", "cost_usd": cost, "latency_ms": elapsed, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "value": None, "tier": "pydantic-ai", "cost_usd": cost, "latency_ms": elapsed, "error": str(exc)[:200]}


def run_schema_guard(prompt: str, schema: Any) -> dict:
    """schema-guard with Haiku primary + Sonnet smart fallback."""
    primary = AnthropicClient(model=PRIMARY_MODEL)
    smart = AnthropicClient(model=SMART_MODEL)
    guard = SchemaGuard(primary=primary, smart_fallback=smart)
    start = time.perf_counter()
    try:
        result = guard.call(prompt, schema)
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.perf_counter() - start) * 1000)
        return {"success": False, "value": None, "tier": "schema-guard", "cost_usd": 0.0, "latency_ms": elapsed, "error": str(exc)[:200]}
    elapsed = int((time.perf_counter() - start) * 1000)
    return {
        "success": result.success,
        "value": result.value.model_dump() if result.value is not None else None,
        "tier": result.tier_reached.value,
        "cost_usd": result.total_cost_usd,
        "latency_ms": elapsed,
        "error": None if result.success else "all tiers failed",
        "partial": result.partial,
        "n_llm_calls": result.n_llm_calls,
    }


# ----------------------------- helpers -----------------------------


def schema_to_json(schema: Any) -> str:
    return json.dumps(schema.model_json_schema(), separators=(",", ":"))


# ----------------------------- main -----------------------------


def summarize(strategy: str, rows: list[dict]) -> dict:
    n = len(rows)
    n_ok = sum(1 for r in rows if r["success"])
    total_cost = sum(r["cost_usd"] for r in rows)
    total_latency = sum(r["latency_ms"] for r in rows) / 1000.0  # seconds
    avg_latency = total_latency / max(n, 1)
    return {
        "strategy": strategy,
        "n": n,
        "compliance_rate": round(n_ok / n, 4) if n else 0.0,
        "total_cost_usd": round(total_cost, 5),
        "total_latency_s": round(total_latency, 1),
        "avg_latency_s": round(avg_latency, 2),
        "cost_per_success_usd": round(total_cost / max(n_ok, 1), 5),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Subset N tasks")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["raw", "instructor", "pydantic-ai", "schema-guard"],
        choices=["raw", "instructor", "pydantic-ai", "schema-guard"],
    )
    parser.add_argument("--out", type=str, default="bench/compare_results.json")
    args = parser.parse_args()

    tasks = get_tasks(limit=args.limit, seed=args.seed)
    print(f"Running {len(tasks)} tasks against {args.strategies}\n")

    all_rows: dict[str, list[dict]] = {s: [] for s in args.strategies}
    runners = {
        "raw": run_raw,
        "instructor": run_instructor,
        "pydantic-ai": run_pydantic_ai,
        "schema-guard": run_schema_guard,
    }

    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] {task['id']}")
        for strat in args.strategies:
            try:
                row = runners[strat](task["prompt"], task["schema"])
            except Exception as exc:  # noqa: BLE001
                row = {
                    "success": False,
                    "value": None,
                    "tier": strat,
                    "cost_usd": 0.0,
                    "latency_ms": 0,
                    "error": f"runner crash: {type(exc).__name__}: {str(exc)[:200]}",
                }
            row["task_id"] = task["id"]
            all_rows[strat].append(row)
            mark = "OK " if row["success"] else "FAIL"
            print(f"  {strat:14s} {mark}  cost=${row['cost_usd']:.5f} lat={row['latency_ms']}ms")

    summaries = [summarize(s, all_rows[s]) for s in args.strategies]
    out = {"summaries": summaries, "rows": all_rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
