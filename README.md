# schema-guard

Cost-aware tiered repair for LLM structured outputs. Same 100 % compliance as `instructor` and `pydantic-ai`, at 2.77× lower cost than instructor and 2.72× lower cost than pydantic-ai, by asking for plain-text JSON instead of a tool call and escalating only when validation fails.

[![ci](https://github.com/Tajaddin/schema-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/Tajaddin/schema-guard/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

## Hero benchmark

50 adversarial structured-output tasks, Claude Haiku 4.5 primary, Claude Sonnet 4.6 smart fallback (used only on T3). Public Anthropic pricing.

| Strategy | Compliance | Total cost | Cost/success | Avg latency | Overhead vs raw |
|---|---:|---:|---:|---:|---:|
| `raw` model + Pydantic validate | 100% | $0.02084 | $0.00042 | 0.97 s | baseline |
| **`schema-guard`** (this project) | **100%** | **$0.02164** | **$0.00043** | **0.92 s** | **+3.8%** |
| `pydantic-ai` Agent | 100% | $0.05893 | $0.00118 | 1.06 s | +183% |
| `instructor` `max_retries=3` | 100% | $0.05990 | $0.00120 | 1.08 s | +187% |

Headline: schema-guard matches the bare-metal raw-model cost (+3.8 % overhead) while pydantic-ai costs 2.72× more and instructor 2.77× more at identical compliance.

The reason: instructor and pydantic-ai request the object as a tool call, so every request carries a tool definition and the tool-use system prompt the API adds, even when the first answer is valid. schema-guard asks for plain-text JSON and validates it; the repair tiers fire only when validation fails. In this run they never fired: all 50 schema-guard tasks stopped at T1 after one call, so the saving comes from the request format, not from skipped retries. The run did not record token counts, so how the gap splits between input and output tokens is not measured.

Full numbers in [`bench/compare_results.json`](bench/compare_results.json). The instructor costs were corrected on 2026-11-06: the original script counted the tokens of every successful instructor call twice, so those recorded costs were halved. The same day, every cost was rescaled from $0.80/$4 to Haiku 4.5's list price of $1/$5 per million tokens, an exact factor of 1.25 because every call in the run went to Haiku. Neither correction was a re-run (see `note` in the results file).

## Why "tiered" repair

A single LLM call to a modern model produces valid JSON ~99% of the time. Existing tools handle the 1% wrong:

| Tool | When validation fails it... | Cost when output is already valid |
|---|---|---|
| `instructor` | Retries with the **same** model + error feedback | Already 2-3× raw because every call uses tool-calling |
| `pydantic-ai` | Retries with the **same** model + Pydantic error | 2-3× raw because every call uses tool-calling |
| `outlines` | Constrained decoding (logit bias), local models only | 0 % (no API overhead) but requires logit access |
| **`schema-guard`** | T1 raw call → T2 cheap repair → T3 smart regen → T4 partial extract | **+3.8% raw** because most calls stop at T1 |

The trick: **don't pay until you have to.** schema-guard issues a normal text completion at T1. If `json.loads` + Pydantic validate pass, it returns immediately at raw-model cost. The repair tiers only fire when the validator complains.

## Architecture

```
   call(prompt, schema)
            │
            ▼
   ┌────────────────────┐
   │  T1: primary LLM   │  →  json.loads + Pydantic validate
   └────────┬───────────┘
            │  fail
            ▼
   ┌────────────────────┐    "repair this JSON to match the schema"
   │  T2: primary +     │    (cheap model + raw output + schema)
   │  repair prompt     │  →  validate
   └────────┬───────────┘
            │  fail
            ▼
   ┌────────────────────┐    "regenerate strictly to schema"
   │  T3: smart model + │    (Sonnet, Opus, GPT-4o, …)
   │  STRICT prompt     │  →  validate
   └────────┬───────────┘
            │  fail
            ▼
   ┌────────────────────┐    regex + Pydantic defaults
   │  T4: partial       │  →  partial=True flag
   │  extraction (free) │
   └────────┬───────────┘
            │
            ▼
        Result(value, tier_reached, cost_usd, attempts, partial)
```

Every step writes an `Attempt` with the model, input/output tokens, exact cost, and latency. The `Result.attempts` list is the audit trail.

## Install

```bash
pip install -e ".[anthropic,openai,groq]"
```

Or, for the comparison benchmark + dev tools:

```bash
pip install -e ".[anthropic,openai,groq,bench,dev]"
```

## Usage

```python
from pydantic import BaseModel
from schema_guard import SchemaGuard, AnthropicClient

class Person(BaseModel):
    name: str
    age: int

primary = AnthropicClient(model="claude-haiku-4-5-20251001")  # cheap default
smart   = AnthropicClient(model="claude-sonnet-4-6")          # used only on T3

guard = SchemaGuard(primary=primary, smart_fallback=smart)

result = guard.call(
    "Extract person from: 'Alice Johnson is 34 years old.'",
    schema=Person,
)

assert result.success
print(result.value)            # Person(name='Alice Johnson', age=34)
print(result.tier_reached)     # Tier.T1_PRIMARY
print(result.total_cost_usd)   # 0.00012
print(result.n_llm_calls)      # 1
```

If you only want to use one model (no `smart_fallback`), T3 falls back to running the primary again with a stricter prompt. Less lift on hard cases, but still cheaper than always-retry:

```python
guard = SchemaGuard(primary=primary)  # smart_fallback defaults to primary
```

### Inspect the audit trail

```python
result = guard.call(prompt, schema)
for attempt in result.attempts:
    print(
        f"{attempt.tier.value:24s}  {attempt.model:30s}  "
        f"in={attempt.input_tokens:5d} out={attempt.output_tokens:5d} "
        f"cost=${attempt.cost_usd:.5f} {attempt.latency_ms}ms "
        f"{'OK' if attempt.success else 'FAIL: ' + (attempt.error or '')}"
    )
```

### OpenAI / Groq backends

```python
from schema_guard import OpenAIClient, GroqClient

primary = OpenAIClient(model="gpt-4o-mini")
smart   = OpenAIClient(model="gpt-4o")
guard   = SchemaGuard(primary=primary, smart_fallback=smart)

# Or use Groq's free Llama tier for the cheap primary
primary = GroqClient(model="llama-3.1-8b-instant")
smart   = AnthropicClient(model="claude-sonnet-4-6")
guard   = SchemaGuard(primary=primary, smart_fallback=smart)
```

## Run the benchmark yourself

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python bench/compare.py --seed 7
```

Cost on Haiku 4.5 + Sonnet 4.6: ~$0.16 for the full 50-task run across all 4 strategies. Outputs `bench/compare_results.json` and prints the summary table.

To run with a subset:

```bash
python bench/compare.py --limit 10 --strategies raw schema-guard
```

## Tests

```bash
pip install -e ".[anthropic,openai,groq,bench,dev]"
pytest -ra
```

```
30 passed
```

Covers each tier in isolation, full escalation chain, cost accumulation, partial extraction, schema validation, adversarial JSON (smart quotes, trailing commas, embedded prose, single quotes, markdown fences), the benchmark's instructor cost accounting, and construction of each provider client against its pinned SDK. Tests that need an SDK extra that is not installed are skipped.

## Project layout

```
.
├── src/schema_guard/
│   ├── core.py          # SchemaGuard class, the tiered escalation
│   ├── types.py         # LLMClient protocol, Tier, Attempt, Result
│   ├── clients.py       # Anthropic / OpenAI / Groq / Mock adapters
│   └── repair.py        # JSON normalization + repair prompts + partial extract
├── tests/               # 30 pytest cases
└── bench/
    ├── dataset.py       # 50 adversarial structured-output tasks
    ├── compare.py       # Head-to-head benchmark runner
    └── compare_results.json
```

## Caveats

Outlines is not in the comparison. Outlines uses logit-level constraints during generation. Closed-API models (Claude, GPT-4o, and so on) do not expose logits, so Outlines does not apply. Running a local model with vLLM or llama.cpp? Outlines is unbeatable (0 % overhead, guaranteed valid output) and you should use it instead of schema-guard.

Compliance differences widen on smaller models. All four strategies hit 100 % on Haiku 4.5 because Haiku is reliable enough no escalation fires. On a less capable model (Llama-3.1-8B, Phi-3-mini), expect raw to drop below 95 % and schema-guard's T2/T3 lift to become visible. The run is left to the reader. The unit tests in `tests/test_core.py` exercise each tier in isolation.

Cost figures assume public list prices. Enterprise contracts, batch APIs, and prompt caching shift the absolute numbers, but the relative shape (instructor and pydantic-ai cost 2.87× and 2.83× as much as raw here) is intrinsic to their algorithms, not to pricing.

## License

MIT. See [LICENSE](LICENSE).
