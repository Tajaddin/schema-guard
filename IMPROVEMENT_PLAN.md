# Improvement plan

## What changes

- Rewrite README prose in spartan voice. Strip em dashes, semicolons, and banned filler words. Keep every section (hero benchmark, tiered explainer, architecture, install, usage, audit trail, alt backends, bench, tests, layout, caveats). Keep file-tree comments.
- Rename "Limitations" to "Caveats" for cross-repo heading consistency.
- Strip em dashes from docstrings and comments in `src/schema_guard/*.py`, `bench/compare.py`, `bench/dataset.py` (file docstring only), and `tests/test_core.py`. Leave em dashes inside the LLM system/user prompt strings in `repair.py` and the prompt fields of the benchmark dataset, since changing prompt text would invalidate the committed bench numbers.
- Pin every dependency to an exact version in `pyproject.toml`.
- Add `.github/workflows/ci.yml` with ruff plus pytest matrix on Python 3.10 / 3.11 / 3.12.
- Run ruff `--fix` and resolve any remaining lint warnings.

## Why

- README has 7 em dashes and 2 semicolons. Style rules forbid both.
- All deps use `>=` ranges. CI reproducibility drifts.
- No CI workflow.

## Out of scope

- Adding new tiers, models, or alt backends. The 4-tier escalation is the value.
- Re-running the comparison benchmark (~$0.18 on the real API). If the polish does not touch the algorithm or bench script, the committed numbers stand.
- Adding Outlines or local-model support. Documented as a caveat.
