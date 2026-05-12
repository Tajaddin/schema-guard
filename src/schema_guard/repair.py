"""Repair helpers: JSON extraction, repair prompts, partial extraction fallback."""

from __future__ import annotations

import json
import re
from typing import Any

# Match fenced code blocks; optional `json` language tag.
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

# Heuristic: first balanced { ... } or [ ... ] blob in a string.
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def strip_fences(text: str) -> str:
    """Pull JSON out of markdown fences if present, otherwise return as-is."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def first_json_blob(text: str) -> str | None:
    """Best-effort: return the first { ... } or [ ... ] substring, balanced or not.

    Used by partial extraction. Returns ``None`` if neither shape is found.
    """
    stripped = strip_fences(text)
    obj_match = _OBJECT_RE.search(stripped)
    arr_match = _ARRAY_RE.search(stripped)
    if obj_match and arr_match:
        return obj_match.group(0) if obj_match.start() < arr_match.start() else arr_match.group(0)
    if obj_match:
        return obj_match.group(0)
    if arr_match:
        return arr_match.group(0)
    return None


def _normalize_quotes(text: str) -> str:
    """Smart-quote → straight, single → double for JSON-likely keys/values."""
    text = (
        text.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    # Single-quoted strings → double-quoted (best effort; doesn't handle escaping perfectly).
    text = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', text)
    return text


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def try_json_loads(text: str) -> tuple[Any | None, str | None]:
    """Try ``json.loads`` on the text directly, then on a fence-stripped version,
    then on a normalized version. Return ``(value, None)`` on success or
    ``(None, error_message)`` on final failure."""
    candidates = [text, strip_fences(text)]
    for cand in candidates:
        try:
            return json.loads(cand), None
        except json.JSONDecodeError as exc:
            last_error = str(exc)
    # Heavier normalization pass.
    normalized = _strip_trailing_commas(_normalize_quotes(strip_fences(text)))
    try:
        return json.loads(normalized), None
    except json.JSONDecodeError as exc:
        last_error = str(exc)
    # Extract first blob and try.
    blob = first_json_blob(text)
    if blob:
        try:
            return json.loads(_strip_trailing_commas(_normalize_quotes(blob))), None
        except json.JSONDecodeError as exc:
            last_error = str(exc)
    return None, last_error


REPAIR_SYSTEM = (
    "You repair malformed JSON. Reply with ONLY a single JSON object or array — "
    "no preamble, no commentary, no code fences. Conform exactly to the schema "
    "you are given. Preserve all field names and value types from the schema."
)


def make_repair_prompt(raw: str, schema_json: str, error: str | None) -> str:
    """Construct the user prompt for T2 repair."""
    error_block = f"Parser error:\n{error}\n\n" if error else ""
    return (
        f"Raw output that failed to parse:\n{raw}\n\n"
        f"{error_block}"
        f"Target JSON schema:\n{schema_json}\n\n"
        "Return only the corrected JSON object/array. No explanation."
    )


def make_strict_regen_prompt(original_prompt: str, schema_json: str, prior_error: str | None) -> str:
    """Construct the user prompt for T3 regeneration with a stronger model."""
    error_block = f"Prior attempts failed with: {prior_error}\n\n" if prior_error else ""
    return (
        f"Original task:\n{original_prompt}\n\n"
        f"{error_block}"
        f"You MUST respond with valid JSON that conforms to this schema:\n{schema_json}\n\n"
        "Return only the JSON object/array."
    )


def schema_to_json(schema_cls: Any) -> str:
    """Render a Pydantic v2 model schema as a stable JSON string."""
    return json.dumps(schema_cls.model_json_schema(), separators=(",", ":"))


def partial_extract(text: str, schema_cls: Any) -> Any | None:
    """T4 fallback: best-effort extraction with the schema's defaults filled in.

    Tries to parse whatever JSON-like substring is in the text. For any field
    the schema requires that isn't present, we fall back to the field's default
    if one exists, or skip if not. Returns the model instance on success, or
    ``None`` if no usable shape can be salvaged.
    """
    parsed, _err = try_json_loads(text)
    if parsed is None:
        blob = first_json_blob(text)
        if blob is None:
            return None
        try:
            parsed = json.loads(_strip_trailing_commas(_normalize_quotes(blob)))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        # If schema is a wrapper around an array, accept the array.
        try:
            return schema_cls.model_validate(parsed)
        except Exception:  # noqa: BLE001
            return None
    # Fill in defaults for any missing required fields where we can.
    fields = getattr(schema_cls, "model_fields", {})
    repaired = dict(parsed)
    for name, info in fields.items():
        if name in repaired:
            continue
        default = info.default
        if default is not None and str(default) != "PydanticUndefined":
            repaired[name] = default
    try:
        return schema_cls.model_validate(repaired)
    except Exception:  # noqa: BLE001
        return None
