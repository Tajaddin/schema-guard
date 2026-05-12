"""Tests for repair helpers."""

from __future__ import annotations

from pydantic import BaseModel

from schema_guard.repair import (
    first_json_blob,
    partial_extract,
    schema_to_json,
    strip_fences,
    try_json_loads,
)


def test_strip_fences_with_json_tag() -> None:
    text = 'Sure:\n```json\n{"a":1}\n```'
    assert strip_fences(text) == '{"a":1}'


def test_strip_fences_plain_fences() -> None:
    text = "```\nhello\n```"
    assert strip_fences(text) == "hello"


def test_strip_fences_no_fences() -> None:
    text = '   {"a":1}   '
    assert strip_fences(text) == '{"a":1}'


def test_try_json_loads_clean() -> None:
    value, err = try_json_loads('{"a":1}')
    assert value == {"a": 1}
    assert err is None


def test_try_json_loads_with_fences() -> None:
    value, _ = try_json_loads('```json\n{"a":1}\n```')
    assert value == {"a": 1}


def test_try_json_loads_smart_quotes() -> None:
    text = '{“a”: “hi”}'
    value, _ = try_json_loads(text)
    assert value == {"a": "hi"}


def test_try_json_loads_single_quotes() -> None:
    text = "{'a': 'hi'}"
    value, _ = try_json_loads(text)
    assert value == {"a": "hi"}


def test_try_json_loads_trailing_comma() -> None:
    text = '{"a": 1, "b": 2,}'
    value, _ = try_json_loads(text)
    assert value == {"a": 1, "b": 2}


def test_try_json_loads_embedded_in_prose() -> None:
    text = 'Sure! Here you go: {"a": 1} let me know.'
    value, _ = try_json_loads(text)
    assert value == {"a": 1}


def test_try_json_loads_total_failure_returns_error() -> None:
    value, err = try_json_loads("there is no json here at all")
    assert value is None
    assert err is not None


def test_first_json_blob_prefers_earlier_shape() -> None:
    # Array comes first, so first_json_blob should return the array.
    blob = first_json_blob('text [1,2,3] then {"a":1}')
    assert blob == "[1,2,3]"


def test_schema_to_json_stable() -> None:
    class M(BaseModel):
        x: int
        y: str

    j = schema_to_json(M)
    assert "x" in j
    assert "integer" in j


def test_partial_extract_uses_default() -> None:
    class M(BaseModel):
        name: str
        age: int = 0

    # The text has only `name`, but age has a default.
    m = partial_extract('Person: {"name": "Eve"} and more', M)
    assert m is not None
    assert m.name == "Eve"
    assert m.age == 0


def test_partial_extract_returns_none_when_no_json() -> None:
    class M(BaseModel):
        name: str

    assert partial_extract("absolutely no json at all", M) is None
