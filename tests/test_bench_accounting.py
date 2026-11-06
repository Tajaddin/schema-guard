"""Cost accounting of the instructor strategy in bench/compare.py.

Runs the real instructor retry loop against the real Anthropic SDK, with the
HTTP layer replaced by a scripted fake, so the token counts are known exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

anthropic = pytest.importorskip("anthropic")
pytest.importorskip("instructor")
httpx = pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import compare  # noqa: E402


class Person(BaseModel):
    name: str
    age: int


VALID = {"name": "Ann", "age": 41}
INVALID = {"name": "Ann", "age": "forty-one"}


def _fake_anthropic(monkeypatch, replies: list[tuple[dict, int, int]]) -> list[dict]:
    """Patch ``anthropic.Anthropic`` so each request gets the next scripted reply."""
    requests: list[dict] = []
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        tool_input, in_tok, out_tok = queue.pop(0)
        return httpx.Response(
            200,
            json={
                "id": f"msg_{len(requests)}",
                "type": "message",
                "role": "assistant",
                "model": compare.PRIMARY_MODEL,
                "content": [
                    {"type": "tool_use", "id": f"toolu_{len(requests)}", "name": "Person", "input": tool_input}
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
            },
        )

    real_cls = anthropic.Anthropic

    class ScriptedAnthropic(real_cls):  # type: ignore[misc, valid-type]
        def __init__(self, **_: object) -> None:
            super().__init__(
                api_key="test-key",
                max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )

    monkeypatch.setattr(anthropic, "Anthropic", ScriptedAnthropic)
    return requests


@pytest.mark.parametrize(
    ("replies", "max_retries", "success"),
    [
        pytest.param([(VALID, 100, 20)], 3, True, id="first-attempt"),
        pytest.param([(INVALID, 100, 20), (VALID, 150, 25)], 3, True, id="one-retry"),
        pytest.param([(INVALID, 100, 20), (INVALID, 150, 25)], 2, False, id="retries-exhausted"),
    ],
)
def test_instructor_cost_counts_each_attempt_once(monkeypatch, replies, max_retries, success) -> None:
    requests = _fake_anthropic(monkeypatch, replies)

    row = compare.run_instructor("Ann is forty-one.", Person, max_retries=max_retries)

    assert len(requests) == len(replies)
    assert row["success"] is success
    expected = compare.cost_of(
        compare.PRIMARY_MODEL,
        sum(in_tok for _, in_tok, _ in replies),
        sum(out_tok for _, _, out_tok in replies),
    )
    assert row["cost_usd"] == pytest.approx(expected)
