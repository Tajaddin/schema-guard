"""Each provider client constructs against the pinned SDK, with no network call."""

from __future__ import annotations

import pytest

from schema_guard import AnthropicClient, GroqClient, OpenAIClient


@pytest.mark.parametrize(
    ("client_cls", "sdk"),
    [(AnthropicClient, "anthropic"), (OpenAIClient, "openai"), (GroqClient, "groq")],
)
def test_client_constructs(client_cls, sdk) -> None:
    pytest.importorskip(sdk)

    client = client_cls(api_key="test-key")

    assert client.model
    assert client.client is not None
