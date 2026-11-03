"""Approved routes reject paid targets before provider resolution."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses"])
@pytest.mark.parametrize("stream", [True, False])
def test_forbidden_model_never_resolves_provider(path, stream):
    settings = Settings(
        model="nvidia_nim/approved",
        allowed_models=("nvidia_nim/approved",),
    )
    payload = {"model": "openai/gpt-6-luna", "stream": stream}
    if path == "/v1/messages":
        payload.update(messages=[{"role": "user", "content": "hello"}], max_tokens=32)
    else:
        payload.update(input="hello", max_output_tokens=32)
    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            side_effect=AssertionError("blocked route reached provider resolution"),
        ) as resolve,
        TestClient(create_test_app(settings)) as client,
    ):
        response = client.post(path, json=payload)
    assert response.status_code == 400
    resolve.assert_not_called()
