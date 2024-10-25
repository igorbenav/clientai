from unittest.mock import MagicMock, patch

import pytest

from clientai.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    ModelError,
    RateLimitError,
    TimeoutError,
)
from clientai.replicate.provider import Provider


class MockReplicateError(Exception):
    def __init__(self, message, status_code=None):
        self.message = message
        self.status_code = status_code

    def __str__(self):
        return self.message


@pytest.fixture
def provider():
    return Provider(api_key="test_key")


@pytest.fixture
def mock_replicate_client():
    with patch("clientai.replicate.provider.Client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance


def test_generate_text_authentication_error(mock_replicate_client, provider):
    error = MockReplicateError("Authentication failed", status_code=401)
    mock_replicate_client.predictions.create.side_effect = error

    with pytest.raises(AuthenticationError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Authentication failed" in str(exc_info.value)
    assert exc_info.value.status_code == 401
    assert isinstance(exc_info.value.original_error, MockReplicateError)


def test_generate_text_rate_limit_error(mock_replicate_client, provider):
    error = MockReplicateError("Rate limit exceeded", status_code=429)
    mock_replicate_client.predictions.create.side_effect = error

    with pytest.raises(RateLimitError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.status_code == 429
    assert isinstance(exc_info.value.original_error, MockReplicateError)


def test_generate_text_model_error(mock_replicate_client, provider):
    error = MockReplicateError("Model not found", status_code=404)
    mock_replicate_client.predictions.create.side_effect = error

    with pytest.raises(ModelError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Model not found" in str(exc_info.value)
    assert exc_info.value.status_code == 404
    assert isinstance(exc_info.value.original_error, MockReplicateError)


def test_generate_text_invalid_request_error(mock_replicate_client, provider):
    error = MockReplicateError("Invalid request", status_code=400)
    mock_replicate_client.predictions.create.side_effect = error

    with pytest.raises(InvalidRequestError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Invalid request" in str(exc_info.value)
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.original_error, MockReplicateError)


def test_generate_text_api_error(mock_replicate_client, provider):
    error = MockReplicateError("API error", status_code=500)
    mock_replicate_client.predictions.create.side_effect = error

    with pytest.raises(APIError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "API error" in str(exc_info.value)
    assert exc_info.value.status_code == 500
    assert isinstance(exc_info.value.original_error, MockReplicateError)


def test_generate_text_timeout_error(mock_replicate_client, provider):
    error = MockReplicateError("Request timed out", status_code=408)
    mock_replicate_client.predictions.create.side_effect = error

    with pytest.raises(TimeoutError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Request timed out" in str(exc_info.value)
    assert exc_info.value.status_code == 408
    assert isinstance(exc_info.value.original_error, MockReplicateError)


def test_chat_error(mock_replicate_client, provider):
    error = MockReplicateError("Chat error", status_code=400)
    mock_replicate_client.predictions.create.side_effect = error

    with pytest.raises(InvalidRequestError) as exc_info:
        provider.chat(
            [{"role": "user", "content": "Test message"}], "test-model"
        )

    assert "Chat error" in str(exc_info.value)
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.original_error, MockReplicateError)


def test_generic_error(mock_replicate_client, provider):
    error = Exception("Unexpected error")
    mock_replicate_client.predictions.create.side_effect = error

    with pytest.raises(APIError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Unexpected error" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, Exception)
