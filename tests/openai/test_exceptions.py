from unittest.mock import MagicMock, patch

import pytest

from clientai.exceptions import (
    APIError,
    AuthenticationError,
    ClientAIError,
    InvalidRequestError,
    ModelError,
    RateLimitError,
    TimeoutError,
)
from clientai.openai.provider import Provider


class MockOpenAIError(Exception):
    def __init__(self, message, error_type, status_code):
        self.message = message
        self.type = error_type
        self.status_code = status_code

    def __str__(self):
        return f"Error code: {self.status_code} - {self.message}"


class MockOpenAIAuthenticationError(MockOpenAIError):
    def __init__(self, message, status_code=401):
        super().__init__(message, "invalid_request_error", status_code)


@pytest.fixture
def provider():
    return Provider(api_key="test_key")


@pytest.fixture
def mock_openai_client():
    with patch("clientai.openai.provider.Client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        mock_instance.with_api_key.return_value = mock_instance
        mock_instance.chat.completions.create.return_value = MagicMock()

        yield mock_instance


def test_generate_text_authentication_error(mock_openai_client, provider):
    error = MockOpenAIAuthenticationError(
        "Incorrect API key provided: test_key. You can find your API key at https://platform.openai.com/account/api-keys."
    )
    mock_openai_client.chat.completions.create.side_effect = error

    with pytest.raises(AuthenticationError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Incorrect API key provided" in str(exc_info.value)
    assert exc_info.value.status_code == 401
    assert isinstance(
        exc_info.value.original_error, MockOpenAIAuthenticationError
    )


def test_generate_text_rate_limit_error(mock_openai_client, provider):
    error = MockOpenAIError("Rate limit exceeded", "rate_limit_error", 429)
    mock_openai_client.chat.completions.create.side_effect = error

    with pytest.raises(RateLimitError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.status_code == 429
    assert isinstance(exc_info.value.original_error, MockOpenAIError)


def test_generate_text_model_error(mock_openai_client, provider):
    error = MockOpenAIError("Model not found", "model_not_found", 404)
    mock_openai_client.chat.completions.create.side_effect = error

    with pytest.raises(ModelError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Model not found" in str(exc_info.value)
    assert exc_info.value.status_code == 404
    assert isinstance(exc_info.value.original_error, MockOpenAIError)


def test_generate_text_invalid_request_error(mock_openai_client, provider):
    error = MockOpenAIError("Invalid request", "invalid_request_error", 400)
    mock_openai_client.chat.completions.create.side_effect = error

    with pytest.raises(InvalidRequestError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Invalid request" in str(exc_info.value)
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.original_error, MockOpenAIError)


def test_generate_text_timeout_error(mock_openai_client, provider):
    error = MockOpenAIError("Request timed out", "timeout_error", 408)
    mock_openai_client.chat.completions.create.side_effect = error

    with pytest.raises(TimeoutError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Request timed out" in str(exc_info.value)
    assert exc_info.value.status_code == 408
    assert isinstance(exc_info.value.original_error, MockOpenAIError)


def test_generate_text_api_error(mock_openai_client, provider):
    error = MockOpenAIError("API error", "api_error", 500)
    mock_openai_client.chat.completions.create.side_effect = error

    with pytest.raises(APIError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "API error" in str(exc_info.value)
    assert exc_info.value.status_code == 500
    assert isinstance(exc_info.value.original_error, MockOpenAIError)


def test_chat_error(mock_openai_client, provider):
    error = MockOpenAIError("Chat error", "invalid_request_error", 400)
    mock_openai_client.chat.completions.create.side_effect = error

    with pytest.raises(InvalidRequestError) as exc_info:
        provider.chat(
            [{"role": "user", "content": "Test message"}], "test-model"
        )

    assert "Chat error" in str(exc_info.value)
    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.original_error, MockOpenAIError)


def test_generic_error(mock_openai_client, provider):
    mock_openai_client.chat.completions.create.side_effect = Exception(
        "Unexpected error"
    )

    with pytest.raises(ClientAIError) as exc_info:
        provider.generate_text("Test prompt", "test-model")

    assert "Unexpected error" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, Exception)
