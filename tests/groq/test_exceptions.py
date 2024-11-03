from unittest.mock import MagicMock, patch

import httpx
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
from clientai.groq.provider import (
    GroqAuthenticationError,
    GroqRateLimitError,
    Provider,
)


@pytest.fixture
def provider():
    return Provider(api_key="test_key")


@pytest.fixture
def mock_groq_client():
    with patch("clientai.groq.provider.Client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.chat.completions.create.return_value = MagicMock()
        yield mock_instance


class MockResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code
        self.request = httpx.Request(
            "GET", "https://api.groq.com/v1/chat/completions"
        )


def test_generate_text_authentication_error(mock_groq_client, provider):
    error = GroqAuthenticationError(
        message="Invalid API key: test_key",
        response=MockResponse(401),
        body=None,
    )
    mock_groq_client.chat.completions.create.side_effect = error

    with pytest.raises(AuthenticationError) as exc_info:
        provider.generate_text("Test prompt", "llama3-8b-8192")

    assert "Invalid API key" in str(exc_info.value)
    assert exc_info.value.status_code == 401
    assert isinstance(exc_info.value.original_error, GroqAuthenticationError)


def test_generate_text_rate_limit_error(mock_groq_client, provider):
    error = GroqRateLimitError(
        message="Rate limit exceeded", response=MockResponse(429), body=None
    )
    mock_groq_client.chat.completions.create.side_effect = error

    with pytest.raises(RateLimitError) as exc_info:
        provider.generate_text("Test prompt", "llama3-8b-8192")

    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.status_code == 429
    assert isinstance(exc_info.value.original_error, GroqRateLimitError)


def test_generate_text_model_error(mock_groq_client, provider):
    from groq import NotFoundError

    error = NotFoundError(
        message="Model not found", response=MockResponse(404), body=None
    )
    mock_groq_client.chat.completions.create.side_effect = error

    with pytest.raises(ModelError) as exc_info:
        provider.generate_text("Test prompt", "invalid-model")

    assert "Model not found" in str(exc_info.value)
    assert exc_info.value.status_code == 404


def test_generate_text_invalid_request_error(mock_groq_client, provider):
    from groq import BadRequestError

    error = BadRequestError(
        message="Invalid request parameters",
        response=MockResponse(400),
        body=None,
    )
    mock_groq_client.chat.completions.create.side_effect = error

    with pytest.raises(InvalidRequestError) as exc_info:
        provider.generate_text("Test prompt", "llama3-8b-8192")

    assert "Invalid request parameters" in str(exc_info.value)
    assert exc_info.value.status_code == 400


def test_generate_text_timeout_error(mock_groq_client, provider):
    from groq import APITimeoutError

    error = APITimeoutError(
        request=httpx.Request("GET", "https://api.groq.com")
    )
    mock_groq_client.chat.completions.create.side_effect = error

    with pytest.raises(TimeoutError) as exc_info:
        provider.generate_text("Test prompt", "llama3-8b-8192")

    assert "Request timed out" in str(exc_info.value)
    assert exc_info.value.status_code == 408


def test_generate_text_api_error(mock_groq_client, provider):
    from groq import InternalServerError

    error = InternalServerError(
        message="Internal server error", response=MockResponse(500), body=None
    )
    mock_groq_client.chat.completions.create.side_effect = error

    with pytest.raises(APIError) as exc_info:
        provider.generate_text("Test prompt", "llama3-8b-8192")

    assert "Internal server error" in str(exc_info.value)
    assert exc_info.value.status_code == 500


def test_chat_error(mock_groq_client, provider):
    from groq import BadRequestError

    error = BadRequestError(
        message="Invalid request parameters",
        response=MockResponse(400),
        body=None,
    )
    mock_groq_client.chat.completions.create.side_effect = error

    with pytest.raises(InvalidRequestError) as exc_info:
        provider.chat(
            [{"role": "user", "content": "Test message"}], "llama3-8b-8192"
        )

    assert "Invalid request parameters" in str(exc_info.value)
    assert exc_info.value.status_code == 400


def test_generic_error(mock_groq_client, provider):
    mock_groq_client.chat.completions.create.side_effect = Exception(
        "Unexpected error"
    )

    with pytest.raises(ClientAIError) as exc_info:
        provider.generate_text("Test prompt", "llama3-8b-8192")

    assert "Unexpected error" in str(exc_info.value)
    assert isinstance(exc_info.value.original_error, Exception)
