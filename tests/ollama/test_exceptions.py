from unittest.mock import patch

import pytest

from clientai.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    ModelError,
    RateLimitError,
    TimeoutError,
)
from clientai.ollama.provider import Provider


@pytest.fixture
def provider():
    return Provider()


@pytest.fixture(autouse=True)
def mock_ollama():
    with patch("clientai.ollama.provider.ollama") as mock:
        mock.RequestError = type("RequestError", (Exception,), {})
        mock.ResponseError = type("ResponseError", (Exception,), {})
        yield mock


@pytest.fixture
def valid_chat_request():
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Test message"}],
        "stream": False,
        "format": "",
        "options": None,
        "keep_alive": None,
    }


def test_generate_text_authentication_error(mock_ollama, provider):
    error = mock_ollama.RequestError("Authentication failed")
    mock_ollama.generate.side_effect = error

    with pytest.raises(AuthenticationError) as exc_info:
        provider.generate_text(prompt="Test prompt", model="test-model")

    assert str(exc_info.value) == "Authentication failed"
    assert exc_info.value.original_exception is error


def test_generate_text_rate_limit_error(mock_ollama, provider):
    error = mock_ollama.RequestError("Rate limit exceeded")
    mock_ollama.generate.side_effect = error

    with pytest.raises(RateLimitError) as exc_info:
        provider.generate_text(prompt="Test prompt", model="test-model")

    assert str(exc_info.value) == "Rate limit exceeded"
    assert exc_info.value.original_exception is error


def test_generate_text_model_error(mock_ollama, provider):
    error = mock_ollama.RequestError("Model not found")
    mock_ollama.generate.side_effect = error

    with pytest.raises(ModelError) as exc_info:
        provider.generate_text(prompt="Test prompt", model="test-model")

    assert str(exc_info.value) == "Model not found"
    assert exc_info.value.original_exception is error


def test_generate_text_invalid_request_error(mock_ollama, provider):
    error = mock_ollama.RequestError("Invalid request")
    mock_ollama.generate.side_effect = error

    with pytest.raises(InvalidRequestError) as exc_info:
        provider.generate_text(prompt="Test prompt", model="test-model")

    assert str(exc_info.value) == "Invalid request"
    assert exc_info.value.original_exception is error


def test_generate_text_timeout_error(mock_ollama, provider):
    error = mock_ollama.ResponseError("Request timed out")
    mock_ollama.generate.side_effect = error

    with pytest.raises(TimeoutError) as exc_info:
        provider.generate_text(prompt="Test prompt", model="test-model")

    assert str(exc_info.value) == "Request timed out"
    assert exc_info.value.original_exception is error


def test_generate_text_api_error(mock_ollama, provider):
    error = mock_ollama.ResponseError("API response error")
    mock_ollama.generate.side_effect = error

    with pytest.raises(APIError) as exc_info:
        provider.generate_text(prompt="Test prompt", model="test-model")

    assert str(exc_info.value) == "API response error"
    assert exc_info.value.original_exception is error


def test_chat_request_error(mock_ollama, provider, valid_chat_request):
    error = mock_ollama.RequestError("Invalid chat request")
    mock_ollama.chat.side_effect = error

    with pytest.raises(InvalidRequestError) as exc_info:
        provider.chat(**valid_chat_request)

    assert str(exc_info.value) == "Invalid chat request"
    assert exc_info.value.original_exception is error


def test_chat_response_error(mock_ollama, provider, valid_chat_request):
    error = mock_ollama.ResponseError("Chat API response error")
    mock_ollama.chat.side_effect = error

    with pytest.raises(APIError) as exc_info:
        provider.chat(**valid_chat_request)

    assert str(exc_info.value) == "Chat API response error"
    assert exc_info.value.original_exception is error
