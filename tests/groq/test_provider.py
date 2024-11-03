from unittest.mock import Mock, patch

import pytest

from clientai.groq.provider import Provider


@pytest.fixture
def mock_client():
    with patch("clientai.groq.provider.Client") as mock:
        client = mock.return_value
        client.chat = Mock()
        client.chat.completions = Mock()
        yield client


@pytest.fixture
def provider(mock_client):
    return Provider(api_key="test_api_key")


def test_generate_text_full_response(mock_client, provider):
    mock_response = {
        "id": "test_id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama2-70b-4096",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "queue_time": 0.1,
            "prompt_time": 0.2,
            "completion_time": 0.3,
            "total_time": 0.6,
        },
        "system_fingerprint": "fp_123456",
        "x_groq": {"id": "req_123456"},
    }
    mock_client.chat.completions.create.return_value = mock_response

    result = provider.generate_text(
        "Test prompt", "llama2-70b-4096", return_full_response=True
    )

    assert result == mock_response
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama2-70b-4096",
        messages=[{"role": "user", "content": "Test prompt"}],
        stream=False,
    )


def test_generate_text_content_only(mock_client, provider):
    mock_response = {
        "id": "test_id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama2-70b-4096",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "queue_time": 0.1,
            "prompt_time": 0.2,
            "completion_time": 0.3,
            "total_time": 0.6,
        },
        "system_fingerprint": "fp_123456",
        "x_groq": {"id": "req_123456"},
    }
    mock_client.chat.completions.create.return_value = mock_response

    result = provider.generate_text(
        "Test prompt", "llama2-70b-4096", return_full_response=False
    )

    assert result == "This is a test response"
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama2-70b-4096",
        messages=[{"role": "user", "content": "Test prompt"}],
        stream=False,
    )


def test_generate_text_stream(mock_client, provider):
    mock_stream = [
        {
            "id": "test_id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "llama2-70b-4096",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": None,
                        "content": chunk,
                    },
                    "finish_reason": None,
                }
            ],
            "system_fingerprint": "fp_123456",
            "x_groq": {"id": "req_123456"},
        }
        for chunk in ["This ", "is ", "a ", "test"]
    ]
    mock_client.chat.completions.create.return_value = iter(mock_stream)

    result = provider.generate_text(
        "Test prompt", "llama2-70b-4096", stream=True
    )

    assert list(result) == ["This ", "is ", "a ", "test"]
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama2-70b-4096",
        messages=[{"role": "user", "content": "Test prompt"}],
        stream=True,
    )


def test_chat_full_response(mock_client, provider):
    mock_response = {
        "id": "test_id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama2-70b-4096",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "queue_time": 0.1,
            "prompt_time": 0.2,
            "completion_time": 0.3,
            "total_time": 0.6,
        },
        "system_fingerprint": "fp_123456",
        "x_groq": {"id": "req_123456"},
    }
    mock_client.chat.completions.create.return_value = mock_response

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(
        messages, "llama2-70b-4096", return_full_response=True
    )

    assert result == mock_response
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama2-70b-4096", messages=messages, stream=False
    )


def test_chat_content_only(mock_client, provider):
    mock_response = {
        "id": "test_id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama2-70b-4096",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "queue_time": 0.1,
            "prompt_time": 0.2,
            "completion_time": 0.3,
            "total_time": 0.6,
        },
        "system_fingerprint": "fp_123456",
        "x_groq": {"id": "req_123456"},
    }
    mock_client.chat.completions.create.return_value = mock_response

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(
        messages, "llama2-70b-4096", return_full_response=False
    )

    assert result == "This is a test response"
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama2-70b-4096", messages=messages, stream=False
    )


def test_chat_stream(mock_client, provider):
    mock_stream = [
        {
            "id": "test_id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "llama2-70b-4096",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": None,
                        "content": chunk,
                    },
                    "finish_reason": None,
                }
            ],
            "system_fingerprint": "fp_123456",
            "x_groq": {"id": "req_123456"},
        }
        for chunk in ["This ", "is ", "a ", "test"]
    ]
    mock_client.chat.completions.create.return_value = iter(mock_stream)

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(messages, "llama2-70b-4096", stream=True)

    assert list(result) == ["This ", "is ", "a ", "test"]
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama2-70b-4096", messages=messages, stream=True
    )


def test_import_error():
    with patch("clientai.groq.provider.GROQ_INSTALLED", False):
        with pytest.raises(ImportError) as exc_info:
            Provider(api_key="test_api_key")
        assert "The groq package is not installed" in str(exc_info.value)
