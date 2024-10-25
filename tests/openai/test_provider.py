from unittest.mock import Mock, patch

import pytest

from clientai.openai._typing import (
    Message,
    OpenAIChoice,
    OpenAIResponse,
    OpenAIStreamChoice,
    OpenAIStreamDelta,
    OpenAIStreamResponse,
)
from clientai.openai.provider import Provider


@pytest.fixture
def mock_client():
    with patch("clientai.openai.provider.Client") as mock:
        client = mock.return_value
        client.chat = Mock()
        client.chat.completions = Mock()
        yield client


@pytest.fixture
def provider(mock_client):
    return Provider(api_key="test_api_key")


def test_generate_text_full_response(mock_client, provider):
    mock_response = OpenAIResponse(
        id="test_id",
        object="chat.completion",
        created=1234567890,
        model="gpt-3.5-turbo",
        choices=[
            OpenAIChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content="This is a test response",
                ),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    )
    mock_client.chat.completions.create.return_value = mock_response

    result = provider.generate_text(
        "Test prompt", "gpt-3.5-turbo", return_full_response=True
    )

    assert result == mock_response
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        stream=False,
    )


def test_generate_text_content_only(mock_client, provider):
    mock_response = OpenAIResponse(
        id="test_id",
        object="chat.completion",
        created=1234567890,
        model="gpt-3.5-turbo",
        choices=[
            OpenAIChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content="This is a test response",
                ),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    )
    mock_client.chat.completions.create.return_value = mock_response

    result = provider.generate_text(
        "Test prompt", "gpt-3.5-turbo", return_full_response=False
    )

    assert result == "This is a test response"
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        stream=False,
    )


def test_generate_text_stream(mock_client, provider):
    mock_stream = [
        OpenAIStreamResponse(
            id="test_id",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIStreamDelta(
                        role=None, content=chunk, function_call=None
                    ),
                    finish_reason=None,
                )
            ],
        )
        for chunk in ["This ", "is ", "a ", "test"]
    ]
    mock_client.chat.completions.create.return_value = iter(mock_stream)

    result = provider.generate_text(
        "Test prompt", "gpt-3.5-turbo", stream=True
    )

    assert list(result) == ["This ", "is ", "a ", "test"]
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test prompt"}],
        stream=True,
    )


def test_chat_full_response(mock_client, provider):
    mock_response = OpenAIResponse(
        id="test_id",
        object="chat.completion",
        created=1234567890,
        model="gpt-3.5-turbo",
        choices=[
            OpenAIChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content="This is a test response",
                ),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    )
    mock_client.chat.completions.create.return_value = mock_response

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(
        messages, "gpt-3.5-turbo", return_full_response=True
    )

    assert result == mock_response
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo", messages=messages, stream=False
    )


def test_chat_content_only(mock_client, provider):
    mock_response = OpenAIResponse(
        id="test_id",
        object="chat.completion",
        created=1234567890,
        model="gpt-3.5-turbo",
        choices=[
            OpenAIChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content="This is a test response",
                ),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    )
    mock_client.chat.completions.create.return_value = mock_response

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(
        messages, "gpt-3.5-turbo", return_full_response=False
    )

    assert result == "This is a test response"
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo", messages=messages, stream=False
    )


def test_chat_stream(mock_client, provider):
    mock_stream = [
        OpenAIStreamResponse(
            id="test_id",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIStreamDelta(
                        role=None, content=chunk, function_call=None
                    ),
                    finish_reason=None,
                )
            ],
        )
        for chunk in ["This ", "is ", "a ", "test"]
    ]
    mock_client.chat.completions.create.return_value = iter(mock_stream)

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(messages, "gpt-3.5-turbo", stream=True)

    assert list(result) == ["This ", "is ", "a ", "test"]
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-3.5-turbo", messages=messages, stream=True
    )


def test_import_error():
    with patch("clientai.openai.provider.OPENAI_INSTALLED", False):
        with pytest.raises(ImportError) as exc_info:
            Provider(api_key="test_api_key")
        assert "The openai package is not installed" in str(exc_info.value)
