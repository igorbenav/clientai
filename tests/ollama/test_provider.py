from unittest.mock import patch

import pytest

from clientai.ollama._typing import OllamaChatResponse, OllamaResponse
from clientai.ollama.provider import Provider


@pytest.fixture(autouse=True)
def reset_mocks():
    with patch("clientai.ollama.provider.ollama") as ollama_mock:
        yield ollama_mock


@pytest.fixture
def provider():
    return Provider()


def test_generate_text_full_response(reset_mocks, provider):
    mock_response = OllamaResponse(
        model="test-model",
        created_at="2023-01-01T00:00:00Z",
        response="This is a test response",
        done=True,
        context=None,
        total_duration=100,
        load_duration=10,
        prompt_eval_count=5,
        prompt_eval_duration=20,
        eval_count=15,
        eval_duration=70,
        done_reason="completed",
    )
    reset_mocks.generate.return_value = mock_response

    result = provider.generate_text(
        "Test prompt", "test-model", return_full_response=True
    )

    assert result == mock_response
    reset_mocks.generate.assert_called_once_with(
        model="test-model", 
        prompt="Test prompt", 
        stream=False,
        options={},
    )


def test_generate_text_content_only(reset_mocks, provider):
    mock_response = OllamaResponse(
        model="test-model",
        created_at="2023-01-01T00:00:00Z",
        response="This is a test response",
        done=True,
        context=None,
        total_duration=100,
        load_duration=10,
        prompt_eval_count=5,
        prompt_eval_duration=20,
        eval_count=15,
        eval_duration=70,
        done_reason="completed",
    )
    reset_mocks.generate.return_value = mock_response

    result = provider.generate_text(
        "Test prompt", "test-model", return_full_response=False
    )

    assert result == "This is a test response"
    reset_mocks.generate.assert_called_once_with(
        model="test-model", 
        prompt="Test prompt", 
        stream=False,
        options={},
    )


def test_generate_text_stream(reset_mocks, provider):
    mock_stream = [
        {"response": "This ", "done": False},
        {"response": "is ", "done": False},
        {"response": "a ", "done": False},
        {"response": "test", "done": True},
    ]
    reset_mocks.generate.return_value = iter(mock_stream)

    result = provider.generate_text("Test prompt", "test-model", stream=True)

    assert list(result) == ["This ", "is ", "a ", "test"]
    reset_mocks.generate.assert_called_once_with(
        model="test-model", 
        prompt="Test prompt", 
        stream=True,
        options={},
    )


def test_chat_full_response(reset_mocks, provider):
    mock_response = OllamaChatResponse(
        model="test-model",
        created_at="2023-01-01T00:00:00Z",
        message={"role": "assistant", "content": "This is a test response"},
        done=True,
        total_duration=100,
        load_duration=10,
        prompt_eval_count=5,
        prompt_eval_duration=20,
        eval_count=15,
        eval_duration=70,
    )
    reset_mocks.chat.return_value = mock_response

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(messages, "test-model", return_full_response=True)

    assert result == mock_response
    reset_mocks.chat.assert_called_once_with(
        model="test-model", 
        messages=messages, 
        stream=False,
        options={},
    )


def test_chat_content_only(reset_mocks, provider):
    mock_response = OllamaChatResponse(
        model="test-model",
        created_at="2023-01-01T00:00:00Z",
        message={"role": "assistant", "content": "This is a test response"},
        done=True,
        total_duration=100,
        load_duration=10,
        prompt_eval_count=5,
        prompt_eval_duration=20,
        eval_count=15,
        eval_duration=70,
    )
    reset_mocks.chat.return_value = mock_response

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(messages, "test-model", return_full_response=False)

    assert result == "This is a test response"
    reset_mocks.chat.assert_called_once_with(
        model="test-model", 
        messages=messages, 
        stream=False,
        options={},
    )


def test_chat_stream(reset_mocks, provider):
    mock_stream = [
        {"message": {"role": "assistant", "content": "This "}, "done": False},
        {"message": {"role": "assistant", "content": "is "}, "done": False},
        {"message": {"role": "assistant", "content": "a "}, "done": False},
        {"message": {"role": "assistant", "content": "test"}, "done": True},
    ]
    reset_mocks.chat.return_value = iter(mock_stream)

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(messages, "test-model", stream=True)

    assert list(result) == ["This ", "is ", "a ", "test"]
    reset_mocks.chat.assert_called_once_with(
        model="test-model", 
        messages=messages, 
        stream=True,
        options={},
    )


def test_generate_text_with_options(reset_mocks, provider):
    mock_response = OllamaResponse(
        model="test-model",
        created_at="2023-01-01T00:00:00Z",
        response="This is a test response",
        done=True,
        context=None,
        total_duration=100,
        load_duration=10,
        prompt_eval_count=5,
        prompt_eval_duration=20,
        eval_count=15,
        eval_duration=70,
        done_reason="completed",
    )
    reset_mocks.generate.return_value = mock_response

    result = provider.generate_text(
        "Test prompt",
        "test-model",
        system_prompt="System prompt",
        temperature=0.7,
        top_p=0.9,
        json_output=True,
    )

    assert result == "This is a test response"
    reset_mocks.generate.assert_called_once_with(
        model="test-model",
        prompt="Test prompt",
        stream=False,
        options={
            "system": "System prompt",
            "temperature": 0.7,
            "top_p": 0.9,
            "format": "json",
        },
    )


def test_chat_with_options(reset_mocks, provider):
    mock_response = OllamaChatResponse(
        model="test-model",
        created_at="2023-01-01T00:00:00Z",
        message={"role": "assistant", "content": "This is a test response"},
        done=True,
        total_duration=100,
        load_duration=10,
        prompt_eval_count=5,
        prompt_eval_duration=20,
        eval_count=15,
        eval_duration=70,
    )
    reset_mocks.chat.return_value = mock_response

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(
        messages,
        "test-model",
        system_prompt="System prompt",
        temperature=0.7,
        top_p=0.9,
        json_output=True,
    )

    expected_messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Test message"},
    ]

    assert result == "This is a test response"
    reset_mocks.chat.assert_called_once_with(
        model="test-model",
        messages=expected_messages,
        stream=False,
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "format": "json",
        },
    )
