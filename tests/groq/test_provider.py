from unittest.mock import Mock, patch

import pytest

from clientai.groq.provider import Provider


class MockChoice:
    def __init__(self, content):
        self.message = Mock()
        self.message.content = content
        self.index = 0
        self.logprobs = None
        self.finish_reason = "stop"


class MockDeltaChoice:
    def __init__(self, content):
        self.delta = Mock()
        self.delta.content = content
        self.index = 0
        self.finish_reason = None


class MockResponse:
    def __init__(self, content):
        self.id = "test_id"
        self.object = "chat.completion"
        self.created = 1234567890
        self.model = "llama2-70b-4096"
        self.choices = [MockChoice(content)]
        self.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "queue_time": 0.1,
            "prompt_time": 0.2,
            "completion_time": 0.3,
            "total_time": 0.6,
        }
        self.system_fingerprint = "fp_123456"
        self.x_groq = {"id": "req_123456"}


class MockStreamChunk:
    def __init__(self, content):
        self.id = "test_id"
        self.object = "chat.completion.chunk"
        self.created = 1234567890
        self.model = "llama2-70b-4096"
        self.choices = [MockDeltaChoice(content)]
        self.system_fingerprint = "fp_123456"
        self.x_groq = {"id": "req_123456"}


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
    mock_response = MockResponse("This is a test response")
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
    mock_response = MockResponse("This is a test response")
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
        MockStreamChunk(chunk) for chunk in ["This ", "is ", "a ", "test"]
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


def test_generate_text_stream_full_response(mock_client, provider):
    chunks = ["This ", "is ", "a ", "test"]
    mock_stream = [MockStreamChunk(chunk) for chunk in chunks]
    mock_client.chat.completions.create.return_value = iter(mock_stream)

    result = provider.generate_text(
        "Test prompt",
        "llama2-70b-4096",
        stream=True,
        return_full_response=True,
    )

    assert list(result) == mock_stream
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama2-70b-4096",
        messages=[{"role": "user", "content": "Test prompt"}],
        stream=True,
    )


def test_chat_full_response(mock_client, provider):
    mock_response = MockResponse("This is a test response")
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
    mock_response = MockResponse("This is a test response")
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
        MockStreamChunk(chunk) for chunk in ["This ", "is ", "a ", "test"]
    ]
    mock_client.chat.completions.create.return_value = iter(mock_stream)

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(messages, "llama2-70b-4096", stream=True)

    assert list(result) == ["This ", "is ", "a ", "test"]
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama2-70b-4096", messages=messages, stream=True
    )


def test_chat_stream_full_response(mock_client, provider):
    chunks = ["This ", "is ", "a ", "test"]
    mock_stream = [MockStreamChunk(chunk) for chunk in chunks]
    mock_client.chat.completions.create.return_value = iter(mock_stream)

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(
        messages, "llama2-70b-4096", stream=True, return_full_response=True
    )

    assert list(result) == mock_stream
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama2-70b-4096", messages=messages, stream=True
    )


def test_import_error():
    with patch("clientai.groq.provider.GROQ_INSTALLED", False):
        with pytest.raises(ImportError) as exc_info:
            Provider(api_key="test_api_key")
        assert "The groq package is not installed" in str(exc_info.value)
