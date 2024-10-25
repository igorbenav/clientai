from unittest.mock import Mock, patch

import pytest

from clientai.exceptions import APIError, TimeoutError
from clientai.replicate.provider import Provider

VALID_MODEL = "owner/name"


@pytest.fixture
def mock_client():
    with patch("clientai.replicate.provider.Client") as mock:
        client = mock.return_value
        client.predictions = Mock()
        yield client


@pytest.fixture
def provider(mock_client):
    return Provider(api_key="test_api_key")


class MockPrediction:
    def __init__(self, id, status, output, error=None):
        self.id = id
        self.status = status
        self.output = output
        self.error = error

    def stream(self):
        if isinstance(self.output, list):
            yield from self.output
        else:
            yield self.output


def test_generate_text_full_response(mock_client, provider):
    mock_prediction = MockPrediction(
        id="test_id", status="succeeded", output="This is a test response"
    )
    mock_client.predictions.create.return_value = mock_prediction
    mock_client.predictions.get.return_value = mock_prediction

    result = provider.generate_text(
        "Test prompt", VALID_MODEL, return_full_response=True
    )

    assert isinstance(result, dict)
    assert result["output"] == "This is a test response"
    mock_client.predictions.create.assert_called_once_with(
        model=VALID_MODEL, input={"prompt": "Test prompt"}, stream=False
    )


def test_generate_text_content_only(mock_client, provider):
    mock_prediction = MockPrediction(
        id="test_id", status="succeeded", output="This is a test response"
    )
    mock_client.predictions.create.return_value = mock_prediction
    mock_client.predictions.get.return_value = mock_prediction

    result = provider.generate_text(
        "Test prompt", VALID_MODEL, return_full_response=False
    )

    assert result == "This is a test response"
    mock_client.predictions.create.assert_called_once_with(
        model=VALID_MODEL, input={"prompt": "Test prompt"}, stream=False
    )


def test_generate_text_stream(mock_client, provider):
    mock_prediction = MockPrediction(
        id="test_id", status="succeeded", output=["This ", "is ", "a ", "test"]
    )
    mock_client.predictions.create.return_value = mock_prediction

    result = provider.generate_text("Test prompt", VALID_MODEL, stream=True)

    assert list(result) == ["This ", "is ", "a ", "test"]
    mock_client.predictions.create.assert_called_once_with(
        model=VALID_MODEL, input={"prompt": "Test prompt"}, stream=True
    )


def test_chat_full_response(mock_client, provider):
    mock_prediction = MockPrediction(
        id="test_id", status="succeeded", output="This is a test response"
    )
    mock_client.predictions.create.return_value = mock_prediction
    mock_client.predictions.get.return_value = mock_prediction

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(messages, VALID_MODEL, return_full_response=True)

    assert isinstance(result, dict)
    assert result["output"] == "This is a test response"
    mock_client.predictions.create.assert_called_once_with(
        model=VALID_MODEL,
        input={"prompt": "user: Test message\nassistant: "},
        stream=False,
    )


def test_chat_content_only(mock_client, provider):
    mock_prediction = MockPrediction(
        id="test_id", status="succeeded", output="This is a test response"
    )
    mock_client.predictions.create.return_value = mock_prediction
    mock_client.predictions.get.return_value = mock_prediction

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(messages, VALID_MODEL, return_full_response=False)

    assert result == "This is a test response"
    mock_client.predictions.create.assert_called_once_with(
        model=VALID_MODEL,
        input={"prompt": "user: Test message\nassistant: "},
        stream=False,
    )


def test_chat_stream(mock_client, provider):
    mock_prediction = MockPrediction(
        id="test_id", status="succeeded", output=["This ", "is ", "a ", "test"]
    )
    mock_client.predictions.create.return_value = mock_prediction

    messages = [{"role": "user", "content": "Test message"}]
    result = provider.chat(messages, VALID_MODEL, stream=True)

    assert list(result) == ["This ", "is ", "a ", "test"]
    mock_client.predictions.create.assert_called_once_with(
        model=VALID_MODEL,
        input={"prompt": "user: Test message\nassistant: "},
        stream=True,
    )


def test_wait_for_prediction_timeout(mock_client, provider):
    mock_prediction = MockPrediction(
        id="test_id", status="processing", output=None
    )
    mock_client.predictions.get.return_value = mock_prediction

    with pytest.raises(TimeoutError) as exc_info:
        provider._wait_for_prediction("test_id", max_wait_time=1)

    assert "Prediction timed out" in str(exc_info.value)
    assert exc_info.value.status_code == 408


def test_wait_for_prediction_failure(mock_client, provider):
    mock_prediction = MockPrediction(
        id="test_id", status="failed", output=None, error="Test error"
    )
    mock_client.predictions.get.return_value = mock_prediction

    with pytest.raises(APIError) as exc_info:
        provider._wait_for_prediction("test_id")
    assert "Prediction failed: Test error" in str(exc_info.value)
