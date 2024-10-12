import time
from collections.abc import Iterator
from typing import Any, List, Union, cast

from .._common_types import Message
from ..ai_provider import AIProvider
from . import REPLICATE_INSTALLED
from ._typing import (
    ReplicateClientProtocol,
    ReplicateGenericResponse,
    ReplicatePredictionProtocol,
    ReplicateResponse,
    ReplicateStreamResponse,
)

if REPLICATE_INSTALLED:
    import replicate  # type: ignore

    Client = replicate.Client
else:
    Client = None  # type: ignore


class Provider(AIProvider):
    """
    Replicate-specific implementation of the AIProvider abstract base class.

    This class provides methods to interact with Replicate's AI models for
    text generation and chat functionality.

    Attributes:
        client: The Replicate client used for making API calls.

    Args:
        api_key: The API key for authenticating with Replicate.

    Raises:
        ImportError: If the Replicate package is not installed.

    Examples:
        Initialize the Replicate provider:
        ```python
        provider = Provider(api_key="your-replicate-api-key")
        ```
    """

    def __init__(self, api_key: str):
        if not REPLICATE_INSTALLED or Client is None:
            raise ImportError(
                "The replicate package is not installed. "
                "Please install it with 'pip install clientai[replicate]'."
            )
        self.client: ReplicateClientProtocol = Client(api_token=api_key)

    def _process_output(self, output: Any) -> str:
        """
        Process the output from Replicate API into a string format.

        Args:
            output: The raw output from Replicate API.

        Returns:
            str: The processed output as a string.
        """
        if isinstance(output, List):
            return "".join(str(item) for item in output)
        elif isinstance(output, str):
            return output
        else:
            return str(output)

    def _wait_for_prediction(
        self, prediction_id: str, max_wait_time: int = 300
    ) -> ReplicatePredictionProtocol:
        """
        Wait for a prediction to complete or fail.

        Args:
            prediction_id: The ID of the prediction to wait for.
            max_wait_time: Maximum time to wait in seconds. Defaults to 300.

        Returns:
            ReplicatePredictionProtocol: The completed prediction.

        Raises:
            TimeoutError: If the prediction doesn't complete within
                          the max_wait_time.
            Exception: If the prediction fails.
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            prediction = self.client.predictions.get(prediction_id)
            if prediction.status == "succeeded":
                return prediction
            elif prediction.status == "failed":
                raise Exception(f"Prediction failed: {prediction.error}")
            time.sleep(1)
        raise TimeoutError("Prediction timed out")

    def _stream_response(
        self,
        prediction: ReplicatePredictionProtocol,
        return_full_response: bool,
    ) -> Iterator[Union[str, ReplicateStreamResponse]]:
        """
        Stream the response from a prediction.

        Args:
            prediction: The prediction to stream.
            return_full_response: If True, yield full response objects.

        Yields:
            Union[str, ReplicateStreamResponse]: Processed output or
                                                 full response objects.
        """
        metadata = cast(ReplicateStreamResponse, prediction.__dict__.copy())
        for event in prediction.stream():
            if return_full_response:
                metadata["output"] = self._process_output(event)
                yield metadata
            else:
                yield self._process_output(event)

    def generate_text(
        self,
        prompt: str,
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> ReplicateGenericResponse:
        """
        Generate text based on a given prompt
        using a specified Replicate model.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the Replicate model to use.
            return_full_response: If True, return the full response object.
                If False, return only the generated text. Defaults to False.
            stream: If True, return an iterator for streaming responses.
                Defaults to False.
            **kwargs: Additional keyword arguments
                      to pass to the Replicate API.

        Returns:
            ReplicateGenericResponse: The generated text, full response object,
            or an iterator for streaming responses.

        Examples:
            Generate text (text only):
            ```python
            response = provider.generate_text(
                "Explain quantum computing",
                model="meta/llama-2-70b-chat:latest",
            )
            print(response)
            ```

            Generate text (full response):
            ```python
            response = provider.generate_text(
                "Explain quantum computing",
                model="meta/llama-2-70b-chat:latest",
                return_full_response=True
            )
            print(response["output"])
            ```

            Generate text (streaming):
            ```python
            for chunk in provider.generate_text(
                "Explain quantum computing",
                model="meta/llama-2-70b-chat:latest",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```
        """
        prediction = self.client.predictions.create(
            model=model, input={"prompt": prompt}, stream=stream, **kwargs
        )

        if stream:
            return self._stream_response(prediction, return_full_response)
        else:
            completed_prediction = self._wait_for_prediction(prediction.id)
            if return_full_response:
                response = cast(
                    ReplicateResponse, completed_prediction.__dict__.copy()
                )
                response["output"] = self._process_output(
                    completed_prediction.output
                )
                return response
            else:
                return self._process_output(completed_prediction.output)

    def chat(
        self,
        messages: List[Message],
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> ReplicateGenericResponse:
        """
        Engage in a chat conversation using a specified Replicate model.

        Args:
            messages: A list of message dictionaries, each containing
                      'role' and 'content'.
            model: The name or identifier of the Replicate model to use.
            return_full_response: If True, return the full response object.
                If False, return only the generated text. Defaults to False.
            stream: If True, return an iterator for streaming responses.
                Defaults to False.
            **kwargs: Additional keyword arguments
                      to pass to the Replicate API.

        Returns:
            ReplicateGenericResponse: The chat response, full response object,
            or an iterator for streaming responses.

        Examples:
            Chat (message content only):
            ```python
            messages = [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital is Paris."},
                {"role": "user", "content": "What is its population?"}
            ]
            response = provider.chat(
                messages,
                model="meta/llama-2-70b-chat:latest",
            )
            print(response)
            ```

            Chat (full response):
            ```python
            response = provider.chat(
                messages,
                model="meta/llama-2-70b-chat:latest",
                return_full_response=True
            )
            print(response["output"])
            ```

            Chat (streaming):
            ```python
            for chunk in provider.chat(
                messages,
                model="meta/llama-2-70b-chat:latest",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```
        """
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt += "\nassistant: "

        prediction = self.client.predictions.create(
            model=model, input={"prompt": prompt}, stream=stream, **kwargs
        )

        if stream:
            return self._stream_response(prediction, return_full_response)
        else:
            completed_prediction = self._wait_for_prediction(prediction.id)
            if return_full_response:
                response = cast(
                    ReplicateResponse, completed_prediction.__dict__.copy()
                )
                response["output"] = self._process_output(
                    completed_prediction.output
                )
                return response
            else:
                return self._process_output(completed_prediction.output)
