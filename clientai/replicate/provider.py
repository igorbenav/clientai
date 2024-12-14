import time
from collections.abc import Iterator
from typing import Any, List, Optional, Union, cast

from .._common_types import Message
from ..ai_provider import AIProvider
from ..exceptions import (
    APIError,
    AuthenticationError,
    ClientAIError,
    InvalidRequestError,
    ModelError,
    RateLimitError,
    TimeoutError,
)
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
    from replicate.exceptions import ReplicateError

    Client = replicate.Client
else:
    Client = None  # type: ignore
    ReplicateError = Exception  # type: ignore


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

    Example:
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

    def _validate_temperature(self, temperature: Optional[float]) -> None:
        """Validate the temperature parameter."""
        if temperature is not None:
            if not isinstance(temperature, (int, float)):  # noqa: UP038
                raise InvalidRequestError("Temperature must be a number")

    def _validate_top_p(self, top_p: Optional[float]) -> None:
        """Validate the top_p parameter."""
        if top_p is not None:
            if not isinstance(top_p, (int, float)):  # noqa: UP038
                raise InvalidRequestError(
                    "Top-p must be a number between 0 and 1"
                )
            if top_p < 0 or top_p > 1:
                raise InvalidRequestError(
                    f"Top-p must be between 0 and 1, got {top_p}"
                )

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
            APIError: If the prediction fails.
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            prediction = self.client.predictions.get(prediction_id)
            if prediction.status == "succeeded":
                return prediction
            elif prediction.status == "failed":
                raise self._map_exception_to_clientai_error(
                    Exception(f"Prediction failed: {prediction.error}")
                )
            time.sleep(1)

        raise self._map_exception_to_clientai_error(
            Exception("Prediction timed out"), status_code=408
        )

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

    def _map_exception_to_clientai_error(
        self, e: Exception, status_code: Optional[int] = None
    ) -> ClientAIError:
        """
        Maps a Replicate exception to the appropriate ClientAI exception.

        Args:
            e (Exception): The exception caught during the API call.
            status_code (int, optional): The HTTP status code, if available.

        Returns:
            ClientAIError: An instance of the appropriate ClientAI exception.
        """
        error_message = str(e)
        status_code = status_code or getattr(e, "status_code", None)

        if (
            "authentication" in error_message.lower()
            or "unauthorized" in error_message.lower()
        ):
            return AuthenticationError(
                error_message, status_code, original_error=e
            )
        elif "rate limit" in error_message.lower():
            return RateLimitError(error_message, status_code, original_error=e)
        elif "not found" in error_message.lower():
            return ModelError(error_message, status_code, original_error=e)
        elif "invalid" in error_message.lower():
            return InvalidRequestError(
                error_message, status_code, original_error=e
            )
        elif "timeout" in error_message.lower() or status_code == 408:
            return TimeoutError(error_message, status_code, original_error=e)
        elif status_code == 400:
            return InvalidRequestError(
                error_message, status_code, original_error=e
            )
        else:
            return APIError(error_message, status_code, original_error=e)

    def generate_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> ReplicateGenericResponse:
        """
        Generate text based on a given prompt
        using a specified Replicate model.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the Replicate model to use.
            system_prompt: Optional system prompt to guide model behavior.
                      If provided, will be added as a system message before
                      the prompt.
            return_full_response: If True, return the full response object.
                If False, return only the generated text.
            stream: If True, return an iterator for streaming responses.
            json_output: If True, set output="json" in the input parameters
                to get JSON-formatted responses. The prompt should specify
                the desired JSON structure.
            **kwargs: Additional keyword arguments to pass
                      to the Replicate API.

        Returns:
            ReplicateGenericResponse: The generated text, full response object,
            or an iterator for streaming responses.

        Example:
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

            Generate JSON output:
            ```python
            response = provider.generate_text(
                '''Create a user profile with:
                {
                    "name": "A random name",
                    "age": "A random age between 20-80",
                    "occupation": "A random occupation"
                }''',
                model="meta/llama-2-70b-chat:latest",
                json_output=True
            )
            print(response)  # Will be JSON formatted
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
        try:
            self._validate_temperature(temperature)
            self._validate_top_p(top_p)

            formatted_prompt = ""
            if system_prompt:
                formatted_prompt = f"<system>{system_prompt}</system>\n"
            formatted_prompt += f"<user>{prompt}</user>\n<assistant>"

            input_params = {"prompt": formatted_prompt}
            if json_output:
                input_params["output"] = "json"
            if temperature is not None:
                input_params["temperature"] = temperature  # type: ignore
            if top_p is not None:
                input_params["top_p"] = top_p  # type: ignore

            prediction = self.client.predictions.create(
                model=model,
                input=input_params,
                stream=stream,
                **kwargs,
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

        except Exception as e:
            raise self._map_exception_to_clientai_error(e)

    def chat(
        self,
        messages: List[Message],
        model: str,
        system_prompt: Optional[str] = None,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> ReplicateGenericResponse:
        """
        Engage in a chat conversation using a specified Replicate model.

        Args:
            messages: A list of message dictionaries, each containing
                      'role' and 'content'.
            model: The name or identifier of the Replicate model to use.
            system_prompt: Optional system prompt to guide model behavior.
                      If provided, will be inserted at the start of the
                      conversation.
            return_full_response: If True, return the full response object.
                If False, return only the generated text.
            stream: If True, return an iterator for streaming responses.
            json_output: If True, set output="json" in the input parameters
                to get JSON-formatted responses. The messages should specify
                the desired JSON structure.
            **kwargs: Additional keyword arguments to pass
                      to the Replicate API.

        Returns:
            ReplicateGenericResponse: The chat response, full response object,
            or an iterator for streaming responses.

        Examples:
            Chat (message content only):
            ```python
            messages = [
                {"role": "user", "content": "What is quantum computing?"},
                {"role": "assistant", "content": "Quantum computing uses..."},
                {"role": "user", "content": "What are its applications?"}
            ]
            response = provider.chat(
                messages,
                model="meta/llama-2-70b-chat:latest",
            )
            print(response)
            ```

            Chat with JSON output:
            ```python
            messages = [
                {"role": "user", "content": '''Create a user profile with:
                {
                    "name": "A random name",
                    "age": "A random age between 20-80",
                    "occupation": "A random occupation"
                }'''}
            ]
            response = provider.chat(
                messages,
                model="meta/llama-2-70b-chat:latest",
                json_output=True
            )
            print(response)  # Will be JSON formatted
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
        try:
            self._validate_temperature(temperature)
            self._validate_top_p(top_p)

            chat_messages = messages.copy()
            if system_prompt:
                chat_messages.insert(
                    0, {"role": "system", "content": system_prompt}
                )

            prompt = "\n".join(
                [
                    f"<{m['role']}>{m['content']}</{m['role']}>"
                    for m in chat_messages
                ]
            )
            prompt += "\n<assistant>"

            input_params = {"prompt": prompt}
            if json_output:
                input_params["output"] = "json"
            if temperature is not None:
                input_params["temperature"] = temperature  # type: ignore
            if top_p is not None:
                input_params["top_p"] = top_p  # type: ignore

            prediction = self.client.predictions.create(
                model=model,
                input=input_params,
                stream=stream,
                **kwargs,
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

        except Exception as e:
            raise self._map_exception_to_clientai_error(e)
