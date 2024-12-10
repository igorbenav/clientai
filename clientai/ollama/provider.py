from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Union, cast

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
from . import OLLAMA_INSTALLED
from ._typing import (
    Message,
    OllamaChatResponse,
    OllamaClientProtocol,
    OllamaGenericResponse,
    OllamaResponse,
    OllamaStreamResponse,
)

if OLLAMA_INSTALLED:
    import ollama  # type: ignore

    Client = ollama.Client
else:
    Client = None  # type: ignore


class Provider(AIProvider):
    """
    Ollama-specific implementation of the AIProvider abstract base class.

    This class provides methods to interact with Ollama's models for
    text generation and chat functionality.

    Attributes:
        client: The Ollama client used for making API calls.

    Args:
        host: The host address for the Ollama server.
            If not provided, the default Ollama client will be used.

    Raises:
        ImportError: If the Ollama package is not installed.

    Example:
        Initialize the Ollama provider:
        ```python
        provider = Provider(host="http://localhost:11434")
        ```
    """

    def __init__(self, host: Optional[str] = None):
        if not OLLAMA_INSTALLED or Client is None:
            raise ImportError(
                "The ollama package is not installed. "
                "Please install it with 'pip install clientai[ollama]'."
            )
        self.client: OllamaClientProtocol = cast(
            OllamaClientProtocol, Client(host=host) if host else ollama
        )

    def _prepare_options(
        self,
        json_output: bool = False,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Prepare the options dictionary for Ollama API calls.

        Args:
            json_output: If True, set format to "json"
            system_prompt: Optional system prompt
            temperature: Optional temperature value
            top_p: Optional top-p value
            **kwargs: Additional options to include

        Returns:
            Dict[str, Any]: The prepared options dictionary
        """
        options: Dict[str, Any] = {}

        if json_output:
            options["format"] = "json"
        if system_prompt:
            options["system"] = system_prompt
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p

        options.update(kwargs)

        return options

    def _validate_temperature(self, temperature: Optional[float]) -> None:
        """[previous implementation remains the same]"""
        if temperature is not None:
            if not isinstance(temperature, (int, float)):  # noqa: UP038
                raise InvalidRequestError(
                    "Temperature must be a number between 0 and 2"
                )
            if temperature < 0 or temperature > 2:
                raise InvalidRequestError(
                    f"Temperature must be between 0 and 2, got {temperature}"
                )

    def _validate_top_p(self, top_p: Optional[float]) -> None:
        """[previous implementation remains the same]"""
        if top_p is not None:
            if not isinstance(top_p, (int, float)):  # noqa: UP038
                raise InvalidRequestError(
                    "Top-p must be a number between 0 and 1"
                )
            if top_p < 0 or top_p > 1:
                raise InvalidRequestError(
                    f"Top-p must be between 0 and 1, got {top_p}"
                )

    def _stream_generate_response(
        self,
        stream: Iterator[OllamaStreamResponse],
        return_full_response: bool,
    ) -> Iterator[Union[str, OllamaStreamResponse]]:
        """
        Process the streaming response from Ollama API for text generation.

        Args:
            stream: The stream of responses from Ollama API.
            return_full_response: If True, yield full response objects.

        Yields:
            Union[str, OllamaStreamResponse]: Processed content or
                                              full response objects.
        """
        for chunk in stream:
            if return_full_response:
                yield chunk
            else:
                yield chunk["response"]

    def _stream_chat_response(
        self,
        stream: Iterator[OllamaChatResponse],
        return_full_response: bool,
    ) -> Iterator[Union[str, OllamaChatResponse]]:
        """
        Process the streaming response from Ollama API for chat.

        Args:
            stream: The stream of responses from Ollama API.
            return_full_response: If True, yield full response objects.

        Yields:
            Union[str, OllamaChatResponse]: Processed content or
                                            full response objects.
        """
        for chunk in stream:
            if return_full_response:
                yield chunk
            else:
                yield chunk["message"]["content"]

    def _map_exception_to_clientai_error(self, e: Exception) -> ClientAIError:
        """
        Maps an Ollama exception to the appropriate ClientAI exception.

        Args:
            e (Exception): The exception caught during the API call.

        Returns:
            ClientAIError: An instance of the appropriate ClientAI exception.
        """
        message = str(e)

        if isinstance(e, ollama.RequestError):
            if "authentication" in message.lower():
                return AuthenticationError(
                    message, status_code=401, original_error=e
                )
            elif "rate limit" in message.lower():
                return RateLimitError(
                    message, status_code=429, original_error=e
                )
            elif "not found" in message.lower():
                return ModelError(message, status_code=404, original_error=e)
            else:
                return InvalidRequestError(
                    message, status_code=400, original_error=e
                )
        elif isinstance(e, ollama.ResponseError):
            if "timeout" in message.lower() or "timed out" in message.lower():
                return TimeoutError(message, status_code=408, original_error=e)
            else:
                return APIError(message, status_code=500, original_error=e)
        else:
            return ClientAIError(message, status_code=500, original_error=e)

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
    ) -> OllamaGenericResponse:
        """
        Generate text based on a given prompt using a specified Ollama model.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the Ollama model to use.
            system_prompt: Optional system prompt to guide model behavior.
                           Uses Ollama's native system parameter.
            return_full_response: If True, return the full response object.
                If False, return only the generated text.
            stream: If True, return an iterator for streaming responses.
            json_output: If True, set format="json" to get JSON-formatted
                responses using Ollama's native JSON support. The prompt
                should specify the desired JSON structure.
            temperature: Optional temperature value for generation (0.0-2.0).
                Controls randomness in the output.
            top_p: Optional top-p value for nucleus sampling (0.0-1.0).
                Controls diversity of the output.
            **kwargs: Additional keyword arguments to pass to the Ollama API.

        Returns:
            OllamaGenericResponse: The generated text, full response object,
            or an iterator for streaming responses.

        Example:
            Generate text (text only):
            ```python
            response = provider.generate_text(
                "Explain machine learning",
                model="llama2",
            )
            print(response)
            ```

            Generate creative text with high temperature:
            ```python
            response = provider.generate_text(
                "Write a story about a space adventure",
                model="llama2",
                temperature=0.8,
                top_p=0.9
            )
            print(response)
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
                model="llama2",
                json_output=True
            )
            print(response)  # Will be JSON formatted
            ```
        """
        try:
            self._validate_temperature(temperature)
            self._validate_top_p(top_p)

            options = self._prepare_options(
                json_output=json_output,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

            response = self.client.generate(
                model=model,
                prompt=prompt,
                stream=stream,
                options=options,
            )

            if stream:
                return cast(
                    OllamaGenericResponse,
                    self._stream_generate_response(
                        cast(Iterator[OllamaStreamResponse], response),
                        return_full_response,
                    ),
                )
            else:
                response = cast(OllamaResponse, response)
                if return_full_response:
                    return response
                else:
                    return response["response"]

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
    ) -> OllamaGenericResponse:
        """
        Engage in a chat conversation using a specified Ollama model.

        Args:
            messages: A list of message dictionaries, each containing
                      'role' and 'content'.
            model: The name or identifier of the Ollama model to use.
            system_prompt: Optional system prompt to guide model behavior.
                           If provided, will be inserted at the start of the
                           conversation.
            return_full_response: If True, return the full response object.
                If False, return only the generated text.
            stream: If True, return an iterator for streaming responses.
            json_output: If True, set format="json" to get JSON-formatted
                responses using Ollama's native JSON support. The messages
                should specify the desired JSON structure.
            temperature: Optional temperature value for generation (0.0-2.0).
                Controls randomness in the output.
            top_p: Optional top-p value for nucleus sampling (0.0-1.0).
                Controls diversity of the output.
            **kwargs: Additional keyword arguments to pass to the Ollama API.

        Returns:
            OllamaGenericResponse: The chat response, full response object,
            or an iterator for streaming responses.

        Example:
            Chat with default settings:
            ```python
            messages = [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is..."},
                {"role": "user", "content": "Give me some examples"}
            ]
            response = provider.chat(
                messages,
                model="llama2",
            )
            print(response)
            ```

            Creative chat with high temperature:
            ```python
            response = provider.chat(
                messages,
                model="llama2",
                temperature=0.8,
                top_p=0.9
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
                model="llama2",
                json_output=True
            )
            print(response)  # Will be JSON formatted
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

            options = self._prepare_options(
                json_output=json_output,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

            response = self.client.chat(
                model=model,
                messages=chat_messages,
                stream=stream,
                options=options,
            )

            if stream:
                return cast(
                    OllamaGenericResponse,
                    self._stream_chat_response(
                        cast(Iterator[OllamaChatResponse], response),
                        return_full_response,
                    ),
                )
            else:
                response = cast(OllamaChatResponse, response)
                if return_full_response:
                    return response
                else:
                    return response["message"]["content"]

        except Exception as e:
            raise self._map_exception_to_clientai_error(e)
