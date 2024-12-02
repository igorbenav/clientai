from collections.abc import Iterator
from typing import Any, List, Union, cast

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
from . import GROQ_INSTALLED
from ._typing import (
    GroqClientProtocol,
    GroqGenericResponse,
    GroqResponse,
    GroqStreamResponse,
)

if GROQ_INSTALLED:
    from groq import (
        APIStatusError,
        APITimeoutError,
        BadRequestError,
        ConflictError,
        Groq,
        InternalServerError,
        NotFoundError,
        PermissionDeniedError,
        UnprocessableEntityError,
    )
    from groq import (
        AuthenticationError as GroqAuthenticationError,
    )
    from groq import (
        RateLimitError as GroqRateLimitError,
    )

    Client = Groq
else:
    Client = None  # type: ignore


class Provider(AIProvider):
    """
    Groq-specific implementation of the AIProvider abstract base class.

    This class provides methods to interact with Groq's models for
    text generation and chat functionality.

    Attributes:
        client: The Groq client used for making API calls.

    Args:
        api_key: The API key for authenticating with Groq.

    Raises:
        ImportError: If the Groq package is not installed.

    Examples:
        Initialize the Groq provider:
        ```python
        provider = Provider(api_key="your-groq-api-key")
        ```
    """

    def __init__(self, api_key: str):
        if not GROQ_INSTALLED or Client is None:
            raise ImportError(
                "The groq package is not installed. "
                "Please install it with 'pip install clientai[groq]'."
            )
        self.client: GroqClientProtocol = cast(
            GroqClientProtocol, Client(api_key=api_key)
        )

    def _stream_response(
        self,
        stream: Iterator[GroqStreamResponse],
        return_full_response: bool,
    ) -> Iterator[Union[str, GroqStreamResponse]]:
        """
        Process the streaming response from Groq API.

        Args:
            stream: The stream of responses from Groq API.
            return_full_response: If True, yield full response objects.

        Yields:
            Union[str, GroqStreamResponse]: Processed content or
                                            full response objects.
        """
        for chunk in stream:
            if return_full_response:
                yield chunk
            else:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

    def _map_exception_to_clientai_error(self, e: Exception) -> ClientAIError:
        """
        Maps a Groq exception to the appropriate ClientAI exception.

        Args:
            e (Exception): The exception caught during the API call.

        Returns:
            ClientAIError: An instance of the appropriate ClientAI exception.
        """
        error_message = str(e)

        if isinstance(e, (GroqAuthenticationError, PermissionDeniedError)):  # noqa: UP038
            return AuthenticationError(
                error_message,
                status_code=getattr(e, "status_code", 401),
                original_error=e,
            )
        elif isinstance(e, GroqRateLimitError):
            return RateLimitError(
                error_message, status_code=429, original_error=e
            )
        elif isinstance(e, NotFoundError):
            return ModelError(error_message, status_code=404, original_error=e)
        elif isinstance(  # noqa: UP038
            e, (BadRequestError, UnprocessableEntityError, ConflictError)
        ):
            return InvalidRequestError(
                error_message,
                status_code=getattr(e, "status_code", 400),
                original_error=e,
            )
        elif isinstance(e, APITimeoutError):
            return TimeoutError(
                error_message, status_code=408, original_error=e
            )
        elif isinstance(e, InternalServerError):
            return APIError(
                error_message,
                status_code=getattr(e, "status_code", 500),
                original_error=e,
            )
        elif isinstance(e, APIStatusError):
            status = getattr(e, "status_code", 500)
            if status >= 500:
                return APIError(
                    error_message, status_code=status, original_error=e
                )
            return InvalidRequestError(
                error_message, status_code=status, original_error=e
            )

        return ClientAIError(error_message, status_code=500, original_error=e)

    def generate_text(
        self,
        prompt: str,
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        **kwargs: Any,
    ) -> GroqGenericResponse:
        """
        Generate text based on a given prompt using a specified Groq model.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the Groq model to use.
            return_full_response: If True, return the full response object.
                If False, return only the generated text. Defaults to False.
            stream: If True, return an iterator for streaming responses.
                Defaults to False.
            json_output: If True, format the response as valid JSON using
                Groq's native JSON mode (beta). Note that this is incompatible
                with streaming and stop sequences. Will return a 400 error with
                code "json_validate_failed" if JSON generation fails. Defaults
                to False.
            **kwargs: Additional keyword arguments to pass to the Groq API.

        Returns:
            GroqGenericResponse: The generated text, full response object,
            or an iterator for streaming responses.

        Raises:
            InvalidRequestError: If json_output and stream are both True.
            ClientAIError: If an error occurs during the API call.

        Examples:
            Generate text (text only):
            ```python
            response = provider.generate_text(
                "Explain quantum computing",
                model="llama3-8b-8192",
            )
            print(response)
            ```

            Generate text (full response):
            ```python
            response = provider.generate_text(
                "Explain quantum computing",
                model="llama3-8b-8192",
                return_full_response=True
            )
            print(response.choices[0].message.content)
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
                model="llama3-8b-8192",
                json_output=True
            )
            print(response)  # Will be valid JSON
            ```
        """
        try:
            messages: List[Message] = [{"role": "user", "content": prompt}]

            completion_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": stream,
            }
            if json_output:
                completion_kwargs["response_format"] = {"type": "json_object"}
            completion_kwargs.update(kwargs)

            response = self.client.chat.completions.create(**completion_kwargs)

            if stream:
                return cast(
                    GroqGenericResponse,
                    self._stream_response(
                        cast(Iterator[GroqStreamResponse], response),
                        return_full_response,
                    ),
                )
            else:
                response = cast(GroqResponse, response)
                if return_full_response:
                    return response
                else:
                    return cast(str, response.choices[0].message.content)

        except Exception as e:
            raise self._map_exception_to_clientai_error(e)

    def chat(
        self,
        messages: List[Message],
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        **kwargs: Any,
    ) -> GroqGenericResponse:
        """
        Engage in a chat conversation using a specified Groq model.

        Args:
            messages: A list of message dictionaries, each containing
                    'role' and 'content'.
            model: The name or identifier of the Groq model to use.
            return_full_response: If True, return the full response object.
                If False, return only the chat content.
            stream: If True, return an iterator for streaming responses.
            json_output: If True, format the response as valid JSON using
                Groq's native JSON mode (beta). Note that this is incompatible
                with streaming and stop sequences. Will return a 400 error with
                code "json_validate_failed" if JSON generation fails.
                Defaults to False.
            **kwargs: Additional keyword arguments to pass to the Groq API.

        Returns:
            GroqGenericResponse: The chat response, full response object,
            or an iterator for streaming responses.

        Raises:
            InvalidRequestError: If json_output and stream are both True.
            ClientAIError: If an error occurs during the API call.

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
                model="llama3-8b-8192",
            )
            print(response)
            ```

            Chat (full response):
            ```python
            response = provider.chat(
                messages,
                model="llama3-8b-8192",
                return_full_response=True
            )
            print(response.choices[0].message.content)
            ```

            Chat with JSON output:
            ```python
            messages = [
                {"role": "user", "content": '''Generate a user profile with:
                {
                    "name": "A random name",
                    "age": "A random age between 20-80",
                    "occupation": "A random occupation"
                }'''}
            ]
            response = provider.chat(
                messages,
                model="llama3-8b-8192",
                json_output=True
            )
            print(response)  # Will be valid JSON
            ```
        """
        try:
            completion_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": stream,
            }
            if json_output:
                completion_kwargs["response_format"] = {"type": "json_object"}
            completion_kwargs.update(kwargs)

            response = self.client.chat.completions.create(**completion_kwargs)

            if stream:
                return cast(
                    GroqGenericResponse,
                    self._stream_response(
                        cast(Iterator[GroqStreamResponse], response),
                        return_full_response,
                    ),
                )
            else:
                response = cast(GroqResponse, response)
                if return_full_response:
                    return response
                else:
                    return cast(str, response.choices[0].message.content)

        except Exception as e:
            raise self._map_exception_to_clientai_error(e)
