from collections.abc import Iterator
from typing import Any, List, Union, cast

from .._common_types import Message
from ..ai_provider import AIProvider
from . import OPENAI_INSTALLED
from ._typing import (
    OpenAIClientProtocol,
    OpenAIGenericResponse,
    OpenAIResponse,
    OpenAIStreamResponse,
)

if OPENAI_INSTALLED:
    import openai  # type: ignore

    Client = openai.OpenAI
else:
    Client = None  # type: ignore


class Provider(AIProvider):
    """
    OpenAI-specific implementation of the AIProvider abstract base class.

    This class provides methods to interact with OpenAI's
    models for text generation and chat functionality.

    Attributes:
        client: The OpenAI client used for making API calls.

    Args:
        api_key: The API key for authenticating with OpenAI.

    Raises:
        ImportError: If the OpenAI package is not installed.

    Examples:
        Initialize the OpenAI provider:
        ```python
        provider = Provider(api_key="your-openai-api-key")
        ```
    """

    def __init__(self, api_key: str):
        if not OPENAI_INSTALLED or Client is None:
            raise ImportError(
                "The openai package is not installed. "
                "Please install it with 'pip install clientai[openai]'."
            )
        self.client: OpenAIClientProtocol = cast(
            OpenAIClientProtocol, Client(api_key=api_key)
        )

    def _stream_response(
        self,
        stream: Iterator[OpenAIStreamResponse],
        return_full_response: bool,
    ) -> Iterator[Union[str, OpenAIStreamResponse]]:
        """
        Process the streaming response from OpenAI API.

        Args:
            stream: The stream of responses from OpenAI API.
            return_full_response: If True, yield full response objects.

        Yields:
            Union[str, OpenAIStreamResponse]: Processed content or full
                                              response objects.
        """
        for chunk in stream:
            if return_full_response:
                yield chunk
            else:
                content = chunk["choices"][0]["delta"].get("content")
                if content:
                    yield content

    def generate_text(
        self,
        prompt: str,
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> OpenAIGenericResponse:
        """
        Generate text based on a given prompt using a specified OpenAI model.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the OpenAI model to use.
            return_full_response: If True, return the full response object.
                If False, return only the generated text. Defaults to False.
            stream: If True, return an iterator for streaming responses.
                Defaults to False.
            **kwargs: Additional keyword arguments to pass to the OpenAI API.

        Returns:
            OpenAIGenericResponse: The generated text, full response object,
            or an iterator for streaming responses.

        Examples:
            Generate text (text only):
            ```python
            response = provider.generate_text(
                "Explain the theory of relativity",
                model="gpt-3.5-turbo",
            )
            print(response)
            ```

            Generate text (full response):
            ```python
            response = provider.generate_text(
                "Explain the theory of relativity",
                model="gpt-3.5-turbo",
                return_full_response=True
            )
            print(response["choices"][0]["message"]["content"])
            ```

            Generate text (streaming):
            ```python
            for chunk in provider.generate_text(
                "Explain the theory of relativity",
                model="gpt-3.5-turbo",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            **kwargs,
        )

        if stream:
            return cast(
                OpenAIGenericResponse,
                self._stream_response(
                    cast(Iterator[OpenAIStreamResponse], response),
                    return_full_response,
                ),
            )
        else:
            response = cast(OpenAIResponse, response)
            if return_full_response:
                return response
            else:
                return response["choices"][0]["message"]["content"]

    def chat(
        self,
        messages: List[Message],
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> OpenAIGenericResponse:
        """
        Engage in a chat conversation using a specified OpenAI model.

        Args:
            messages: A list of message dictionaries, each containing
                      'role' and 'content'.
            model: The name or identifier of the OpenAI model to use.
            return_full_response: If True, return the full response object.
                If False, return only the generated text. Defaults to False.
            stream: If True, return an iterator for streaming responses.
                Defaults to False.
            **kwargs: Additional keyword arguments to pass to the OpenAI API.

        Returns:
            OpenAIGenericResponse: The chat response, full response object,
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
                model="gpt-3.5-turbo",
            )
            print(response)
            ```

            Chat (full response):
            ```python
            response = provider.chat(
                messages,
                model="gpt-3.5-turbo",
                return_full_response=True
            )
            print(response["choices"][0]["message"]["content"])
            ```

            Chat (streaming):
            ```python
            for chunk in provider.chat(
                messages,
                model="gpt-3.5-turbo",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```
        """
        response = self.client.chat.completions.create(
            model=model, messages=messages, stream=stream, **kwargs
        )

        if stream:
            return cast(
                OpenAIGenericResponse,
                self._stream_response(
                    cast(Iterator[OpenAIStreamResponse], response),
                    return_full_response,
                ),
            )
        else:
            response = cast(OpenAIResponse, response)
            if return_full_response:
                return response
            else:
                return response["choices"][0]["message"]["content"]
