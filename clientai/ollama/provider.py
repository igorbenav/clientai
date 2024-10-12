from collections.abc import Iterator
from typing import Any, List, Optional, Union, cast

from ..ai_provider import AIProvider
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

    Examples:
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

    def generate_text(
        self,
        prompt: str,
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> OllamaGenericResponse:
        """
        Generate text based on a given prompt using a specified Ollama model.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the Ollama model to use.
            return_full_response: If True, return the full response object.
                If False, return only the generated text. Defaults to False.
            stream: If True, return an iterator for streaming responses.
                Defaults to False.
            **kwargs: Additional keyword arguments to pass to the Ollama API.

        Returns:
            OllamaGenericResponse: The generated text, full response object,
            or an iterator for streaming responses.

        Examples:
            Generate text (text only):
            ```python
            response = provider.generate_text(
                "Explain the concept of machine learning",
                model="llama2",
            )
            print(response)
            ```

            Generate text (full response):
            ```python
            response = provider.generate_text(
                "Explain the concept of machine learning",
                model="llama2",
                return_full_response=True
            )
            print(response["response"])
            ```

            Generate text (streaming):
            ```python
            for chunk in provider.generate_text(
                "Explain the concept of machine learning",
                model="llama2",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```
        """
        response = self.client.generate(
            model=model, prompt=prompt, stream=stream, **kwargs
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

    def chat(
        self,
        messages: List[Message],
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> OllamaGenericResponse:
        """
        Engage in a chat conversation using a specified Ollama model.

        Args:
            messages: A list of message dictionaries, each containing
                      'role' and 'content'.
            model: The name or identifier of the Ollama model to use.
            return_full_response: If True, return the full response object.
                If False, return only the generated text. Defaults to False.
            stream: If True, return an iterator for streaming responses.
                Defaults to False.
            **kwargs: Additional keyword arguments to pass to the Ollama API.

        Returns:
            OllamaGenericResponse: The chat response, full response object,
            or an iterator for streaming responses.

        Examples:
            Chat (message content only):
            ```python
            messages = [
                {"role": "user", "content": "What is the capital of Japan?"},
                {"role": "assistant", "content": "The capital is Tokyo."},
                {"role": "user", "content": "What is its population?"}
            ]
            response = provider.chat(
                messages,
                model="llama2",
            )
            print(response)
            ```

            Chat (full response):
            ```python
            response = provider.chat(
                messages,
                model="llama2",
                return_full_response=True
            )
            print(response["message"]["content"])
            ```

            Chat (streaming):
            ```python
            for chunk in provider.chat(
                messages,
                model="llama2",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```
        """
        response = self.client.chat(
            model=model, messages=messages, stream=stream, **kwargs
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
