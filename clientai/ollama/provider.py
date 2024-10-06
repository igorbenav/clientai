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
    Client = None


class Provider(AIProvider):
    def __init__(self, host: Optional[str] = None):
        if not OLLAMA_INSTALLED or Client is None:
            raise ImportError(
                "The ollama package is not installed. "
                "Please install it with 'pip install clientai[ollama]'."
            )
        self.client: OllamaClientProtocol = (
            Client(host=host) if host else ollama
        )

    def _stream_generate_response(
        self,
        stream: Iterator[OllamaStreamResponse],
        return_full_response: bool,
    ) -> Iterator[Union[str, OllamaStreamResponse]]:
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
