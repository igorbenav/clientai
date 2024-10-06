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
    Client = None


class Provider(AIProvider):
    def __init__(self, api_key: str):
        if not OPENAI_INSTALLED or Client is None:
            raise ImportError(
                "The openai package is not installed. "
                "Please install it with 'pip install clientai[openai]'."
            )
        self.client: OpenAIClientProtocol = Client(api_key=api_key)

    def _stream_response(
        self,
        stream: Iterator[OpenAIStreamResponse],
        return_full_response: bool,
    ) -> Iterator[Union[str, OpenAIStreamResponse]]:
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
