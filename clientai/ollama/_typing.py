from __future__ import annotations

from collections.abc import Iterator
from typing import Any, List, Optional, Protocol, TypedDict, Union

from .._common_types import GenericResponse, Message


class OllamaResponse(TypedDict):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]]
    total_duration: Optional[int]
    load_duration: Optional[int]
    prompt_eval_count: Optional[int]
    prompt_eval_duration: Optional[int]
    eval_count: Optional[int]
    eval_duration: Optional[int]
    done_reason: Optional[str]


class OllamaStreamResponse(TypedDict):
    model: str
    created_at: str
    response: str
    done: bool


class OllamaChatResponse(TypedDict):
    model: str
    created_at: str
    message: Message
    done: bool
    total_duration: Optional[int]
    load_duration: Optional[int]
    prompt_eval_count: Optional[int]
    prompt_eval_duration: Optional[int]
    eval_count: Optional[int]
    eval_duration: Optional[int]


OllamaProvider = Any
OllamaFullResponse = Union[OllamaResponse, OllamaChatResponse]
OllamaStreamChunk = Union[str, OllamaStreamResponse]

OllamaGenericResponse = GenericResponse[
    str, OllamaFullResponse, OllamaStreamChunk
]


class OllamaClientProtocol(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[OllamaResponse, Iterator[OllamaStreamResponse]]: ...

    def chat(
        self,
        model: str,
        messages: List[Message],
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[OllamaChatResponse, Iterator[OllamaStreamResponse]]: ...


Client = "ollama.Client"
