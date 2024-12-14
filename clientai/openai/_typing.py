from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    Union,
)

from .._common_types import GenericResponse


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "function"]
    content: str


@dataclass
class OpenAIChoice:
    index: int
    message: Message
    finish_reason: Optional[str]


class OpenAIUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class OpenAIResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage


@dataclass
class OpenAIStreamDelta:
    role: Optional[Literal["system", "user", "assistant", "function"]]
    content: Optional[str]
    function_call: Optional[Dict[str, Any]]


@dataclass
class OpenAIStreamChoice:
    index: int
    delta: OpenAIStreamDelta
    finish_reason: Optional[str]


@dataclass
class OpenAIStreamResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIStreamChoice]


class OpenAIChatCompletionProtocol(Protocol):
    def create(
        self, **kwargs: Any
    ) -> Union[OpenAIResponse, Iterator[OpenAIStreamResponse]]: ...


class OpenAIChatProtocol(Protocol):
    completions: OpenAIChatCompletionProtocol


class OpenAIClientProtocol(Protocol):
    chat: OpenAIChatProtocol


class OpenAIChatCompletions(Protocol):
    def create(
        self,
        model: str,
        messages: List[Message],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[OpenAIResponse, OpenAIStreamResponse]: ...


OpenAIProvider = Any
OpenAIFullResponse = Union[OpenAIResponse, OpenAIStreamResponse]
OpenAIStreamChunk = Union[str, OpenAIStreamResponse]

OpenAIGenericResponse = GenericResponse[
    str, OpenAIFullResponse, OpenAIStreamChunk
]

Client = "openai.OpenAI"
