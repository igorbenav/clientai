from __future__ import annotations

from collections.abc import Iterator
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    Union,
)

from .._common_types import GenericResponse


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class GroqChoice(TypedDict):
    index: int
    message: Message
    logprobs: Optional[Any]
    finish_reason: Optional[str]


class GroqUsage(TypedDict):
    queue_time: float
    prompt_tokens: int
    prompt_time: float
    completion_tokens: int
    completion_time: float
    total_tokens: int
    total_time: float


class GroqMetadata(TypedDict):
    id: str


class GroqResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[GroqChoice]
    usage: GroqUsage
    system_fingerprint: str
    x_groq: GroqMetadata


class GroqStreamDelta(TypedDict):
    role: Optional[Literal["system", "user", "assistant"]]
    content: Optional[str]


class GroqStreamChoice(TypedDict):
    index: int
    delta: GroqStreamDelta
    finish_reason: Optional[str]


class GroqStreamResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[GroqStreamChoice]
    system_fingerprint: str
    x_groq: GroqMetadata


class GroqChatCompletionProtocol(Protocol):
    def create(
        self, **kwargs: Any
    ) -> Union[GroqResponse, Iterator[GroqStreamResponse]]: ...


class GroqChatProtocol(Protocol):
    completions: GroqChatCompletionProtocol


class GroqClientProtocol(Protocol):
    chat: GroqChatProtocol


GroqProvider = Any
GroqFullResponse = Union[GroqResponse, GroqStreamResponse]
GroqStreamChunk = Union[str, GroqStreamResponse]

GroqGenericResponse = GenericResponse[str, GroqFullResponse, GroqStreamChunk]

Client = "groq.Groq"
