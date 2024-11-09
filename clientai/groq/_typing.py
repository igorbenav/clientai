from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
)

from .._common_types import GenericResponse


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class GroqChoice:
    index: int
    message: Message
    logprobs: Optional[Any]
    finish_reason: Optional[str]


@dataclass
class GroqUsage:
    queue_time: float
    prompt_tokens: int
    prompt_time: float
    completion_tokens: int
    completion_time: float
    total_tokens: int
    total_time: float


@dataclass
class GroqMetadata:
    id: str


@dataclass
class GroqResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[GroqChoice]
    usage: GroqUsage
    system_fingerprint: str
    x_groq: GroqMetadata


@dataclass
class GroqStreamDelta:
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None


@dataclass
class GroqStreamChoice:
    index: int
    delta: GroqStreamDelta
    finish_reason: Optional[str]


@dataclass
class GroqStreamResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[GroqStreamChoice]
    system_fingerprint: str
    x_groq: GroqMetadata


class GroqChatCompletionProtocol(Protocol):
    def create(
        self,
        *,
        messages: List[dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
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
