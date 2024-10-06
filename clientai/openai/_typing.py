from __future__ import annotations

from collections.abc import Iterator
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

from .._common_types import GenericResponse, Message


class OpenAIChoice(TypedDict):
    index: int
    message: Message
    finish_reason: Optional[str]


class OpenAIUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage


class OpenAIStreamDelta(TypedDict):
    role: Optional[Literal["system", "user", "assistant", "function"]]
    content: Optional[str]
    function_call: Optional[Dict[str, Any]]


class OpenAIStreamChoice(TypedDict):
    index: int
    delta: OpenAIStreamDelta
    finish_reason: Optional[str]


class OpenAIStreamResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIStreamChoice]


class OpenAIChatCompletionProtocol(Protocol):
    def create(
        self, **kwargs: Any
    ) -> Union[OpenAIResponse, Iterator[OpenAIStreamResponse]]:
        ...


class OpenAIChatProtocol(Protocol):
    completions: OpenAIChatCompletionProtocol


class OpenAIClientProtocol(Protocol):
    chat: OpenAIChatProtocol


OpenAIProvider = Any
OpenAIFullResponse = Union[OpenAIResponse, OpenAIStreamResponse]
OpenAIStreamChunk = Union[str, OpenAIStreamResponse]

OpenAIGenericResponse = GenericResponse[
    str, OpenAIFullResponse, OpenAIStreamChunk
]

Client = "openai.OpenAI"
