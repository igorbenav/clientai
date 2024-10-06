from typing import Any, Generic, List, Protocol, TypeVar, Union

from ._common_types import GenericResponse, Message, R, S, T
from .ollama._typing import (
    OllamaChatResponse,
    OllamaResponse,
    OllamaStreamResponse,
)
from .openai._typing import OpenAIResponse, OpenAIStreamResponse
from .replicate._typing import ReplicateResponse, ReplicateStreamResponse

ProviderResponse = Union[
    OpenAIResponse, ReplicateResponse, OllamaResponse, OllamaChatResponse
]

ProviderStreamResponse = Union[
    OpenAIStreamResponse, ReplicateStreamResponse, OllamaStreamResponse
]


class AIProviderProtocol(Protocol, Generic[R, T, S]):
    def generate_text(
        self,
        prompt: str,
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenericResponse[R, T, S]:
        ...

    def chat(
        self,
        messages: List[Message],
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenericResponse[R, T, S]:
        ...


P = TypeVar("P", bound=AIProviderProtocol)

AIGenericResponse = GenericResponse[
    str, ProviderResponse, ProviderStreamResponse
]
