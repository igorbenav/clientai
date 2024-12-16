from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Dict, Optional, Protocol, TypedDict, Union

from .._common_types import GenericResponse


class ReplicatePredictionProtocol(Protocol):
    id: str
    status: str
    error: Optional[str]
    output: Any

    def stream(self) -> Iterator[Any]:
        ...


ReplicatePrediction = ReplicatePredictionProtocol


class ReplicateMetrics(TypedDict):
    batch_size: float
    input_token_count: int
    output_token_count: int
    predict_time: float
    predict_time_share: float
    time_to_first_token: float
    tokens_per_second: float


class ReplicateUrls(TypedDict):
    cancel: str
    get: str
    stream: str


class ReplicateResponse(TypedDict):
    id: str
    model: str
    version: str
    status: str
    input: Dict[str, Any]
    output: str
    logs: str
    error: Optional[Any]
    metrics: Optional[ReplicateMetrics]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    urls: ReplicateUrls


ReplicateStreamResponse = ReplicateResponse

ReplicateProvider = Any
ReplicateFullResponse = ReplicateResponse
ReplicateStreamChunk = Union[str, ReplicateStreamResponse]


class ReplicatePredictionsProtocol(Protocol):
    @staticmethod
    def create(**kwargs: Any) -> ReplicatePredictionProtocol:
        ...

    @staticmethod
    def get(id: str) -> ReplicatePredictionProtocol:
        ...


class ReplicateClientProtocol(Protocol):
    predictions: ReplicatePredictionsProtocol


ReplicateGenericResponse = GenericResponse[
    str, ReplicateFullResponse, ReplicateStreamChunk
]

Client = "replicate.Client"
