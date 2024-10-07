import time
from collections.abc import Iterator
from typing import Any, List, Union, cast

from .._common_types import Message
from ..ai_provider import AIProvider
from . import REPLICATE_INSTALLED
from ._typing import (
    ReplicateClientProtocol,
    ReplicateGenericResponse,
    ReplicatePredictionProtocol,
    ReplicateResponse,
    ReplicateStreamResponse,
)

if REPLICATE_INSTALLED:
    import replicate  # type: ignore

    Client = replicate.Client
else:
    Client = None  # type: ignore


class Provider(AIProvider):
    def __init__(self, api_key: str):
        if not REPLICATE_INSTALLED or Client is None:
            raise ImportError(
                "The replicate package is not installed. "
                "Please install it with 'pip install clientai[replicate]'."
            )
        self.client: ReplicateClientProtocol = Client(api_token=api_key)

    def _process_output(self, output: Any) -> str:
        if isinstance(output, List):
            return "".join(str(item) for item in output)
        elif isinstance(output, str):
            return output
        else:
            return str(output)

    def _wait_for_prediction(
        self, prediction_id: str, max_wait_time: int = 300
    ) -> ReplicatePredictionProtocol:
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            prediction = self.client.predictions.get(prediction_id)
            if prediction.status == "succeeded":
                return prediction
            elif prediction.status == "failed":
                raise Exception(f"Prediction failed: {prediction.error}")
            time.sleep(1)
        raise TimeoutError("Prediction timed out")

    def _stream_response(
        self,
        prediction: ReplicatePredictionProtocol,
        return_full_response: bool,
    ) -> Iterator[Union[str, ReplicateStreamResponse]]:
        metadata = cast(ReplicateStreamResponse, prediction.__dict__.copy())
        for event in prediction.stream():
            if return_full_response:
                metadata["output"] = self._process_output(event)
                yield metadata
            else:
                yield self._process_output(event)

    def generate_text(
        self,
        prompt: str,
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> ReplicateGenericResponse:
        prediction = self.client.predictions.create(
            model=model, input={"prompt": prompt}, stream=stream, **kwargs
        )

        if stream:
            return self._stream_response(prediction, return_full_response)
        else:
            completed_prediction = self._wait_for_prediction(prediction.id)
            if return_full_response:
                response = cast(
                    ReplicateResponse, completed_prediction.__dict__.copy()
                )
                response["output"] = self._process_output(
                    completed_prediction.output
                )
                return response
            else:
                return self._process_output(completed_prediction.output)

    def chat(
        self,
        messages: List[Message],
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> ReplicateGenericResponse:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt += "\nassistant: "

        prediction = self.client.predictions.create(
            model=model, input={"prompt": prompt}, stream=stream, **kwargs
        )

        if stream:
            return self._stream_response(prediction, return_full_response)
        else:
            completed_prediction = self._wait_for_prediction(prediction.id)
            if return_full_response:
                response = cast(
                    ReplicateResponse, completed_prediction.__dict__.copy()
                )
                response["output"] = self._process_output(
                    completed_prediction.output
                )
                return response
            else:
                return self._process_output(completed_prediction.output)
