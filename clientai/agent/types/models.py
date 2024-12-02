from typing import Any, Dict, TypedDict, Union


class ModelParameters(TypedDict, total=False):
    name: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: Union[str, list[str]]
    stream: bool
    extra: Dict[str, Any]
