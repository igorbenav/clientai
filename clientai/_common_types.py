from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Dict, TypeVar, Union

JsonDict = Dict[str, Any]
Message = Dict[str, str]

T = TypeVar("T", covariant=True)
S = TypeVar("S", covariant=True)
R = TypeVar("R", covariant=True)

GenericResponse = Union[R, T, Iterator[Union[R, S]]]
