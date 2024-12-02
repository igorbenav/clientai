from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, Union

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class StoredValue(Generic[V]):
    """Represents a value stored in Redis memory."""

    value: V
    metadata: Dict[str, Any]
    timestamp: datetime


class RedisClientProtocol(Protocol):
    """Protocol for Redis client operations."""

    def set(
        self,
        name: Union[str, bytes],
        value: Union[str, bytes, int, float],
        ex: Optional[Union[float, timedelta]] = None,
        px: Optional[Union[float, timedelta]] = None,
        nx: bool = False,
        xx: bool = False,
        keepttl: bool = False,
        get: bool = False,
        exat: Optional[Any] = None,
        pxat: Optional[Any] = None,
    ) -> Optional[bool]:
        ...

    def get(self, name: Union[str, bytes]) -> Optional[bytes]:
        ...

    def delete(self, *names: Union[str, bytes]) -> int:
        ...

    def keys(self, pattern: Union[str, bytes] = "*") -> list[bytes]:
        ...
