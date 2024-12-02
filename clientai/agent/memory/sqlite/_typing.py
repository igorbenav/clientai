from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class StoredValue(Generic[V]):
    """Represents a value stored in SQLite memory."""

    value: V
    metadata: Dict[str, Any]
    timestamp: datetime


class SQLiteConnectionProtocol(Protocol):
    """Protocol for SQLite connection operations."""

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        ...

    def commit(self) -> None:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> SQLiteConnectionProtocol:
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        ...


class SQLiteCursorProtocol(Protocol):
    """Protocol for SQLite cursor operations."""

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        ...

    def fetchone(self) -> Optional[tuple[Any, ...]]:
        ...

    def fetchall(self) -> list[tuple[Any, ...]]:
        ...
