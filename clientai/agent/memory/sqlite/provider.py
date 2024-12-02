import json
import sqlite3
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, TypeVar

from ..base import Memory
from ._typing import SQLiteConnectionProtocol, StoredValue

K = TypeVar("K")
V = TypeVar("V")


class SQLiteConnectionWrapper:
    """Wrapper for sqlite3.Connection that properly
    implements SQLiteConnectionProtocol."""

    def __init__(self, path: Path | str):
        self.conn = sqlite3.connect(path)

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        return self.conn.execute(sql, parameters)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SQLiteConnectionProtocol:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.conn.commit()
        self.conn.close()


class SQLiteMemory(Memory[K, V]):
    """SQLite-based memory implementation for persistent storage."""

    def __init__(
        self,
        db_path: str | Path,
        table_name: str = "agent_memory",
        init_db: bool = True,
    ):
        """Initialize SQLite memory storage."""
        self.db_path = Path(db_path)
        self.table_name = table_name

        if init_db:
            self._initialize_db()

    def _get_connection(self) -> SQLiteConnectionProtocol:
        """Get a SQLite connection."""
        return SQLiteConnectionWrapper(self.db_path)

    def _initialize_db(self) -> None:
        """Create the necessary database table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    def _serialize_value(self, stored: StoredValue[V]) -> tuple[str, str, str]:
        """Serialize a StoredValue for database storage."""
        return (
            json.dumps(stored.value),
            json.dumps(stored.metadata),
            stored.timestamp.isoformat(),
        )

    def _deserialize_value(self, row: tuple[str, str, str]) -> StoredValue[V]:
        """Deserialize database row to StoredValue."""
        value, metadata, timestamp = row
        return StoredValue(
            value=json.loads(value),
            metadata=json.loads(metadata),
            timestamp=datetime.fromisoformat(timestamp),
        )

    def store(self, key: K, value: V, **kwargs: Any) -> None:
        """Store a value with associated metadata."""
        stored = StoredValue(
            value=value,
            metadata=kwargs.get("metadata", {}),
            timestamp=datetime.now(),
        )

        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {self.table_name}
                (key, value, metadata, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (str(key), *self._serialize_value(stored)),
            )

    def retrieve(
        self, key: K, default: Optional[V] = None, **kwargs: Any
    ) -> Optional[V]:
        """Retrieve a value by its key."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT value, metadata, timestamp
                FROM {self.table_name}
                WHERE key = ?
                """,
                (str(key),),
            )
            row = cursor.fetchone()

        if row:
            stored = self._deserialize_value(row)
            return stored.value
        return default

    def remove(self, key: K, **kwargs: Any) -> None:
        """Remove a value by its key."""
        with self._get_connection() as conn:
            conn.execute(
                f"DELETE FROM {self.table_name} WHERE key = ?", (str(key),)
            )

    def clear(self, **kwargs: Any) -> None:
        """Clear all stored values."""
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
