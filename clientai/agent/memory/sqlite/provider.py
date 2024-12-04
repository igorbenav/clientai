import json
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, TypeVar, cast

from ..base import Memory
from ._typing import SQLiteConnectionProtocol

K = TypeVar("K")
V = TypeVar("V")


class SQLiteConnectionWrapper:
    """
    Wrapper for sqlite3.Connection that implements SQLiteConnectionProtocol.

    This wrapper ensures proper implementation of the connection protocol
    and provides consistent context management behavior.

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: Path | str):
        self.conn = sqlite3.connect(path)

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        """Execute a SQL query with parameters."""
        return self.conn.execute(sql, parameters)

    def commit(self) -> None:
        """Commit the current transaction."""
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self) -> SQLiteConnectionProtocol:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """
        Exit the context manager, committing
        changes and closing connection.
        """
        self.conn.commit()
        self.conn.close()


class SQLiteMemory(Memory[K, V]):
    """
    SQLite-based memory implementation for persistent storage.

    This implementation stores values in a SQLite database, providing
    persistent storage across agent sessions. All values are JSON-serialized
    before storage.

    Args:
        db_path: Path to the SQLite database file.
        table_name: Name of the table used for storage. Default "agent_memory".
        init_db: Whether to initialize the database table. Defaults to True.

    Examples:
        Create and use SQLite memory:
        >>> memory = SQLiteMemory("agent_data.db")
        >>> memory.store("key1", {"value": 42})
        >>> value = memory.retrieve("key1")
        >>> print(value)  # Output: {'value': 42}

        Use custom table name:
        >>> memory = SQLiteMemory(
        ...     db_path="custom.db",
        ...     table_name="custom_memory"
        ... )
    """

    def __init__(
        self,
        db_path: str | Path,
        table_name: str = "agent_memory",
        init_db: bool = True,
    ):
        self.db_path = Path(db_path)
        self.table_name = table_name

        if init_db:
            self._initialize_db()

    def _get_connection(self) -> SQLiteConnectionProtocol:
        """Create and return a new database connection."""
        return SQLiteConnectionWrapper(self.db_path)

    def _initialize_db(self) -> None:
        """Create the necessary database table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """
            )

    def store(self, key: K, value: V, **kwargs: Any) -> None:
        """
        Store a value in SQLite with JSON serialization.

        Args:
            key: The key under which to store the value.
            value: The value to store (must be JSON-serializable).
            **kwargs: Additional arguments (ignored in this implementation).

        Raises:
            TypeError: If the value cannot be JSON serialized.
            sqlite3.Error: If the database operation fails.
        """
        with self._get_connection() as conn:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {self.table_name}
                (key, value)
                VALUES (?, ?)
                """,
                (str(key), json.dumps(value)),
            )

    def retrieve(
        self, key: K, default: Optional[V] = None, **kwargs: Any
    ) -> Optional[V]:
        """
        Retrieve a value by its key with JSON deserialization.

        Args:
            key: The key of the value to retrieve.
            default: Value to return if key is not found.
            **kwargs: Additional arguments (ignored in this implementation).

        Returns:
            The deserialized value if found, otherwise the default value.

        Raises:
            json.JSONDecodeError: If the stored value is not valid JSON.
            sqlite3.Error: If the database operation fails.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT value
                FROM {self.table_name}
                WHERE key = ?
                """,
                (str(key),),
            )
            row = cursor.fetchone()

        if row:
            # Use cast to tell mypy that json.loads returns our expected type V
            return cast(V, json.loads(row[0]))
        return default

    def remove(self, key: K, **kwargs: Any) -> None:
        """
        Remove a value by its key.

        Args:
            key: The key of the value to remove.
            **kwargs: Additional arguments (ignored in this implementation).

        Raises:
            sqlite3.Error: If the database operation fails.
        """
        with self._get_connection() as conn:
            conn.execute(
                f"DELETE FROM {self.table_name} WHERE key = ?", (str(key),)
            )

    def clear(self, **kwargs: Any) -> None:
        """
        Clear all stored values.

        Args:
            **kwargs: Additional arguments (ignored in this implementation).

        Raises:
            sqlite3.Error: If the database operation fails.
        """
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
