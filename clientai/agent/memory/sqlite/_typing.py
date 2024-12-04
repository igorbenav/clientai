from __future__ import annotations

from types import TracebackType
from typing import Any, Optional, Protocol, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class SQLiteConnectionProtocol(Protocol):
    """
    Protocol defining the required interface for SQLite connections.

    This protocol ensures that connection objects provide the necessary
    methods for executing queries, managing transactions, and proper
    context management.

    Methods:
        execute: Execute a SQL query with optional parameters.
        commit: Commit the current transaction.
        close: Close the connection.
        __enter__: Enter the context manager.
        __exit__: Exit the context manager.
    """

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        """
        Execute a SQL query with optional parameters.

        Args:
            sql: The SQL query to execute.
            parameters: Optional tuple of parameters for the query.

        Returns:
            The query result (implementation specific).
        """
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def close(self) -> None:
        """Close the database connection."""
        ...

    def __enter__(self) -> SQLiteConnectionProtocol:
        """Enter the context manager."""
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the context manager."""
        ...


class SQLiteCursorProtocol(Protocol):
    """
    Protocol defining the required interface for SQLite cursors.

    This protocol ensures that cursor objects provide the necessary
    methods for executing queries and fetching results.

    Methods:
        execute: Execute a SQL query with optional parameters.
        fetchone: Fetch the next row of a query result.
        fetchall: Fetch all remaining rows of a query result.
    """

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        """
        Execute a SQL query with optional parameters.

        Args:
            sql: The SQL query to execute.
            parameters: Optional tuple of parameters for the query.

        Returns:
            The cursor object for method chaining.
        """
        ...

    def fetchone(self) -> Optional[tuple[Any, ...]]:
        """
        Fetch the next row of a query result.

        Returns:
            The next row as a tuple, or None if no more rows are available.
        """
        ...

    def fetchall(self) -> list[tuple[Any, ...]]:
        """
        Fetch all remaining rows of a query result.

        Returns:
            A list of all remaining rows, where each row is a tuple.
        """
        ...
