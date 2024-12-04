from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional, Protocol, TypeVar, Union

K = TypeVar("K")
V = TypeVar("V")


class RedisClientProtocol(Protocol):
    """
    Protocol defining the required interface for Redis clients.

    This protocol ensures that Redis client objects provide the necessary
    methods for basic key-value operations and key management.

    Methods:
        set: Set a key-value pair with optional expiration.
        get: Retrieve a value by key.
        delete: Delete one or more keys.
        keys: Find all keys matching a pattern.
    """

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
        """
        Set a key-value pair with optional parameters.

        Args:
            name: The key name.
            value: The value to store.
            ex: Expire time in seconds.
            px: Expire time in milliseconds.
            nx: Only set if key does not exist.
            xx: Only set if key exists.
            keepttl: Retain the TTL of the existing key.
            get: Return the old value.
            exat: Expire at a specific Unix time in seconds.
            pxat: Expire at a specific Unix time in milliseconds.

        Returns:
            True if set was successful, False otherwise.
        """
        ...

    def get(self, name: Union[str, bytes]) -> Optional[bytes]:
        """
        Get the value of a key.

        Args:
            name: The key name.

        Returns:
            The value as bytes if found, None otherwise.
        """
        ...

    def delete(self, *names: Union[str, bytes]) -> int:
        """
        Delete one or more keys.

        Args:
            *names: One or more key names to delete.

        Returns:
            Number of keys that were deleted.
        """
        ...

    def keys(self, pattern: Union[str, bytes] = "*") -> list[bytes]:
        """
        Find all keys matching the given pattern.

        Args:
            pattern: Pattern to match against key names.

        Returns:
            List of matching key names as bytes.
        """
        ...
