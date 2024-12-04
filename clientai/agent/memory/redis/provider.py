import json
from typing import Any, Optional, TypeVar, cast

from ...._constants import REDIS_INSTALLED
from ..base import Memory
from ._typing import RedisClientProtocol

if REDIS_INSTALLED:
    import redis  # type: ignore
else:
    redis = None  # type: ignore

K = TypeVar("K")
V = TypeVar("V")


class RedisMemory(Memory[K, V]):
    """
    Redis-based memory implementation for distributed storage.

    This implementation stores values in Redis, providing distributed
    storage with optional expiration. All values are JSON-serialized
    before storage.

    Args:
        host: Redis server hostname. Defaults to "localhost".
        port: Redis server port. Defaults to 6379.
        db: Redis database number. Defaults to 0.
        prefix: Prefix for all keys. Defaults to "agent_memory:".
        **redis_kwargs: Additional arguments passed to Redis client.

    Raises:
        ImportError: If redis-py package is not installed.

    Examples:
        Basic usage with default settings:
        >>> memory = RedisMemory()
        >>> memory.store("key1", {"value": 42})
        >>> value = memory.retrieve("key1")
        >>> print(value)  # Output: {'value': 42}

        Custom configuration:
        >>> memory = RedisMemory(
        ...     host="redis.example.com",
        ...     port=6380,
        ...     db=1,
        ...     prefix="myapp:",
        ...     password="secret"
        ... )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "agent_memory:",
        **redis_kwargs: Any,
    ):
        if not REDIS_INSTALLED:
            raise ImportError(
                "Redis support requires redis-py. "
                "Install it with: pip install clientai[redis]"
            )

        self.prefix = prefix
        self.client: RedisClientProtocol = redis.Redis(
            host=host, port=port, db=db, **redis_kwargs
        )

    def _make_key(self, key: K) -> str:
        """
        Create a prefixed key string.

        Args:
            key: The original key.

        Returns:
            The key with the configured prefix added.
        """
        return f"{self.prefix}{str(key)}"

    def store(self, key: K, value: V, **kwargs: Any) -> None:
        """
        Store a value in Redis with JSON serialization.

        Args:
            key: The key under which to store the value.
            value: The value to store (must be JSON-serializable).
            **kwargs: Additional arguments passed to Redis set:
                     - ex: Expire time in seconds
                     - px: Expire time in milliseconds
                     - nx: Only set if key does not exist
                     - xx: Only set if key exists
                     - keepttl: Retain the time to live of the existing key

        Raises:
            TypeError: If the value cannot be JSON serialized.
            redis.RedisError: If the Redis operation fails.
        """
        self.client.set(self._make_key(key), json.dumps(value), **kwargs)

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
            redis.RedisError: If the Redis operation fails.
        """
        data = self.client.get(self._make_key(key))
        if data:
            return cast(V, json.loads(data))
        return default

    def remove(self, key: K, **kwargs: Any) -> None:
        """
        Remove a value by its key.

        Args:
            key: The key of the value to remove.
            **kwargs: Additional arguments (ignored in this implementation).

        Raises:
            redis.RedisError: If the Redis operation fails.
        """
        self.client.delete(self._make_key(key))

    def clear(self, **kwargs: Any) -> None:
        """
        Clear all stored values with this prefix.

        This method only removes keys that match the configured prefix,
        not all keys in the Redis database.

        Args:
            **kwargs: Additional arguments (ignored in this implementation).

        Raises:
            redis.RedisError: If the Redis operation fails.
        """
        keys = self.client.keys(f"{self.prefix}*")
        if keys:
            self.client.delete(*keys)
