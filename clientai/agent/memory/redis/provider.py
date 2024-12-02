import json
from datetime import datetime
from typing import Any, Optional, TypeVar

from ...._constants import REDIS_INSTALLED
from ..base import Memory
from ._typing import RedisClientProtocol, StoredValue

if REDIS_INSTALLED:
    import redis  # type: ignore
else:
    redis = None  # type: ignore

K = TypeVar("K")
V = TypeVar("V")


class RedisMemory(Memory[K, V]):
    """Redis-based memory implementation for distributed storage."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "agent_memory:",
        **redis_kwargs: Any,
    ):
        """Initialize Redis memory storage."""
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
        """Create a prefixed key string."""
        return f"{self.prefix}{str(key)}"

    def _serialize_value(self, stored: StoredValue[V]) -> str:
        """Serialize a StoredValue for Redis storage."""
        return json.dumps(
            {
                "value": stored.value,
                "metadata": stored.metadata,
                "timestamp": stored.timestamp.isoformat(),
            }
        )

    def _deserialize_value(self, data: bytes) -> StoredValue[V]:
        """Deserialize Redis data to StoredValue."""
        parsed = json.loads(data)
        return StoredValue(
            value=parsed["value"],
            metadata=parsed["metadata"],
            timestamp=datetime.fromisoformat(parsed["timestamp"]),
        )

    def store(self, key: K, value: V, **kwargs: Any) -> None:
        """Store a value with associated metadata."""
        stored = StoredValue(
            value=value,
            metadata=kwargs.get("metadata", {}),
            timestamp=datetime.now(),
        )
        self.client.set(self._make_key(key), self._serialize_value(stored))

    def retrieve(
        self, key: K, default: Optional[V] = None, **kwargs: Any
    ) -> Optional[V]:
        """Retrieve a value by its key."""
        data = self.client.get(self._make_key(key))
        if data:
            stored = self._deserialize_value(data)
            return stored.value
        return default

    def remove(self, key: K, **kwargs: Any) -> None:
        """Remove a value by its key."""
        self.client.delete(self._make_key(key))

    def clear(self, **kwargs: Any) -> None:
        """Clear all stored values with this prefix."""
        keys = self.client.keys(f"{self.prefix}*")
        if keys:
            self.client.delete(*keys)
