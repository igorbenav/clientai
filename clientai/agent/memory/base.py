from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Memory(Generic[K, V], ABC):
    """
    Abstract base class defining the interface for memory implementations.

    This class provides a generic interface for storing and retrieving
    values in memory systems. It supports different key and value types
    through type parameters and allows for flexible implementations
    (e.g., in-memory, persistent storage).

    Type Parameters:
        K: The type of keys used for storing and retrieving values.
        V: The type of values to be stored in memory.

    Examples:
        Define a concrete memory implementation:
        >>> class DictMemory(Memory[str, int]):
        ...     def __init__(self):
        ...         self._storage = {}
        ...
        ...     def store(self, key: str, value: int, **kwargs) -> None:
        ...         self._storage[key] = value
        ...
        ...     def retrieve(self, key: str, default: int = 0) -> int:
        ...         return self._storage.get(key, default)
        ...
        ...     def remove(self, key: str) -> None:
        ...         self._storage.pop(key, None)
        ...
        ...     def clear(self) -> None:
        ...         self._storage.clear()
    """

    @abstractmethod
    def store(self, key: K, value: V, **kwargs: Any) -> None:
        """
        Store a value in memory with the specified key.

        Args:
            key: The key under which to store the value.
            value: The value to store.
            **kwargs: Additional implementation-specific arguments for storage
                     operations (e.g., expiration, metadata).

        Raises:
            TypeError: If key or value types don't match the
                       memory implementation.
            ValueError: If the value cannot be stored for
                        implementation-specific reasons.

        Examples:
            >>> memory = ConcreteMemory[str, dict]()
            >>> memory.store("user_data", {"name": "Alice"}, ttl=3600)
        """
        pass

    @abstractmethod
    def retrieve(
        self, key: K, default: Optional[V] = None, **kwargs: Any
    ) -> Optional[V]:
        """
        Retrieve a value from memory by its key.

        Args:
            key: The key of the value to retrieve.
            default: Value to return if key is not found. Defaults to None.
            **kwargs: Additional implementation-specific arguments for
                      retrieval operations (e.g., consistency requirements).

        Returns:
            The stored value if found, otherwise the default value.

        Examples:
            >>> memory = ConcreteMemory[str, int]()
            >>> value = memory.retrieve("counter", default=0)
            >>> value = memory.retrieve("data", consistency="strong")
        """
        pass

    @abstractmethod
    def remove(self, key: K, **kwargs: Any) -> None:
        """
        Remove a value from memory by its key.

        Args:
            key: The key of the value to remove.
            **kwargs: Additional implementation-specific arguments for removal
                     operations (e.g., soft delete flags).

        Examples:
            >>> memory = ConcreteMemory[str, bytes]()
            >>> memory.remove("temp_file")
            >>> memory.remove("user_data", soft_delete=True)
        """
        pass

    @abstractmethod
    def clear(self, **kwargs: Any) -> None:
        """
        Remove all values from memory.

        Args:
            **kwargs: Additional implementation-specific arguments for clear
                     operations (e.g., backup flags, clear scope).

        Examples:
            >>> memory = ConcreteMemory[str, Any]()
            >>> memory.clear()
            >>> memory.clear(backup=True)
            >>> memory.clear(scope="temporary")
        """
        pass
