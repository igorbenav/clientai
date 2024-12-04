from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def remember(key: str, **kwargs: Any):
    """
    Decorator for storing function results in memory with a fixed key.

    Provides direct memory storage without LLM-based decision making.
    Stores the function's return value in memory using the specified key.

    Args:
        key: The key under which to store the function's result.
        **kwargs: Additional arguments passed to the memory store operation.

    Returns:
        A decorated function that automatically stores its result.

    Examples:
        Basic usage with a simple key:
        >>> @remember("last_calculation")
        ... def calculate_total(self, numbers: List[float]) -> float:
        ...     return sum(numbers)

        Store with additional metadata:
        >>> @remember("user_preference", category="settings")
        ... def save_preference(self, preference: str) -> str:
        ...     return preference

        The stored value can later be retrieved using the same key:
        >>> value = agent.memory.retrieve("last_calculation")
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **inner_kwargs: Any) -> T:
            result = func(self, *args, **inner_kwargs)
            if hasattr(self, "memory"):
                self.memory.store(key, result, **kwargs)
            return result

        return wrapper

    return decorator


def recall(key: str, default: Any = None, **kwargs: Any):
    """
    Decorator for retrieving values from memory with a fixed key.

    Retrieves a value from memory and passes it as the first argument to the
    decorated function. Uses direct key lookup without LLM decision making.

    Args:
        key: The key to retrieve from memory.
        default: Value to use if key is not found in memory.
        **kwargs: Additional arguments passed to memory retrieval operation.

    Returns:
        A decorated function that receives the retrieved value as its first
        argument.

    Examples:
        Basic usage for displaying previous results:
        >>> @recall("last_calculation")
        ... def display_result(
        ...     self,
        ...     prev_result: float,
        ...     format_type: str
        ... ) -> str:
        ...     return f"{format_type}: {prev_result}"

        Using a default value:
        >>> @recall("user_preference", default="default_theme")
        ... def apply_theme(self, theme: str) -> None:
        ...     set_application_theme(theme)

        The decorated function receives the stored value first:
        >>> # If last_calculation = 42.0
        >>> # Internally calls: display_result(42.0, "formatted")
        >>> display_result("formatted")

    Raises:
        AttributeError: If the agent instance doesn't have a memory attribute.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **inner_kwargs: Any) -> T:
            if hasattr(self, "memory"):
                value = self.memory.retrieve(key, default, **kwargs)
                return func(self, value, *args, **inner_kwargs)
            return func(self, *args, **inner_kwargs)

        return wrapper

    return decorator
