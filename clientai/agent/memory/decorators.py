from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union, overload

T = TypeVar("T")


@overload
def smart_remember(
    func: None = None, **kwargs: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    ...


@overload
def smart_remember(func: Callable[..., T], **kwargs: Any) -> Callable[..., T]:
    ...


def smart_remember(
    func: Optional[Callable[..., T]] = None, **kwargs: Any
) -> Union[Callable[[Callable[..., T]], Callable[..., T]], Callable[..., T]]:
    """
    Decorator that uses LLM to intelligently store function results in memory.

    The LLM analyzes the function output and current context to decide
    where and how to store the information across different memory types
    (working, episodic, semantic).

    Args:
        func: The function to decorate. Can be None when using decorator with
            parameters.
        **kwargs: Additional arguments passed to the memory store operation.

    Returns:
        A decorated function that automatically stores its result in memory
        based on LLM decisions.

    Examples:
        Basic usage:
            @smart_remember
            def process_data(self, data: str) -> str:
                return f"Processed: {data}"

        With parameters:
            @smart_remember(tags=["important"])
            def analyze_results(self, results: dict) -> str:
                return json.dumps(results)
    """

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f)
        def wrapper(self: Any, *args: Any, **inner_kwargs: Any) -> T:
            result = f(self, *args, **inner_kwargs)

            if hasattr(self, "memory_manager"):
                decisions = self.memory_manager.decide_storage(
                    content=result, context=self.context.state
                )

                for decision in decisions:
                    for memory_type in decision.store_in:
                        memory = self.memories[memory_type]
                        memory.store(
                            decision.key,
                            result,
                            importance=decision.importance,
                            metadata=decision.metadata,
                            **kwargs,
                        )

            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


@overload
def smart_recall(
    func: None = None, query: Optional[str] = None, **kwargs: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    ...


@overload
def smart_recall(
    func: str, query: Optional[str] = None, **kwargs: Any
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    ...


@overload
def smart_recall(
    func: Callable[..., T], query: Optional[str] = None, **kwargs: Any
) -> Callable[..., T]:
    ...


def smart_recall(
    func: Optional[Union[Callable[..., T], str]] = None,
    query: Optional[str] = None,
    **kwargs: Any,
) -> Union[Callable[[Callable[..., T]], Callable[..., T]], Callable[..., T]]:
    """
    Decorator that uses LLM to intelligently retrieve relevant memories.

    The LLM analyzes the query, current task, and context to determine which
    memories to retrieve. Retrieved memories are passed to the decorated
    function via the 'retrieved_memories' keyword argument.

    Args:
        func: The function to decorate or a query string. When a string is
            provided, it's used as the retrieval query.
        query: Optional explicit query string for memory retrieval.
        **kwargs: Additional arguments passed to the memory retrieval
                  operation.

    Returns:
        A decorated function that receives automatically retrieved memories.

    Examples:
        Using function name as query:
            @smart_recall
            def analyze_data(self, data: str, retrieved_memories: dict) -> str:
                return f"Analysis with context: {retrieved_memories}"

        With explicit query:
            @smart_recall("historical_performance")
            def get_metrics(self, retrieved_memories: dict) -> dict:
                return process_metrics(retrieved_memories)

        Using as query string:
            @smart_recall("user_preferences")
            def customize_response(self, retrieved_memories: dict) -> str:
                return format_response(retrieved_memories)
    """

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f)
        def wrapper(self: Any, *args: Any, **inner_kwargs: Any) -> T:
            if hasattr(self, "memory_manager"):
                current_query = query or str(args[0] if args else "")

                memories = self.memory_manager.decide_retrieval(
                    query=current_query,
                    task=f.__name__,
                    context=self.context.state,
                )

                inner_kwargs["retrieved_memories"] = memories

            return f(self, *args, **inner_kwargs)

        return wrapper

    if isinstance(func, str):
        return smart_recall(None, query=func, **kwargs)

    if func is None:
        return decorator

    return decorator(func)


def remember(key: str, **kwargs: Any):
    """
    Simple decorator for storing function results in memory with a fixed key.

    Provides direct memory storage without LLM-based decision making.
    Stores the function's return value in memory using the specified key.

    Args:
        key: The key under which to store the function's result.
        **kwargs: Additional arguments passed to the memory store operation.

    Returns:
        A decorated function that automatically stores its result.

    Examples:
        @remember("last_calculation")
        def calculate_total(self, numbers: List[float]) -> float:
            return sum(numbers)

        @remember("user_preference", category="settings")
        def save_preference(self, preference: str) -> str:
            return preference
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
    Simple decorator for retrieving values from memory with a fixed key.

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
        @recall("last_calculation")
        def display_result(self, prev_result: float, format_type: str) -> str:
            return f"{format_type}: {prev_result}"

        @recall("user_preference", default="default_theme")
        def apply_theme(self, theme: str) -> None:
            set_application_theme(theme)
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
