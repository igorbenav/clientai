from typing import Callable, Optional, TypeVar

from .base import Tool

T = TypeVar("T")


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable[..., T]], Tool]:
    """Decorator that converts a function into a Tool instance.

    Creates a Tool with optional custom name and description. If not provided,
    uses the function's name and docstring. The decorated function must have
    proper type hints.

    Args:
        name: Optional custom name for the tool. Defaults to function name.
        description: Optional custom description. Defaults to docstring.

    Returns:
        A decorator function that creates a Tool instance.

    Raises:
        ValueError: If the function lacks required type hints
                    or has invalid signature.

    Example:
        Basic usage with automatic name and description:
        ```python
        @tool
        def calculate(x: int, y: int) -> int:
            '''Add two numbers together.'''
            return x + y
        ```

        Custom name and description:
        ```python
        @tool(
            name="Calculator",
            description="Performs addition of two integers"
        )
        def add_numbers(x: int, y: int) -> int:
            return x + y
        ```

        Using the decorated tool:
        ```python
        result = calculate(5, 3)  # Returns 8
        print(calculate.name)  # Output: "calculate"
        ```

    Notes:
        - Decorated function must have type hints for all parameters
        - Return type hint is required
        - Description defaults to function docstring if not provided
        - Tool inherits all function attributes and metadata
    """

    if callable(name):
        return Tool.create(func=name)

    def decorator(func: Callable[..., T]) -> Tool:
        """Create a Tool instance from the decorated function.

        Args:
            func: The function to convert into a tool.

        Returns:
            A Tool instance wrapping the original function.

        Raises:
            ValueError: If function signature validation fails.
        """
        return Tool.create(
            func=func,
            name=name,
            description=description,
        )

    return decorator
