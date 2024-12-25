from typing import Callable

from ..steps.base import Step
from ..tools.base import Tool
from .exceptions import StepError, ToolError


def validate_step(step: Step) -> None:
    """Validate step configuration and metadata.

    Checks that the step has a valid callable
    function with proper configuration.

    Args:
        step: The Step instance to validate.

    Raises:
        StepError: If the step has no function or the function is not callable.
    """
    if step.func is None:  # pragma: no cover
        raise StepError(f"Step {step.name} has no function")

    if not callable(step.func):  # pragma: no cover
        raise StepError(f"Step {step.name} function is not callable")


def validate_tool(tool: Tool) -> None:
    """Validate tool configuration and signature.

    Checks that the tool has a valid callable function
    with proper configuration.

    Args:
        tool: The Tool instance to validate.

    Raises:
        ToolError: If the tool has no function or the function is not callable.
    """
    if tool.func is None:  # pragma: no cover
        raise ToolError(f"Tool {tool.name} has no function")

    if not callable(tool.func):  # pragma: no cover
        raise ToolError(f"Tool {tool.name} function is not callable")


def validate_callable(func: Callable) -> None:
    """Validate that a callable has proper type hints.

    Checks that the function has complete type annotations
    including return type.

    Args:
        func: The function to validate.

    Raises:
        ValueError: If the function is missing type annotations or return type.

    Example:
        ```python
        def valid_func(x: int) -> str:
            return str(x)

        validate_callable(valid_func)  # Passes

        def invalid_func(x):  # No type hints
            return str(x)

        validate_callable(invalid_func)  # Raises ValueError
        ```
    """
    if not callable(func):  # pragma: no cover
        raise ValueError(f"Object {func} is not callable")
