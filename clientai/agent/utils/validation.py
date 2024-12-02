from typing import Callable

from ..steps.base import Step
from ..tools.base import Tool
from .exceptions import StepError, ToolError


def validate_step(step: Step) -> None:
    """Validate step configuration and metadata."""
    if step.func is None:
        raise StepError(f"Step {step.name} has no function")

    if not callable(step.func):
        raise StepError(f"Step {step.name} function is not callable")


def validate_tool(tool: Tool) -> None:
    """Validate tool configuration and signature."""
    if tool.func is None:
        raise ToolError(f"Tool {tool.name} has no function")

    if not callable(tool.func):
        raise ToolError(f"Tool {tool.name} function is not callable")


def validate_callable(func: Callable) -> None:
    """Validate that a callable has proper type hints."""
    try:
        annotations = func.__annotations__
        if "return" not in annotations:
            raise ValueError(
                f"Function {func.__name__} missing return type annotation"
            )
    except AttributeError:
        raise ValueError(f"Function {func.__name__} has no type annotations")
