from dataclasses import dataclass, field
from functools import cached_property
from inspect import getdoc
from typing import Any, Callable, Optional, TypeVar

from .types import ToolSignature

T = TypeVar("T")
ToolCallable = Callable[..., Any]


@dataclass(frozen=True)
class Tool:
    """
    A callable tool with metadata for use in agent workflows.

    Represents a function with associated metadata (name, description,
    signature) that can be used as a tool by an agent.
    Tools are immutable and can be called like regular functions.

    Attributes:
        func: The underlying function that implements the tool's logic.
        name: The tool's display name.
        description: Human-readable description of the tool's purpose.
        _signature: Internal cached signature information.

    Examples:
        Create a tool from a function:
        >>> def calculate(x: int, y: int) -> int:
        ...     '''Add two numbers.'''
        ...     return x + y
        >>> tool = Tool.create(calculate)
        >>> result = tool(5, 3)
        >>> print(result)  # Output: 8

        Create a tool with custom metadata:
        >>> tool = Tool.create(
        ...     calculate,
        ...     name="Calculator",
        ...     description="Performs basic arithmetic"
        ... )
    """

    func: ToolCallable
    name: str
    description: str
    _signature: ToolSignature = field(
        default_factory=lambda: ToolSignature.empty(),  # type: ignore
        repr=False,
    )

    @classmethod
    def create(
        cls,
        func: ToolCallable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "Tool":
        """
        Create a new Tool instance from a callable.

        Factory method that creates a Tool with proper signature analysis and
        metadata extraction. Uses function's docstring as description if none
        provided.

        Args:
            func: The function to convert into a tool.
            name: Optional custom name for the tool. Defaults to function name.
            description: Optional custom description. Defaults to function
                docstring.

        Returns:
            A new Tool instance.

        Examples:
            >>> def format_text(text: str, uppercase: bool = False) -> str:
            ...     '''Format input text.'''
            ...     return text.upper() if uppercase else text
            >>> tool = Tool.create(format_text)
            >>> tool = Tool.create(
            ...     format_text,
            ...     name="Formatter",
            ...     description="Text formatting utility"
            ... )
        """
        actual_name = name or func.__name__
        actual_description = (
            description or getdoc(func) or "No description available"
        )
        signature = ToolSignature.from_callable(func, actual_name)

        return cls(
            func=func,
            name=actual_name,
            description=actual_description,
            _signature=signature,
        )

    @property
    def signature(self) -> ToolSignature:
        """
        Get the tool's signature information.

        Provides access to the analyzed signature of the tool's function,
        creating it if not already cached.

        Returns:
            Signature information for the tool.

        Examples:
            >>> tool = Tool.create(my_function)
            >>> sig = tool.signature
            >>> print(sig.parameters)
        """
        if self._signature is None:
            sig = ToolSignature.from_callable(self.func, self.name)
            object.__setattr__(self, "_signature", sig)
            return sig
        return self._signature

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool's function with the provided arguments.

        Makes the Tool instance callable, delegating to
        the underlying function.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the tool's function execution.

        Examples:
            >>> tool = Tool.create(calculate)
            >>> result = tool(5, 3)
            >>> result = tool(x=5, y=3)
        """
        return self.func(*args, **kwargs)

    @cached_property
    def signature_str(self) -> str:
        """
        Get a string representation of the tool's signature.

        Provides a cached, formatted string version of the tool's signature
        for display purposes.

        Returns:
            A formatted string representing the tool's signature.

        Examples:
            >>> tool = Tool.create(calculate)
            >>> print(tool.signature_str)
            # Output: "calculate(x: int, y: int) -> int"
        """
        return self.signature.format()

    def __str__(self) -> str:
        """
        Get a string representation of the tool.

        Returns:
            A string showing the tool's name and signature.

        Examples:
            >>> tool = Tool.create(calculate)
            >>> print(tool)
            # Output: "Tool(name='calculate', signature='calculate(x: int...')"
        """
        return f"Tool(name='{self.name}', signature='{self.signature_str}')"
