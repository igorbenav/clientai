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
    signature) that can be used as a tool by an agent. Tools are immutable
    and can be called like regular functions.

    Tools can be created in three ways:
    1. Using the @tool decorator
    2. Using the @agent.register_tool decorator
    3. Direct registration with agent.register_tool()

    Attributes:
        func: The underlying function that implements the tool's logic.
        name: The tool's display name.
        description: Human-readable description of the tool's purpose.
        _signature: Internal cached signature information.

    Examples:
        Using the @tool decorator:
        >>> @tool
        ... def calculate(x: int, y: int) -> int:
        ...     '''Add two numbers.'''
        ...     return x + y
        >>> result = calculate(5, 3)
        >>> print(result)  # Output: 8

        Using @tool with parameters:
        >>> @tool(name="Calculator", description="Performs basic arithmetic")
        ... def add(x: int, y: int) -> int:
        ...     return x + y

        Direct creation and registration:
        >>> def multiply(x: int, y: int) -> int:
        ...     '''Multiply two numbers.'''
        ...     return x * y
        >>> tool = Tool.create(
        ...     multiply,
        ...     name="Multiplier",
        ...     description="Performs multiplication"
        ... )
        >>> agent.register_tool(tool)
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
            Basic tool creation:
            >>> def format_text(text: str, uppercase: bool = False) -> str:
            ...     '''Format input text.'''
            ...     return text.upper() if uppercase else text
            >>> tool = Tool.create(format_text)

            Custom metadata:
            >>> tool = Tool.create(
            ...     format_text,
            ...     name="Formatter",
            ...     description="Text formatting utility"
            ... )

            For agent registration:
            >>> agent.register_tool(
            ...     Tool.create(
            ...         format_text,
            ...         name="Formatter",
            ...         description="Formats text input"
            ...     )
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
            >>> @tool
            ... def my_function(x: int, y: str) -> str:
            ...     return f"{y}: {x}"
            >>> sig = my_function.signature
            >>> print(sig.parameters)  # Shows parameter information
        """
        if self._signature is None:
            sig = ToolSignature.from_callable(self.func, self.name)
            object.__setattr__(self, "_signature", sig)
            return sig
        return self._signature

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool's function with the provided arguments.

        Makes the Tool instance callable, delegating to the underlying
        function. This allows tools to be used like regular functions
        while maintaining their metadata.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the tool's function execution.

        Examples:
            Using a tool with positional arguments:
            >>> @tool
            ... def calculate(x: int, y: int) -> int:
            ...     return x + y
            >>> result = calculate(5, 3)

            Using a tool with keyword arguments:
            >>> result = calculate(x=5, y=3)
        """
        return self.func(*args, **kwargs)

    @cached_property
    def signature_str(self) -> str:
        """
        Get a string representation of the tool's signature.

        Provides a cached, formatted string version of the tool's signature
        for display purposes. This is useful for documentation and debugging.

        Returns:
            A formatted string representing the tool's signature.

        Examples:
            >>> @tool
            ... def calculate(x: int, y: int) -> int:
            ...     return x + y
            >>> print(calculate.signature_str)
            # Output: "calculate(x: int, y: int) -> int"
        """
        return self.signature.format()

    def format_tool_info(self, indent: str = "") -> str:
        """
        Format the tool's information in a standardized way for LLM prompts.

        Creates a consistently formatted string representation of the tool
        that includes its name, signature, and description in a hierarchical
        format. This format is designed to be clear for both human readers
        and LLM processing.

        Args:
            indent: Optional indentation prefix for each line.
                   Useful for nested formatting. Defaults to no indentation.

        Returns:
            A formatted string containing the tool's complete information in
            a standardized format.

        Examples:
            Basic formatting:
            >>> @tool(name="Calculator")
            ... def add(x: int, y: int) -> int:
            ...     '''Add two numbers together.'''
            ...     return x + y
            >>> print(add.format_tool_info())
            # Output:
            # - Calculator
            #   Signature: add(x: int, y: int) -> int
            #   Description: Add two numbers together

            With custom indentation:
            >>> print(add.format_tool_info("  "))
            # Output:
            #   - Calculator
            #     Signature: add(x: int, y: int) -> int
            #     Description: Add two numbers together

        Note:
            This format is specifically designed to be:
            1. Hierarchical with clear indentation
            2. Consistent across all tools
            3. Easy to parse in LLM prompts
            4. Human-readable for debugging
        """
        return (
            f"{indent}- {self.name}\n"
            f"{indent}  Signature: {self.signature_str}\n"
            f"{indent}  Description: {self.description}"
        )

    def __str__(self) -> str:
        """
        Get a complete string representation of the tool.

        Provides a formatted string containing all relevant tool
        information using the standardized format defined by
        format_tool_info(). This ensures consistency between
        string representation and prompt formatting.

        Returns:
            A formatted string containing the tool's complete information.

        Example:
            >>> @tool(name="Calculator")
            ... def add(x: int, y: int) -> int:
            ...     '''Add two numbers together.'''
            ...     return x + y
            >>> print(str(add))
            # Output:
            # - Calculator
            #   Signature: add(x: int, y: int) -> int
            #   Description: Add two numbers together

        Note:
            This method uses format_tool_info() to ensure consistency between
            string representation and prompt formatting. The format is designed
            to be both human-readable and suitable for LLM processing.
        """
        return self.format_tool_info()
