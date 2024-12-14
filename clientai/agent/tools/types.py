from enum import Enum
from inspect import Parameter, signature
from typing import (
    Any,
    Callable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    Union,
    get_type_hints,
)


class ToolScope(str, Enum):
    """Enumeration of valid scopes for tool execution.

    Defines the contexts in which tools can be executed within
    the agent workflow. Tools can be restricted to specific step
    types or allowed in all steps.

    Attributes:
        THINK: Tools for analysis and reasoning steps.
        ACT: Tools for action and decision steps.
        OBSERVE: Tools for data gathering and observation steps.
        SYNTHESIZE: Tools for combining and summarizing steps.
        ALL: Tools available in all step types.

    Example:
        ```python
        # Create a scope
        scope = ToolScope.THINK
        print(scope)  # Output: "think"

        # Parse from string
        scope = ToolScope.from_str("act")
        print(scope == ToolScope.ACT)  # Output: True
        ```

    Raises:
        ValueError: When attempting to create from invalid scope string.
    """

    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    SYNTHESIZE = "synthesize"
    ALL = "all"

    @classmethod
    def from_str(cls, scope: str) -> "ToolScope":
        """Convert a string to a ToolScope enum value.

        Args:
            scope: String representation of the scope.

        Returns:
            ToolScope: The corresponding enum value.

        Raises:
            ValueError: If the string doesn't match a valid scope.

        Example:
            ```python
            scope = ToolScope.from_str("think")
            print(scope == ToolScope.THINK)  # Output: True

            # Raises ValueError
            scope = ToolScope.from_str("invalid")
            ```
        """
        try:
            return cls[scope.upper()]
        except KeyError:
            valid = [s.value for s in cls]
            raise ValueError(
                f"Invalid scope: '{scope}'. Must be one of: {', '.join(valid)}"
            )

    def __str__(self) -> str:
        return self.value


class ParameterInfo(NamedTuple):
    """Information about a tool parameter.

    Attributes:
        type_: The type annotation of the parameter.
        default: The default value of the parameter, if any.
    """

    type_: Any
    default: Any = None


class ToolSignature:
    """Represents the signature of a tool function.

    Captures the complete signature information of a tool function including
    its name, parameters (with types and defaults), and return type.
    Provides methods for signature analysis and formatting.

    Attributes:
        name: The name of the tool function.
        parameters: Tuple of parameter names and their type information.
        return_type: The return type annotation of the function.

    Example:
        ```python
        def example_tool(x: int, y: str = "default") -> bool:
            return True

        sig = ToolSignature.from_callable(example_tool)
        print(sig.format())
        # Output: "example_tool(x: int, y: str = 'default') -> bool"
        ```
    """

    def __init__(
        self,
        name: str,
        parameters: List[Tuple[str, ParameterInfo]],
        return_type: Any,
    ):
        self._name = name
        self._parameters = tuple(parameters)
        self._return_type = return_type
        self._str_repr: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> Tuple[Tuple[str, ParameterInfo], ...]:
        return self._parameters

    @property
    def return_type(self) -> Any:
        return self._return_type

    @classmethod
    def from_callable(
        cls, func: Callable, name: Optional[str] = None
    ) -> "ToolSignature":
        """Create a ToolSignature from a callable.

        Analyzes the callable's signature and type hints to create a complete
        signature representation.

        Args:
            func: The callable to analyze.
            name: Optional custom name for the signature.

        Returns:
            ToolSignature: A new signature instance.

        Example:
            ```python
            def example(x: int, y: str = "test") -> bool:
                return True

            sig = ToolSignature.from_callable(example, "custom_name")
            print(sig.name)  # Output: "custom_name"
            ```
        """
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        sig = signature(func)

        parameters: List[Tuple[str, ParameterInfo]] = []
        for param_name, param in sig.parameters.items():
            param_type = hints.get(param_name, Any)
            default = (
                param.default if param.default is not Parameter.empty else None
            )
            parameters.append((param_name, ParameterInfo(param_type, default)))

        return cls(
            name=name or func.__name__,
            parameters=parameters,
            return_type=hints.get("return", Any),
        )

    def format(self) -> str:
        """Format the signature as a string.

        Returns:
            str: A formatted string representation of the signature.

        Example:
            ```python
            def example(x: int, y: Optional[str] = None) -> bool:
                return True

            sig = ToolSignature.from_callable(example)
            print(sig.format())
            # Output: "example(x: int, y: Optional[str] = None) -> bool"
            ```
        """
        if self._str_repr is not None:
            return self._str_repr

        params = []
        for name, info in self._parameters:
            if info.type_ == Any:
                type_str = "Any"
            elif (
                hasattr(info.type_, "__origin__")
                and info.type_.__origin__ is Union
            ):
                if (
                    len(info.type_.__args__) == 2
                    and type(None) in info.type_.__args__
                ):
                    inner_type = next(
                        t for t in info.type_.__args__ if t is not type(None)
                    )
                    type_str = f"Optional[{inner_type.__name__}]"
                else:
                    type_str = str(info.type_).replace("typing.", "")
            else:
                type_str = (
                    info.type_.__name__
                    if hasattr(info.type_, "__name__")
                    else str(info.type_).replace("typing.", "")
                )

            if info.default is None:
                params.append(f"{name}: {type_str}")
            else:
                default_str = (
                    repr(info.default)
                    if isinstance(info.default, str)
                    else str(info.default)
                )
                params.append(f"{name}: {type_str} = {default_str}")

        if self._return_type == Any:
            return_str = "Any"
        else:
            return_str = (
                self._return_type.__name__
                if hasattr(self._return_type, "__name__")
                else str(self._return_type).replace("typing.", "")
            )

        self._str_repr = f"{self._name}({', '.join(params)}) -> {return_str}"
        return self._str_repr


class ToolProtocol(Protocol):
    """Protocol defining the interface that tools must implement.

    Defines the required attributes and methods that all tools must provide
    to be usable within the agent system.

    Attributes:
        func: The underlying callable implementing the tool's logic.
        name: The display name of the tool.
        description: Human-readable description of the tool's purpose.
        signature: Complete signature information for the tool.

    Methods:
        __call__: Execute the tool with provided arguments.
    """

    func: Callable[..., Any]
    name: str
    description: str
    signature: ToolSignature

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool with provided arguments.

        Args:
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            Any: The result of tool execution.
        """
        ...
