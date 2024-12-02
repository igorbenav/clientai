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
    get_type_hints,
)


class ToolScope(str, Enum):
    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    SYNTHESIZE = "synthesize"
    ALL = "all"

    @classmethod
    def from_str(cls, scope: str) -> "ToolScope":
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
    type_: Any
    default: Any = None


class ToolSignature:
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
        hints = get_type_hints(func)
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
        if self._str_repr is not None:
            return self._str_repr

        params = []
        for name, info in self._parameters:
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

        return_str = (
            self._return_type.__name__
            if hasattr(self._return_type, "__name__")
            else str(self._return_type).replace("typing.", "")
        )

        self._str_repr = f"{self._name}({', '.join(params)}) -> {return_str}"
        return self._str_repr


class ToolProtocol(Protocol):
    func: Callable[..., Any]
    name: str
    description: str
    signature: ToolSignature

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...
