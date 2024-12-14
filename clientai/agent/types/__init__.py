from enum import Enum
from typing import Any, Callable, Protocol, TypeVar

AgentInput = TypeVar("AgentInput")
AgentOutput = TypeVar("AgentOutput")
StepResult = TypeVar("StepResult")


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


class ToolProtocol(Protocol):
    func: Callable[..., Any]
    name: str
    description: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


__all__ = [
    "AgentInput",
    "AgentOutput",
    "StepResult",
    "ToolScope",
    "ToolProtocol",
]
