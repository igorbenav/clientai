from typing import Any, Callable, List, Protocol, Union

from ..steps.base import Step
from ..tools.base import Tool
from .common import AgentInput, AgentOutput, StepResult, ToolResult


class StepProtocol(Protocol[StepResult]):
    def execute(self, input_data: Any) -> StepResult:
        ...

    def validate(self, input_data: Any) -> bool:
        ...


class ToolProtocol(Protocol[ToolResult]):
    def __call__(self, *args: Any, **kwargs: Any) -> ToolResult:
        ...

    def validate(self, *args: Any, **kwargs: Any) -> bool:
        ...


class AgentProtocol(Protocol[AgentInput, AgentOutput]):
    def run(self, input_data: AgentInput) -> AgentOutput:
        ...

    def register_tool(
        self,
        tool: Union[Callable[..., Any], Tool, ToolProtocol[Any]],
        *,
        name: str | None = None,
        description: str | None = None,
        scopes: Union[List[str], str] = "all",
    ) -> Tool:
        ...

    def get_tools(self, scope: str | None = None) -> List[Tool]:
        ...

    def reset(self) -> None:
        ...


class StepExecutionProtocol(Protocol):
    def execute_step(
        self, step: Step, agent: Any, *args: Any, **kwargs: Any
    ) -> Any:
        ...
