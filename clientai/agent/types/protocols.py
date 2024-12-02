from typing import Any, List, Protocol

from ..steps.base import Step
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

    def add_tool(self, tool: ToolProtocol[Any], scopes: List[str]) -> None:
        ...

    def get_tools(self, scope: str) -> List[ToolProtocol[Any]]:
        ...

    def reset(self) -> None:
        ...


class StepExecutionProtocol(Protocol):
    def execute_step(
        self, step: Step, agent: Any, *args: Any, **kwargs: Any
    ) -> Any:
        ...
