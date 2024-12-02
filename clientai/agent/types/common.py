from typing import Any, TypeVar

AgentInput = TypeVar("AgentInput", bound=Any, contravariant=True)
AgentOutput = TypeVar("AgentOutput", bound=Any, covariant=True)
StepResult = TypeVar("StepResult", bound=Any, covariant=True)
ToolResult = TypeVar("ToolResult", bound=Any, covariant=True)
ModelResult = TypeVar("ModelResult", bound=Any)
