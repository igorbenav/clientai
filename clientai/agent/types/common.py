"""Type variables for agent input/output and results.

This module defines generic type variables used throughout the agent system
to provide type safety and clarity for agent operations, step executions,
tool usage, and model interactions.
"""

from typing import Any, TypeVar

AgentInput = TypeVar("AgentInput", bound=Any, contravariant=True)
"""Type variable representing input accepted by an agent.

This is a contravariant type variable that can represent any type of input
data that an agent can process.
"""

AgentOutput = TypeVar("AgentOutput", bound=Any, covariant=True)
"""Type variable representing output produced by an agent.

This is a covariant type variable that can represent any type of output
data that an agent can produce.
"""

StepResult = TypeVar("StepResult", bound=Any, covariant=True)
"""Type variable representing results from workflow steps.

This is a covariant type variable that can represent any type of result
produced by executing a workflow step.
"""

ToolResult = TypeVar("ToolResult", bound=Any, covariant=True)
"""Type variable representing results from tool executions.

This is a covariant type variable that can represent any type of result
produced by executing a tool.
"""

ModelResult = TypeVar("ModelResult", bound=Any)
"""Type variable representing results from model executions.

This type variable can represent any type of result produced by
executing a language model.
"""
