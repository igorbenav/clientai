from typing import Any, Callable, List, Optional, Protocol, Union

from ..steps.base import Step
from ..tools.base import Tool
from .common import AgentInput, AgentOutput, StepResult, ToolResult


class StepProtocol(Protocol[StepResult]):
    """Protocol defining the interface for step execution.

    A protocol that defines the required methods for step execution, including
    validation and execution of step logic.

    Type Args:
        StepResult: The type of result returned by the step.

    Methods:
        execute: Execute the step with provided input data.
        validate: Validate that input data meets step requirements.
    """

    def execute(self, input_data: Any) -> StepResult:
        """Execute the step with provided input data.

        Args:
            input_data: Input data for step execution.

        Returns:
            StepResult: The result of step execution.
        """
        ...

    def validate(self, input_data: Any) -> bool:
        """Validate input data meets step requirements.

        Args:
            input_data: Input data to validate.

        Returns:
            bool: True if input is valid, False otherwise.
        """
        ...


class ToolProtocol(Protocol[ToolResult]):
    """Protocol defining the interface for tool execution.

    A protocol that defines the required methods for tool execution, including
    validation and calling of tool functions.

    Type Args:
        ToolResult: The type of result returned by the tool.

    Methods:
        __call__: Execute the tool with provided arguments.
        validate: Validate that arguments meet tool requirements.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> ToolResult:
        """Execute the tool with provided arguments.

        Args:
            *args: Positional arguments for tool execution.
            **kwargs: Keyword arguments for tool execution.

        Returns:
            ToolResult: The result of tool execution.
        """
        ...

    def validate(self, *args: Any, **kwargs: Any) -> bool:
        """Validate arguments meet tool requirements.

        Args:
            *args: Positional arguments to validate.
            **kwargs: Keyword arguments to validate.

        Returns:
            bool: True if arguments are valid, False otherwise.
        """
        ...


class AgentProtocol(Protocol[AgentInput, AgentOutput]):
    """Protocol defining the interface for agent execution.

    A protocol that defines the required methods for agent execution, including
    running workflows, managing tools, and maintaining state.

    Type Args:
        AgentInput: The type of input accepted by the agent.
        AgentOutput: The type of output produced by the agent.

    Methods:
        run: Execute the agent's workflow with provided input.
        register_tool: Register a new tool with the agent.
        get_tools: Retrieve registered tools, optionally filtered by scope.
        reset: Reset agent state.
    """

    def run(self, input_data: AgentInput) -> AgentOutput:
        """Execute the agent's workflow with provided input.

        Args:
            input_data: Input data for workflow execution.

        Returns:
            AgentOutput: The result of workflow execution.
        """
        ...

    def register_tool(
        self,
        tool: Union[Callable[..., Any], Tool, ToolProtocol[Any]],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scopes: Union[List[str], str] = "all",
    ) -> Tool:
        """Register a new tool with the agent.

        Args:
            tool: The tool to register, either as a callable or Tool instance.
            name: Optional name for the tool.
            description: Optional description of the tool.
            scopes: Scopes where the tool can be used.

        Returns:
            Tool: The registered tool instance.
        """
        ...

    def get_tools(self, scope: Optional[str] = None) -> List[Tool]:
        """Retrieve registered tools, optionally filtered by scope.

        Args:
            scope: Optional scope to filter tools by.

        Returns:
            List[Tool]: List of matching tools.
        """
        ...

    def reset(self) -> None:
        """Reset agent state.

        Clears all internal state and returns agent to initial configuration.
        """
        ...


class StepExecutionProtocol(Protocol):
    """Protocol defining the interface for step execution engines.

    A protocol that defines the required methods for executing workflow steps,
    including handling arguments and results.
    """

    def execute_step(self, step: Step, *args: Any, **kwargs: Any) -> Any:
        """Execute a workflow step.

        Args:
            step: The step to execute.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of step execution.
        """
        ...
