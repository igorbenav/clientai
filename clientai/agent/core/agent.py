import logging
from typing import Any, Callable, Dict, List, Optional, Union

from clientai._typing import AIProviderProtocol
from clientai.client_ai import ClientAI
from clientai.exceptions import ClientAIError

from ..config.models import ModelConfig
from ..config.tools import ToolConfig
from ..formatting import AgentFormatter
from ..tools.registry import Tool, ToolRegistry
from ..tools.types import ToolScope
from .context import AgentContext
from .execution import StepExecutionEngine
from .workflow import WorkflowManager

logger = logging.getLogger(__name__)


class Agent:
    """
    A framework for creating and managing LLM-powered agents.

    The Agent class is designed to handle workflows with
    multiple execution steps, leveraging tools and context
    management to enhance task execution.

    Attributes:
        context: Tracks the agent's state and memory.
        provider: Interface for the underlying AI provider.
        execution_engine: Executes individual workflow steps.
        workflow_manager: Coordinates the workflow execution order.
        tool_registry: Handles the registration and use of tools.

    Methods:
        add_tool: Add a new tool to the agent.
        use_tool: Execute a registered tool by its name.
        get_tools: Retrieve tools filtered by scope.
        run: Execute the agent's workflow starting with the input data.
        reset_context: Clear the agent's memory and state.
        reset: Fully reset the agent, including context and workflow.

    Examples:
        Create an agent and define its steps:
        ```python
        class ExampleAgent(Agent):
            @think("greet", send_to_llm=False)
            def greeting_step(self, name: str) -> str:
                # This step runs locally and does not send a prompt to the LLM
                return f"Hello, {name}!"

            @act("farewell", send_to_llm=True)
            def farewell_step(self, name: str) -> str:
                # This step sends a prompt to the LLM for processing
                return f"Generate a farewell message for {name}."

        # Initialize the agent with tools
        agent = ExampleAgent(
            client,
            default_model="gpt-4",
            tools=[ToolConfig(calculator_tool, ["all"])]
        )

        # Execute the workflow
        result = agent.run("John")
        ```

        Define tools and use them in steps:
        ```python
        class MathAgent(Agent):
            @think("calculate", send_to_llm=False)
            def calculate_expression(self, expression: str) -> str:
                return self.use_tool("Calculator", expression)

        # Register a calculator tool
        agent = MathAgent(client, "gpt-4")
        agent.add_tool(
            tool=lambda x: str(eval(x)),
            scopes=["all"],
            name="Calculator",
            description="Evaluates mathematical expressions."
        )

        # Run the agent
        print(agent.run("2 + 2"))  # Output: 4
        ```
    """

    def __init__(
        self,
        client: ClientAI[AIProviderProtocol, Any, Any],
        default_model: Union[str, Dict[str, Any], ModelConfig],
        tools: Optional[List[ToolConfig]] = None,
        **default_model_kwargs: Any,
    ):
        """
        Initialize the Agent.

        Args:
            client: The AI client for API communication.
            default_model: Default LLM configuration.
            tools: List of tools for the agent.
            **default_model_kwargs: Additional parameters for the model.

        Raises:
            ValueError: If the default model is missing or invalid.
            ImportError: If required packages for the
                         AI provider are unavailable.

        Examples:
            Initialize an Agent:
            ```python
            client_ai = ClientAI(provider="openai")
            agent = Agent(client_ai, default_model="gpt-4")
            ```
        """
        if not default_model:
            raise ValueError("default_model must be specified")

        self._client = client
        self._default_model_kwargs = default_model_kwargs
        self._default_model = self._create_model_config(default_model)

        self.context = AgentContext()
        self.tool_registry = ToolRegistry()
        self.execution_engine = StepExecutionEngine(
            client=self._client,
            default_model=self._default_model,
            default_kwargs=self._default_model_kwargs,
        )
        self.workflow_manager = WorkflowManager()

        if tools:
            for tool_config in tools:
                self.tool_registry.register(tool_config)

        self.workflow_manager.register_class_steps(self)

    def _create_model_config(
        self, model: Union[str, Dict[str, Any], ModelConfig]
    ) -> ModelConfig:
        """Create a ModelConfig instance from various input types."""
        if isinstance(model, str):
            return ModelConfig(name=model)
        if isinstance(model, dict):
            if "name" not in model:
                raise ValueError(
                    "Model configuration must include a 'name' parameter"
                )
            return ModelConfig(**model)
        if isinstance(model, ModelConfig):
            return model
        raise ValueError(
            "default_model must be a string, dict with 'name', or ModelConfig"
        )

    def add_tool(
        self,
        tool: Union[Callable, Tool],
        scopes: Union[List[str], str, None] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Add a tool to the agent's registry.

        Args:
            tool: The tool to add, either as a function or Tool instance.
            scopes: Scope(s) where the tool can be used. Can be a single scope
                    string, list of scope strings, or None. Default ["all"]
            name: Optional custom name for the tool.
            description: Optional tool description.

        Raises:
            ValueError: If tool configuration is invalid
                        or if scope is invalid.

        Example:
            >>> agent.add_tool(
            ...     tool=my_function,
            ...     scopes=["think", "act"],
            ...     name="MyTool",
            ...     description="Does something useful"
            ... )
        """
        tool_scopes: List[ToolScope] = []

        if isinstance(scopes, str):
            tool_scopes = [ToolScope.from_str(scopes)]
        elif isinstance(scopes, list):
            tool_scopes = [ToolScope.from_str(s) for s in scopes]
        else:
            tool_scopes = [ToolScope.ALL]

        config = ToolConfig(
            tool=tool,
            scopes=frozenset(tool_scopes),
            name=name,
            description=description,
        )

        self.tool_registry.register(config)

    def use_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a registered tool by name.

        Args:
            name: Name of the tool to execute.
            *args: Positional arguments to pass to the tool.
            **kwargs: Keyword arguments to pass to the tool.

        Returns:
            The result of the tool execution.

        Raises:
            ValueError: If tool is not found.
            ClientAIError: If tool execution fails.
        """
        tool = self.tool_registry.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry")

        try:
            return tool(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Tool execution failed for '{name}': {e}")
            raise ClientAIError(
                f"Tool execution failed for '{name}': {str(e)}"
            ) from e

    def get_tools(self, scope: Optional[str] = None) -> List[Tool]:
        """
        Retrieve all tools registered with the agent,
        optionally filtered by scope.

        Args:
            scope: The scope to filter tools by. If None, retrieves all tools.

        Returns:
            List[Tool]: A list of tools available in the specified scope.

        Examples:
            Get all tools:
            ```python
            tools = agent.get_tools()
            print(tools)

            # Get tools for a specific scope
            scoped_tools = agent.get_tools("calculate")
            print(scoped_tools)
            ```
        """
        return self.tool_registry.get_for_scope(scope)

    def run(self, input_data: Any) -> Any:
        """
        Execute the agent's workflow starting with the provided input data.

        Args:
            input_data (Any): The initial input data to start the workflow.

        Returns:
            Any: The result of the workflow execution.

        Raises:
            Exception: If the workflow execution fails.

        Examples:
            Run the agent with input:
            ```python
            result = agent.run("Analyze this dataset")
            print(result)
            ```
        """
        self.context.current_input = input_data
        try:
            return self.workflow_manager.execute(
                self, input_data, self.execution_engine
            )
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

    def reset_context(self) -> None:
        """
        Clear the agent's context, removing all memory and state information.

        Examples:
            Reset the context:
            ```python
            agent.reset_context()
            print(agent.context.memory)  # Output: []
            ```
        """
        self.context.clear()

    def reset(self) -> None:
        """
        Clear the agent's context, removing all memory and state information.

        Examples:
            Reset the context:
            ```python
            agent.reset_context()
            print(agent.context.memory)  # Output: []
            ```
        """
        self.context.clear()
        self.workflow_manager.reset()

    def __str__(self) -> str:
        """
        Provide a formatted string representation of the agent.

        Returns:
            str: A human-readable summary of the agent and its configuration.

        Examples:
            Display the agent configuration:
            ```python
            print(agent)
            ```
        """
        formatter = AgentFormatter()
        return formatter.format_agent(self)
