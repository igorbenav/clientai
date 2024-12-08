import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from clientai._typing import AIProviderProtocol
from clientai.client_ai import ClientAI

from ..config.models import ModelConfig
from ..config.tools import ToolConfig
from ..exceptions import AgentError, ToolError, WorkflowError
from ..formatting import AgentFormatter
from ..tools import ToolSelectionConfig
from ..tools.registry import Tool, ToolRegistry
from ..tools.types import ToolScope
from .context import AgentContext
from .execution import StepExecutionEngine
from .workflow import WorkflowManager

logger = logging.getLogger(__name__)


class Agent:
    """
    A framework for creating and managing LLM-powered agents
    with automated tool selection.

    The Agent class provides a flexible framework for building AI agents that:

    - Execute multi-step workflows with LLM integration
    - Register and manage tools through decorators or direct registration
    - Automatically select and use appropriate tools
    - Maintain context and state across steps
    - Interact with language models for decision making

    Key Features:
        - Simple tool registration through decorators or direct methods
        - Automated tool selection with configurable confidence thresholds
        - Support for different models in main workflow vs tool selection
        - Flexible step configuration with decorators
        - Built-in context management
        - Comprehensive workflow control

    Tool Registration:
        Tools can be registered in two ways:
        1. Using the agent's register_tool decorator:
           ```python
           @agent.register_tool(
               name="TextProcessor",
               description="Formats text"
            )
           def process_text(text: str) -> str:
               return text.upper()
           ```

        2. Direct registration with register_tool():
           ```python
           def multiply(x: int, y: int) -> int:
               return x * y

           agent.register_tool(
               multiply,
               name="Multiplier",
               description="Multiplies numbers"
           )
           ```

    Example:
        Creating an agent with tools and steps:
        ```python
        class MyAgent(Agent):
            # Workflow step
            @think("analyze", use_tools=True)
            def analyze_data(self, input_data: str) -> str:
                return f"Analyze this data: {input_data}"

            @act("process")
            def process_analysis(self, analysis: str) -> str:
                return f"Process analysis results: {analysis}"

        # Initialize the agent
        agent = MyAgent(
            client=my_client,
            model="gpt-4",
            tool_confidence=0.7,
            tool_model="llama-2"
        )

        # Direct tool registration
        agent.register_tool(
            utility_function,
            name="Utility",
            description="Utility function"
        )

        # Run the agent
        result = agent.run("Input data")
        ```

    Attributes:
        context: Manages the agent's state and memory
        tool_registry: Registry of available tools
        execution_engine: Handles step execution
        workflow_manager: Manages workflow execution order
    """

    def __init__(
        self,
        client: ClientAI[AIProviderProtocol, Any, Any],
        default_model: Union[str, Dict[str, Any], ModelConfig],
        tools: Optional[List[ToolConfig]] = None,
        tool_selection_config: Optional[ToolSelectionConfig] = None,
        tool_confidence: Optional[float] = None,
        tool_model: Optional[Union[str, Dict[str, Any], ModelConfig]] = None,
        max_tools_per_step: Optional[int] = None,
        **default_model_kwargs: Any,
    ) -> None:
        """
        Initialize an Agent instance with specified configurations.

        This constructor sets up the agent with its core components
        and configurations. It allows for either detailed configuration
        via ToolSelectionConfig or simplified configuration via
        individual parameters.

        Args:
            client: The AI client for model interactions
            default_model: The primary model config for the agent. Can be:
                - A string (model name)
                - A dict with model parameters
                - A ModelConfig instance
            tools: Optional list of pre-configured tools to register
            tool_selection_config: Complete tool selection configuration
            tool_confidence: Confidence threshold for tool selection (0.0-1.0)
            tool_model: Model to use for tool selection decisions
            max_tools_per_step: Maximum number of tools to use in a single step
            **default_model_kwargs: Additional kwargs for the default model

        Raises:
            ValueError: If default_model is not specified or if both
                        tool_selection_config and individual tool
                        parameters are provided

        Example:
            ```python
            # Simple initialization
            agent = MyAgent(
                client=client,
                model="gpt-4",
                tool_confidence=0.8,
                tool_model="llama-2"
            )

            # Detailed initialization
            agent = MyAgent(
                client=client,
                default_model=ModelConfig(
                    name="gpt-4",
                    temperature=0.7
                ),
                tool_selection_config=ToolSelectionConfig(
                    confidence_threshold=0.8,
                    max_tools_per_step=3
                ),
                tools=[
                    ToolConfig(my_tool, ["think", "act"])
                ]
            )
            ```
        """
        try:
            if not default_model:
                raise ValueError("default_model must be specified")

            if tool_selection_config and any(
                x is not None
                for x in [tool_confidence, tool_model, max_tools_per_step]
            ):
                raise ValueError(
                    "Cannot specify both tool_selection_config and individual "
                    "tool parameters "
                    "(tool_confidence, tool_model, max_tools_per_step)"
                )

            self._client = client
            self._default_model_kwargs = default_model_kwargs
            self._default_model = self._create_model_config(default_model)

            if tool_selection_config:
                self._tool_selection_config = tool_selection_config
            else:
                config_params = {}
                if tool_confidence is not None:
                    config_params["confidence_threshold"] = tool_confidence
                if max_tools_per_step is not None:
                    config_params["max_tools_per_step"] = max_tools_per_step
                self._tool_selection_config = ToolSelectionConfig.create(
                    **config_params
                )

            self._tool_model = (
                self._create_model_config(tool_model)
                if tool_model is not None
                else self._default_model
            )

            self.context = AgentContext()
            self.tool_registry = ToolRegistry()
            self.execution_engine = StepExecutionEngine(
                client=self._client,
                default_model=self._default_model,
                default_kwargs=self._default_model_kwargs,
                tool_selection_config=self._tool_selection_config,
                tool_model=self._tool_model,
            )
            self.workflow_manager = WorkflowManager()

            if tools:
                for tool_config in tools:
                    self.tool_registry.register(tool_config)

            self._register_class_tools()
            self.workflow_manager.register_class_steps(self)

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise AgentError(f"Failed to initialize agent: {str(e)}") from e

    def _create_model_config(
        self, model: Union[str, Dict[str, Any], ModelConfig]
    ) -> ModelConfig:
        """
        Create a ModelConfig instance from various input types.

        Args:
            model: Model specification in one of three formats:
                - String: Model name
                - Dict: Model configuration parameters
                - ModelConfig: Existing configuration instance

        Returns:
            ModelConfig: A validated model configuration

        Raises:
            ValueError: If model specification is invalid
        """
        try:
            if isinstance(model, str):
                return ModelConfig(name=model)

            if isinstance(model, dict):
                if "name" not in model:
                    raise ValueError(
                        "Model configuration must include a 'name' parameter"
                    )
                try:
                    return ModelConfig(**model)
                except TypeError as e:
                    raise ValueError(
                        f"Invalid model configuration parameters: {str(e)}"
                    )

            if isinstance(model, ModelConfig):
                return model

            raise ValueError(
                "Model must be a string, dict with "
                "'name', or ModelConfig instance"
            )

        except ValueError as e:
            logger.error(f"Model configuration error: {e}")
            raise AgentError(str(e)) from e
        except Exception as e:
            logger.error(f"Unexpected error creating model configuration: {e}")
            raise AgentError(
                f"Unexpected error creating model configuration: {str(e)}"
            ) from e

    def _should_stream(self, stream_override: Optional[bool] = None) -> bool:
        """
        Determine if the workflow should return a streaming response.
        Can be overridden by explicit stream parameter.

        Args:
            stream_override: Optional bool to override step configuration

        Returns:
            bool: True if streaming should be enabled
        """
        try:
            if stream_override is not None:
                return stream_override

            try:
                steps = self.workflow_manager.get_steps()
            except Exception as e:
                raise ValueError(
                    f"Failed to retrieve workflow steps: {str(e)}"
                )

            if not steps:
                return False

            try:
                last_step = next(reversed(steps.values()))
                return getattr(last_step, "stream", False)
            except StopIteration:
                raise ValueError("No steps found in workflow")
            except AttributeError as e:
                raise ValueError(f"Invalid step configuration: {str(e)}")

        except ValueError as e:
            logger.error(f"Stream configuration error: {e}")
            raise AgentError(str(e)) from e
        except Exception as e:
            logger.error(
                f"Unexpected error determining stream configuration: {e}"
            )
            raise AgentError(
                f"Unexpected error determining stream configuration: {str(e)}"
            ) from e

    def _handle_streaming(
        self,
        result: Union[str, Iterator[str]],
        stream_override: Optional[bool] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Process the workflow result based on streaming configuration.

        Args:
            result: Raw result from workflow execution
            stream_override: Optional bool to override streaming configuration

        Returns:
            Union[str, Iterator[str]]: Either a streaming iterator
            or complete string based on configuration and override

        Example:
            ```python
            # Internal usage in run method
            final_result = self._handle_streaming(step_result, stream=False)
            ```
        """
        try:
            try:
                should_stream = self._should_stream(stream_override)
            except Exception as e:
                raise ValueError(
                    f"Failed to determine streaming configuration: {str(e)}"
                )

            if not should_stream:
                if isinstance(result, str):
                    return result
                try:
                    return "".join(list(result))
                except Exception as e:
                    raise ValueError(
                        f"Failed to join stream results: {str(e)}"
                    )

            if isinstance(result, str):
                return iter([result])

            if not hasattr(result, "__iter__"):
                raise ValueError(
                    "Result must be either a string or an iterator"
                )

            return result

        except ValueError as e:
            logger.error(f"Stream handling error: {e}")
            raise AgentError(str(e)) from e
        except Exception as e:
            logger.error(f"Unexpected error handling stream: {e}")
            raise AgentError(
                f"Unexpected error handling stream: {str(e)}"
            ) from e

    def use_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a registered tool by its name.

        Allows direct execution of a registered tool with provided arguments.
        This method is useful when you need to use a specific tool directly
        rather than relying on automated tool selection.

        Args:
            name: Name of the tool to execute
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            The result of the tool execution

        Raises:
            ValueError: If the tool is not found
            ClientAIError: If tool execution fails

        Example:
            ```python
            # Execute a tool directly
            result = agent.use_tool("calculator", x=5, y=3)
            print(result)  # Output: 8
            ```
        """
        try:
            tool = self.tool_registry.get(name)
            if not tool:
                raise ValueError(f"Tool '{name}' not found in registry")

            try:
                return tool(*args, **kwargs)
            except TypeError as e:
                raise ValueError(
                    f"Invalid arguments for tool '{name}': {str(e)}"
                )
            except Exception as e:
                raise ValueError(f"Tool execution failed: {str(e)}")

        except ValueError as e:
            logger.error(f"Tool usage error: {e}")
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.error(f"Unexpected error using tool '{name}': {e}")
            raise ToolError(
                f"Unexpected error using tool '{name}': {str(e)}"
            ) from e

    def get_tools(self, scope: Optional[str] = None) -> List[Tool]:
        """
        Retrieve tools available to the agent, optionally filtered by scope.

        Args:
            scope: Optional scope to filter tools by.
                   If None, returns all tools.

        Returns:
            List[Tool]: List of available tools matching the criteria

        Example:
            ```python
            # Get all tools
            all_tools = agent.get_tools()

            # Get tools for thinking steps
            think_tools = agent.get_tools("think")
            ```
        """
        return self.tool_registry.get_for_scope(scope)

    def run(
        self, input_data: Any, *, stream: Optional[bool] = None
    ) -> Union[str, Iterator[str]]:
        """
        Execute the agent's workflow with the provided input data.
        Streaming can be controlled by the stream parameter or
        step configuration.

        Args:
            input_data: The initial input to start the workflow
            stream: Optional bool to override streaming configuration.
                    If provided, overrides the last step's stream setting.

        Returns:
            Union[str, Iterator[str]]: Either a complete response string
            or a streaming iterator based on streaming configuration

        Raises:
            Exception: If workflow execution fails

        Example:
            ```python
            # Force streaming on
            for chunk in agent.run("Analyze this", stream=True):
                print(chunk, end="", flush=True)

            # Force streaming off
            result = agent.run("Analyze this", stream=False)
            print(result)

            # Use step configuration
            result = agent.run("Analyze this")  # Uses last step's setting
            ```
        """
        try:
            self.context.current_input = input_data
            try:
                result = self.workflow_manager.execute(
                    self,
                    input_data,
                    self.execution_engine,
                    stream_override=stream,
                )
            except Exception as e:
                raise ValueError(f"Workflow execution failed: {str(e)}")

            try:
                return self._handle_streaming(result, stream)
            except Exception as e:
                raise ValueError(f"Stream handling failed: {str(e)}")

        except ValueError as e:
            logger.error(f"Workflow execution error: {e}")
            raise WorkflowError(str(e)) from e
        except Exception as e:
            logger.error(f"Unexpected error during workflow execution: {e}")
            raise WorkflowError(
                f"Unexpected error during workflow execution: {str(e)}"
            ) from e

    def reset_context(self) -> None:
        """
        Reset the agent's context, clearing all memory and state.

        Example:
            ```python
            agent.reset_context()
            print(len(agent.context.memory))  # Output: 0
            ```
        """
        self.context.clear()

    def reset(self) -> None:
        """
        Perform a complete reset of the agent.

        Clears context, memory, and workflow state, essentially
        returning the agent to its initial state.

        Example:
            ```python
            agent.reset()
            ```
        """
        self.context.clear()
        self.workflow_manager.reset()

    def register_tool(
        self,
        tool: Union[Callable[..., Any], Tool],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scopes: Union[List[str], str] = "all",
    ) -> Tool:
        """
        Register a tool with the agent for use in specified workflow scopes.

        Provides a flexible way to register tools with
        the agent through multiple methods:

        1. Direct registration of functions
        2. Registration of pre-created Tool instances
        3. As a decorator for class methods

        The unified registration supports scope control and can be used both
        as a method and decorator.

        Args:
            tool: Function to register as a tool or a pre-created Tool instance
            name: Optional custom name for the tool
            description: Optional description of the tool's functionality
            scopes: List of scopes where the tool can be used, or a single
                    scope string. Valid scopes are:
                    "think", "act", "observe", "synthesize", "all"
                    Defaults to "all"

        Returns:
            Tool: The registered Tool instance

        Raises:
            ToolError: If tool validation or registration fails
            ValueError: If scopes are invalid or tool is already registered

        Examples:
            Direct function registration:
            ```python
            def add(x: int, y: int) -> int:
                return x + y

            tool = agent.register_tool(
                add,
                name="Calculator",
                description="Adds numbers",
                scopes=["think", "act"]
            )
            ```

            Register pre-created Tool:
            ```python
            my_tool = Tool.create(multiply, name="Multiplier")
            agent.register_tool(my_tool, scopes="all")
            ```

            As a decorator:
            ```python
            class MyAgent(Agent):
                @register_tool(
                    name="TextProcessor",
                    description="Processes text",
                    scopes=["act", "synthesize"]
                )
                def process_text(text: str) -> str:
                    return text.upper()
            ```

            Register with specific scopes:
            ```python
            agent.register_tool(
                utility_function,
                name="Utility",
                description="Utility function",
                scopes=["think", "observe"]
            )
            ```
        """
        try:
            # Convert single scope to list
            if isinstance(scopes, str):
                scopes = [scopes]

            # Validate and convert scopes
            try:
                tool_scopes = frozenset(ToolScope.from_str(s) for s in scopes)
            except ValueError as e:
                raise ValueError(f"Invalid tool scope: {str(e)}")

            # Create or validate tool instance
            if isinstance(tool, Tool):
                tool_instance = tool
            else:
                try:
                    tool_instance = Tool.create(
                        func=tool,
                        name=name,
                        description=description,
                    )
                except ValueError as e:
                    raise ValueError(f"Invalid tool function: {str(e)}")

            # Create and register tool configuration
            tool_config = ToolConfig(
                tool=tool_instance,
                scopes=tool_scopes,
                name=tool_instance.name,
                description=tool_instance.description,
            )

            try:
                self.tool_registry.register(tool_config)
            except ValueError as e:
                raise ValueError(f"Tool registration failed: {str(e)}")

            return tool_instance

        except ValueError as e:
            logger.error(f"Tool validation error: {e}")
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.error(f"Failed to register tool: {e}")
            raise ToolError(f"Failed to register tool: {str(e)}") from e

    def register_tool_decorator(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scopes: Union[List[str], str] = "all",
    ) -> Callable[[Callable[..., Any]], Tool]:
        """
        Create a decorator for registering tools with the agent.

        This method provides a decorator-based way to register tools,
        allowing for clean integration of tool registration with function
        definitions. It supports all the same functionality as direct
        registration, including scope control.

        Args:
            name: Optional custom name for the tool
            description: Optional description of the tool's functionality
            scopes: List of scopes or single scope string where the tool
                    can be used. Defaults to "all"

        Returns:
            Callable: A decorator function that registers the decorated
                      function as a tool

        Examples:
            Basic usage with scopes:
            ```python
            class MyAgent(Agent):
                @register_tool_decorator(
                    name="Calculator",
                    scopes=["think", "act"]
                )
                def add(x: int, y: int) -> int:
                    return x + y

                @register_tool_decorator(
                    name="TextProcessor",
                    description="Processes text input",
                    scopes="synthesize"
                )
                def process_text(text: str) -> str:
                    return text.upper()
            ```
        """

        def decorator(func: Callable[..., Any]) -> Tool:
            return self.register_tool(
                func, name=name, description=description, scopes=scopes
            )

        return decorator

    def _register_class_tools(self) -> None:
        """
        Register any tools defined as class methods using decorators.

        This internal method scans the class for methods decorated with
        @register_tool_decorator and registers them with the agent.
        It supports both standalone tools and class-bound tool methods.

        The method is called during agent initialization to ensure all
        decorated tools are properly registered.
        """
        try:
            logger.debug("Registering class-level tools")
            for name, attr in self.__class__.__dict__.items():
                if hasattr(attr, "_is_tool"):
                    logger.debug(f"Found class tool: {name}")
                    method = getattr(self, name)
                    scopes = getattr(attr, "_tool_scopes", "all")
                    self.register_tool(
                        tool=method,
                        name=getattr(attr, "_tool_name", name),
                        description=getattr(attr, "_tool_description", None),
                        scopes=scopes,
                    )
        except AttributeError as e:
            logger.error(
                f"Invalid tool attribute during class registration: {e}"
            )
            raise ToolError(f"Invalid tool attribute: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to register class tools: {e}")
            raise ToolError(f"Failed to register class tools: {str(e)}") from e

    def __str__(self) -> str:
        """
        Provide a formatted string representation of the agent.

        Returns:
            str: A human-readable description of the agent's configuration

        Example:
            ```python
            print(agent)  # Displays detailed agent configuration
            ```
        """
        formatter = AgentFormatter()
        return formatter.format_agent(self)
