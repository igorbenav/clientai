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
    """A framework for creating and managing LLM-powered
    agents with automated tool selection.

    The Agent class provides a flexible system for building AI agents that can:
    - Execute multi-step workflows with LLM integration
    - Automatically select and use appropriate tools
    - Maintain context and state across steps
    - Handle streaming responses
    - Manage tool registration and scoping

    Attributes:
        context: Manages the agent's state and memory
        tool_registry: Registry of available tools
        execution_engine: Handles step execution
        workflow_manager: Manages workflow execution order

    Example:
        Create a simple agent with tools:
        ```python
        class AnalysisAgent(Agent):
            @think("analyze")
            def analyze_data(self, input_data: str) -> str:
                return f"Please analyze this data: {input_data}"

            @act("process")
            def process_results(self, analysis: str) -> str:
                return f"Process these results: {analysis}"

        # Initialize with tools
        agent = AnalysisAgent(
            client=client,
            default_model="gpt-4",
            tools=[calculator, text_processor],
            tool_confidence=0.8
        )

        # Run the agent
        result = agent.run("Analyze data: [1, 2, 3]")
        ```

        Using tools with custom registration:
        ```python
        class UtilityAgent(Agent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # Register tools directly
                self.register_tool(
                    utility_function,
                    name="Utility",
                    description="Utility function",
                    scopes=["think", "act"]
                )

            # Or use decorator
            @register_tool(
                name="Calculator",
                description="Performs calculations",
                scopes=["think"]
            )
            def calculate(self, x: int, y: int) -> int:
                return x + y
        ```

    Notes:
        - Tools can be registered via decorator or direct registration
        - Steps are executed in order of definition
        - Context maintains state across workflow execution
        - Tool selection is automatic based on confidence thresholds
        - Streaming can be controlled at step or run level
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
        max_history_size: Optional[int] = None,
        **default_model_kwargs: Any,
    ) -> None:
        """Initialize an Agent instance with specified configurations.

        Args:
            client: The AI client for model interactions
            default_model: Primary model config for the agent. Can be:
                - A string (model name)
                - A dict with model parameters
                - A ModelConfig instance
            tools: Optional list of tools to register. Can be:
                - Functions (with proper type hints and docstrings)
                - Tool instances
                - ToolConfig instances
            tool_selection_config: Complete tool selection configuration
            tool_confidence: Confidence threshold for tool selection (0.0-1.0)
            tool_model: Model to use for tool selection decisions
            max_tools_per_step: Maximum tools allowed per step
            max_history_size: Maximum number of previous interactions to
                              maintain in context history (defaults to 10)
            **default_model_kwargs: Additional kwargs for default model

        Raises:
            AgentError: If initialization fails due to:
                - Invalid model configuration
                - Incompatible tool configurations
                - Component initialization failure

        Example:
            Basic initialization:
            ```python
            agent = MyAgent(
                client=client,
                default_model="gpt-4",
                tool_confidence=0.8
            )
            ```

            Detailed configuration:
            ```python
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
                tool_model="llama-2"
            )
            ```

        Notes:
            - Cannot specify both tool_selection_config and
              individual tool parameters
            - Model can be specified as string, dict, or ModelConfig
            - Tools can be pre-configured or added after initialization
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

            self.context = AgentContext(
                max_history_size=max_history_size
                if max_history_size is not None
                else 10
            )
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
                for tool_item in tools:
                    if isinstance(tool_item, ToolConfig):
                        tool_config = tool_item
                    elif isinstance(tool_item, Tool):
                        tool_config = ToolConfig(tool=tool_item)
                    else:
                        try:
                            tool_instance = Tool.create(func=tool_item)
                            tool_config = ToolConfig(tool=tool_instance)
                        except Exception as e:  # pragma: no cover
                            raise ValueError(
                                f"Failed to create tool from "
                                f"function: {str(e)}"
                            )
                    self.tool_registry.register(tool_config)

            self._register_class_tools()
            self.workflow_manager.register_class_steps(self)

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise AgentError(f"Failed to initialize agent: {str(e)}") from e

    def _create_model_config(
        self, model: Union[str, Dict[str, Any], ModelConfig]
    ) -> ModelConfig:
        """Create a ModelConfig instance from various input types.

        Args:
            model: Model specification (string, dict, or ModelConfig)

        Returns:
            ModelConfig: Validated model configuration

        Raises:
            AgentError: If model specification is invalid
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
                except TypeError as e:  # pragma: no cover
                    raise ValueError(
                        f"Invalid model configuration parameters: {str(e)}"
                    )

            if isinstance(model, ModelConfig):
                return model

            raise ValueError(
                "Model must be a string, dict with "
                "'name', or ModelConfig instance"
            )  # pragma: no cover

        except ValueError as e:
            logger.error(f"Model configuration error: {e}")
            raise AgentError(str(e)) from e
        except Exception as e:  # pragma: no cover
            logger.error(f"Unexpected error creating model configuration: {e}")
            raise AgentError(
                f"Unexpected error creating model configuration: {str(e)}"
            ) from e

    def _should_stream(self, stream_override: Optional[bool] = None) -> bool:
        """Determine if workflow should return streaming response.

        Args:
            stream_override: Optional bool to override step configuration

        Returns:
            bool: Whether to enable streaming

        Raises:
            AgentError: If stream configuration cannot be determined
        """
        try:
            if stream_override is not None:
                return stream_override

            try:
                steps = self.workflow_manager.get_steps()
            except Exception as e:  # pragma: no cover
                raise ValueError(
                    f"Failed to retrieve workflow steps: {str(e)}"
                )

            if not steps:  # pragma: no cover
                return False

            try:
                last_step = next(reversed(steps.values()))
                return getattr(last_step, "stream", False)
            except StopIteration:  # pragma: no cover
                raise ValueError("No steps found in workflow")
            except AttributeError as e:  # pragma: no cover
                raise ValueError(f"Invalid step configuration: {str(e)}")

        except ValueError as e:  # pragma: no cover
            logger.error(f"Stream configuration error: {e}")
            raise AgentError(str(e)) from e
        except Exception as e:  # pragma: no cover
            logger.error(
                f"Unexpected error determining stream configuration: {e}"
            )
            raise AgentError(
                f"Unexpected error determining stream configuration: {str(e)}"
            ) from e

    def _handle_streaming(
        self,
        result: Union[str, Iterator[str], Any],
        stream_override: Optional[bool] = None,
    ) -> Union[str, Iterator[str], Any]:
        """Process workflow result based on streaming configuration.

        Args:
            result: Raw result from workflow execution. Could be:
                - String: Direct response
                - Iterator[str]: Streamed response chunks
                - Any: Validated result from a JSON output step
            stream_override: Optional streaming configuration override

        Returns:
            Processed result based on streaming settings:
                - String: For non-streaming responses
                - Iterator: For streamed responses
                - Any: For validated results from JSON output steps

        Raises:
            AgentError: If stream handling fails
        """
        try:
            try:
                should_stream = self._should_stream(stream_override)
            except Exception as e:  # pragma: no cover
                raise ValueError(
                    f"Failed to determine streaming configuration: {str(e)}"
                )

            if not isinstance(result, (str, Iterator)):  # noqa: UP038   # pragma: no cover
                return result

            if not should_stream:
                if isinstance(result, str):
                    return result
                try:
                    chunks = list(result)
                    return "".join(chunks)
                except Exception as e:  # pragma: no cover
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

        except ValueError as e:  # pragma: no cover
            logger.error(f"Stream handling error: {e}")
            raise AgentError(str(e)) from e
        except Exception as e:  # pragma: no cover
            logger.error(f"Unexpected error handling stream: {e}")
            raise AgentError(
                f"Unexpected error handling stream: {str(e)}"
            ) from e

    def use_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered tool by its name.

        Allows direct tool execution with provided arguments, bypassing
        automatic tool selection.

        Args:
            name: Name of the tool to execute
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            Any: Result of the tool execution

        Raises:
            ToolError: If tool execution fails or tool isn't found

        Example:
            Direct tool execution:
            ```python
            # Execute a calculator tool
            result = agent.use_tool("calculator", x=5, y=3)
            print(result)  # Output: 8

            # Execute a text processor
            result = agent.use_tool(
                "text_processor",
                text="hello",
                uppercase=True
            )
            print(result)  # Output: "HELLO"
            ```

        Notes:
            - Tool must be registered before use
            - Arguments must match tool's signature
            - Does not affect agent's tool usage history
        """
        try:
            tool = self.tool_registry.get(name)
            if not tool:
                raise ValueError(f"Tool '{name}' not found in registry")

            try:
                return tool(*args, **kwargs)
            except TypeError as e:  # pragma: no cover
                raise ValueError(
                    f"Invalid arguments for tool '{name}': {str(e)}"
                )
            except Exception as e:  # pragma: no cover
                raise ValueError(f"Tool execution failed: {str(e)}")

        except ValueError as e:
            logger.error(f"Tool usage error: {e}")
            raise ToolError(str(e)) from e
        except Exception as e:  # pragma: no cover
            logger.error(f"Unexpected error using tool '{name}': {e}")
            raise ToolError(
                f"Unexpected error using tool '{name}': {str(e)}"
            ) from e

    def get_tools(self, scope: Optional[str] = None) -> List[Tool]:
        """Retrieve tools available to the agent, optionally filtered by scope.

        Args:
            scope: Optional scope to filter tools by. Valid scopes:
                - "think": Analysis and reasoning tools
                - "act": Action and execution tools
                - "observe": Data gathering tools
                - "synthesize": Summary and integration tools
                - None: Return all tools

        Returns:
            List[Tool]: List of available tools matching the criteria

        Example:
            Get tools by scope:
            ```python
            # Get all tools
            all_tools = agent.get_tools()
            print(f"Total tools: {len(all_tools)}")

            # Get thinking tools
            think_tools = agent.get_tools("think")
            for tool in think_tools:
                print(f"- {tool.name}: {tool.description}")

            # Get action tools
            act_tools = agent.get_tools("act")
            print(f"Action tools: {[t.name for t in act_tools]}")
            ```
        """
        return self.tool_registry.get_for_scope(scope)

    def run(
        self, input_data: Any, *, stream: Optional[bool] = None
    ) -> Union[str, Iterator[str]]:
        """Execute the agent's workflow with the provided input data.

        Args:
            input_data: The initial input to process
            stream: Optional bool to override streaming configuration.
                If provided, overrides the last step's stream setting.

        Returns:
            Union[str, Iterator[str]]: Either:
                - Complete response string (streaming disabled)
                - Iterator of response chunks (streaming enabled)

        Raises:
            WorkflowError: If workflow execution fails
            StepError: If a required step fails
            ClientAIError: If LLM interaction fails

        Example:
            Basic execution:
            ```python
            # Without streaming
            result = agent.run("Analyze this data")
            print(result)

            # With streaming
            for chunk in agent.run("Process this", stream=True):
                print(chunk, end="", flush=True)

            # Use step configuration
            result = agent.run("Analyze this")  # Uses last step's setting
            ```

        Notes:
            - Streaming can be controlled by parameter or step configuration
            - Workflow executes steps in defined order
            - Context is updated after each step
            - Tool selection occurs automatically if enabled
        """
        try:
            self.context.set_input(input_data)

            try:
                result = self.workflow_manager.execute(
                    agent=self,
                    input_data=input_data,
                    engine=self.execution_engine,
                    stream_override=stream,
                )
            except Exception as e:
                raise ValueError(f"Workflow execution failed: {str(e)}")

            try:
                return self._handle_streaming(result, stream)
            except Exception as e:  # pragma: no cover
                raise ValueError(f"Stream handling failed: {str(e)}")

        except ValueError as e:
            logger.error(f"Workflow execution error: {e}")
            raise WorkflowError(str(e)) from e
        except Exception as e:  # pragma: no cover
            logger.error(f"Unexpected error during workflow execution: {e}")
            raise WorkflowError(
                f"Unexpected error during workflow execution: {str(e)}"
            ) from e

    def reset_context(self) -> None:
        """Reset the agent's context, clearing all memory and state.

        Example:
            ```python
            # After processing
            print(len(agent.context.memory))  # Output: 5

            # Reset context
            agent.reset_context()
            print(len(agent.context.memory))  # Output: 0
            print(agent.context.state)  # Output: {}
            ```

        Notes:
            - Clears memory, state, and results
            - Does not affect workflow or tool registration
            - Resets iteration counter
        """
        self.context.clear()

    def reset(self) -> None:
        """Perform a complete reset of the agent.

        Resets all state including context,
        workflow state, and iteration counters.

        Example:
            ```python
            # Complete reset
            agent.reset()
            print(len(agent.context.memory))  # Output: 0
            print(len(agent.workflow_manager.get_steps()))  # Output: 0
            ```

        Notes:
            - More comprehensive than reset_context
            - Clears workflow state
            - Maintains tool registration
            - Returns agent to initial state
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

        Example:
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
            if isinstance(scopes, str):
                scopes = [scopes]

            try:
                tool_scopes = frozenset(ToolScope.from_str(s) for s in scopes)
            except ValueError as e:
                raise ValueError(f"Invalid tool scope: {str(e)}")

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

            tool_config = ToolConfig(
                tool=tool_instance,
                scopes=tool_scopes,
                name=tool_instance.name,
                description=tool_instance.description,
            )

            try:
                self.tool_registry.register(tool_config)
            except ValueError as e:  # pragma: no cover
                raise ValueError(f"Tool registration failed: {str(e)}")

            return tool_instance

        except ValueError as e:  # pragma: no cover
            logger.error(f"Tool validation error: {e}")
            raise ToolError(str(e)) from e
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to register tool: {e}")
            raise ToolError(f"Failed to register tool: {str(e)}") from e

    def register_tool_decorator(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scopes: Union[List[str], str] = "all",
    ) -> Callable[[Callable[..., Any]], Tool]:
        """Create a decorator for registering tools with the agent.

        A decorator-based approach for registering tools,
        providing a clean way to integrate tool registration
        with method definitions.

        Args:
            name: Optional custom name for the tool
            description: Optional description of the tool's functionality
            scopes: List of scopes or single scope where tool can be used.
                Valid scopes: "think", "act", "observe", "synthesize", "all"

        Returns:
            Callable: A decorator function that registers
                      the decorated method as a tool

        Example:
            Basic tool registration with scopes:
            ```python
            class MyAgent(Agent):
                @register_tool_decorator(
                    name="Calculator",
                    description="Adds two numbers",
                    scopes=["think", "act"]
                )
                def add_numbers(self, x: int, y: int) -> int:
                    return x + y

                @register_tool_decorator(scopes="observe")
                def get_data(self, query: str) -> List[int]:
                    return [1, 2, 3]  # Example data
            ```

            Multiple tools with different scopes:
            ```python
            class AnalysisAgent(Agent):
                @register_tool_decorator(
                    name="TextAnalyzer",
                    scopes=["think"]
                )
                def analyze_text(self, text: str) -> dict:
                    return {"words": len(text.split())}

                @register_tool_decorator(
                    name="DataFormatter",
                    scopes=["synthesize"]
                )
                def format_data(self, data: dict) -> str:
                    return json.dumps(data, indent=2)
            ```

        Notes:
            - Decorated methods become available tools
            - Tool name defaults to method name if not provided
            - Description defaults to method docstring
            - Tools are registered during agent initialization
        """

        def decorator(func: Callable[..., Any]) -> Tool:
            return self.register_tool(
                func, name=name, description=description, scopes=scopes
            )

        return decorator

    def _register_class_tools(self) -> None:
        """Register tools defined as class methods using decorators.

        Raises:
            ToolError: If tool registration fails
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
        except AttributeError as e:  # pragma: no cover
            logger.error(
                f"Invalid tool attribute during class registration: {e}"
            )
            raise ToolError(f"Invalid tool attribute: {str(e)}") from e
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to register class tools: {e}")
            raise ToolError(f"Failed to register class tools: {str(e)}") from e

    def __str__(self) -> str:
        """Provide a formatted string representation of the agent.

        Returns:
            str: A human-readable description of the agent's configuration

        Example:
            ```python
            agent = MyAgent(client=client, model="gpt-4")
            print(agent)
            # Output example:
            # ╭─ MyAgent (openai provider)
            # │
            # │ Configuration:
            # │ ├─ Model: gpt-4
            # │ └─ Parameters: temperature=0.7
            # │
            # │ Workflow:
            # │ ├─ 1. analyze
            # │ │  ├─ Type: think
            # │ │  ├─ Model: gpt-4
            # │ │  └─ Description: Analyzes input data
            # ...
            ```
        """
        formatter = AgentFormatter()
        return formatter.format_agent(self)
