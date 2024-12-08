import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union

from clientai._typing import AIProviderProtocol
from clientai.client_ai import ClientAI

from ..config.defaults import DEFAULT_STEP_CONFIGS
from ..config.tools import ToolConfig
from ..core import Agent
from ..exceptions import AgentError, ToolError
from ..steps.decorators import (
    act,
    create_step_decorator,
    observe,
    synthesize,
    think,
)
from ..steps.types import StepType
from ..tools import ToolSelectionConfig

logger = logging.getLogger(__name__)


def _process_tools(
    tools: Optional[List[Union[Callable[..., Any], ToolConfig]]] = None,
) -> Optional[List[ToolConfig]]:
    """
    Convert a list of tools into ToolConfig objects.

    This helper function processes a list of tools, which can be either
    callable functions or pre-configured ToolConfig objects, and ensures
    they are all properly configured as ToolConfig instances. It performs
    validation on each tool and provides detailed error messages if
    problems are found.

    Args:
        tools: List of tools to process. Each tool can be either:
            - A callable function with proper type hints and docstring
            - A pre-configured ToolConfig object

    Returns:
        Optional[List[ToolConfig]]: List of validated ToolConfig objects,
        or None if no tools were provided

    Raises:
        ToolError: If any of the following occur:
            - A tool cannot be converted to ToolConfig
            - A tool lacks required type hints
            - A tool's configuration is invalid
            - Any unexpected error during tool processing

    Examples:
        Process simple function tools:
        ```python
        def add(x: int, y: int) -> int:
            '''Add two numbers.'''
            return x + y

        def multiply(x: int, y: int) -> int:
            '''Multiply two numbers.'''
            return x * y

        configs = _process_tools([add, multiply])
        ```

        Process mixed tools:
        ```python
        existing_config = ToolConfig(
            tool=divide,
            name="Divider",
            description="Divides numbers"
        )
        configs = _process_tools([add, existing_config])
        ```
    """
    if not tools:
        return None

    try:
        logger.debug(f"Processing {len(tools)} tools")
        processed_tools = []

        for tool in tools:
            try:
                processed_tool = (
                    tool
                    if isinstance(tool, ToolConfig)
                    else ToolConfig(tool=tool)
                )
                processed_tools.append(processed_tool)
            except Exception as e:
                raise ValueError(f"Failed to process tool {tool}: {str(e)}")

        return processed_tools

    except ValueError as e:
        logger.error(f"Tool processing error: {e}")
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error processing tools: {e}")
        raise ToolError(f"Unexpected error processing tools: {str(e)}") from e


def _create_tool_config(
    tool_selection_config: Optional[ToolSelectionConfig],
    tool_confidence: Optional[float],
    max_tools_per_step: Optional[int],
) -> Optional[ToolSelectionConfig]:
    """
    Create and validate tool selection configuration
    from provided parameters.

    This helper function creates a ToolSelectionConfig instance from either a
    complete configuration object or individual parameters. It ensures all
    parameters are valid and provides appropriate defaults when needed.

    Args:
        tool_selection_config: Complete configuration object, if provided.
            If this is provided, individual parameters must be None.
        tool_confidence: Confidence threshold for tool selection (0.0-1.0).
            Higher values mean tools must be more confidently selected.
        max_tools_per_step: Maximum number of tools that can be used in
            a single step. Must be at least 1 if specified.

    Returns:
        Optional[ToolSelectionConfig]: A validated configuration object,
        or None if no configuration is needed

    Raises:
        ValueError: If any of the following occur:
            - Both tool_selection_config and individual parameters are provided
            - tool_confidence is outside the valid range (0.0-1.0)
            - max_tools_per_step is less than 1
            - Configuration parameters are otherwise invalid
            - Any unexpected error during configuration creation

    Examples:
        Create config from complete object:
        ```python
        config = _create_tool_config(
            tool_selection_config=ToolSelectionConfig(
                confidence_threshold=0.8,
                max_tools_per_step=3
            ),
            tool_confidence=None,
            max_tools_per_step=None
        )
        ```

        Create config from individual parameters:
        ```python
        config = _create_tool_config(
            tool_selection_config=None,
            tool_confidence=0.7,
            max_tools_per_step=2
        )
        ```
    """
    try:
        if tool_selection_config:
            if any(
                p is not None for p in [tool_confidence, max_tools_per_step]
            ):
                raise ValueError(
                    "Cannot specify both tool_selection_config and individual "
                    "tool parameters (tool_confidence, max_tools_per_step)"
                )
            return tool_selection_config

        if any(p is not None for p in [tool_confidence, max_tools_per_step]):
            config_params = {}

            if tool_confidence is not None:
                if not 0 <= tool_confidence <= 1:
                    raise ValueError(
                        "Tool confidence must be between 0.0 and 1.0"
                    )
                config_params["confidence_threshold"] = tool_confidence

            if max_tools_per_step is not None:
                if max_tools_per_step < 1:
                    raise ValueError(
                        "Maximum tools per step must be at least 1"
                    )
                config_params["max_tools_per_step"] = max_tools_per_step

            logger.debug(f"Creating tool config with params: {config_params}")
            return ToolSelectionConfig.create(**config_params)

        return None

    except ValueError as e:
        logger.error(f"Tool configuration error: {e}")
        raise ValueError(str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error creating tool configuration: {e}")
        raise ValueError(
            f"Unexpected error creating tool configuration: {str(e)}"
        ) from e


def _create_model_config(
    model: str,
    system_prompt: str,
    temperature: Optional[float],
    top_p: Optional[float],
    step_config: Dict[str, Any],
    stream: bool,
    model_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create and validate model configuration
    with appropriate defaults and overrides.

    This helper function creates a complete model configuration dictionary,
    applying defaults from step configuration and validating all parameters.
    It ensures that core parameters cannot be overridden through model_kwargs
    and that all parameters are within valid ranges.

    Args:
        model: Name of the model to use (e.g., "gpt-4")
        system_prompt: System prompt defining the model's behavior
        temperature: Optional temperature setting (0.0-1.0) for
            controlling randomness
        top_p: Optional top_p setting (0.0-1.0) for nucleus sampling
        step_config: Default configuration for the step type
        stream: Whether to enable response streaming
        model_kwargs: Additional model parameters to include

    Returns:
        Dict[str, Any]: Complete model configuration dictionary

    Raises:
        ValueError: If any of the following occur:
            - Core parameters are attempted to be overridden in model_kwargs
            - Temperature or top_p are outside valid ranges
            - Configuration parameters are otherwise invalid
            - Any unexpected error during configuration creation

    Examples:
        Basic configuration:
        ```python
        config = _create_model_config(
            model="gpt-4",
            system_prompt="You are a helpful assistant",
            temperature=0.7,
            top_p=None,
            step_config=DEFAULT_STEP_CONFIGS["think"],
            stream=False,
            model_kwargs={}
        )
        ```

        Configuration with custom parameters:
        ```python
        config = _create_model_config(
            model="gpt-4",
            system_prompt="You are a helpful assistant",
            temperature=0.4,
            top_p=0.9,
            step_config={},
            stream=True,
            model_kwargs={
                "presence_penalty": 0.6,
                "frequency_penalty": 0.3
            }
        )
        ```
    """
    try:
        config = {
            "name": model,
            "system_prompt": system_prompt,
            "stream": stream,
        }

        # Validate model kwargs
        invalid_kwargs = [
            k for k in model_kwargs if k in ["name", "system_prompt", "stream"]
        ]
        if invalid_kwargs:
            raise ValueError(
                f"Cannot override core parameters in model_kwargs: "
                f"{', '.join(invalid_kwargs)}"
            )

        config.update(model_kwargs)

        if temperature is not None:
            if not 0 <= temperature <= 1:
                raise ValueError("Temperature must be between 0.0 and 1.0")
            config["temperature"] = temperature
        elif "temperature" in step_config:
            config["temperature"] = step_config["temperature"]

        if top_p is not None:
            if not 0 <= top_p <= 1:
                raise ValueError("Top_p must be between 0.0 and 1.0")
            config["top_p"] = top_p
        elif "top_p" in step_config:
            config["top_p"] = step_config["top_p"]

        logger.debug(f"Created model config: {config}")
        return config

    except ValueError as e:
        logger.error(f"Model configuration error: {e}")
        raise ValueError(str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error creating model configuration: {e}")
        raise ValueError(
            f"Unexpected error creating model configuration: {str(e)}"
        ) from e


def create_agent(
    client: ClientAI[AIProviderProtocol, Any, Any],
    role: str,
    system_prompt: str,
    model: str,
    step: str = "act",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: bool = False,
    tools: Optional[List[Union[Callable[..., Any], ToolConfig]]] = None,
    tool_selection_config: Optional[ToolSelectionConfig] = None,
    tool_confidence: Optional[float] = None,
    tool_model: Optional[str] = None,
    max_tools_per_step: Optional[int] = None,
    **model_kwargs: Any,
) -> Agent:
    """
    Create a single-step agent with minimal configuration.

    This factory function simplifies the creation of specialized agents that
    perform a single task. It handles configuration of the model, step type,
    and tools while providing sensible defaults based on the intended use case.

    The function supports both simple configurations for basic use cases and
    detailed configurations for more complex requirements.

    Args:
        client: The AI client for model interactions
        role: The role of the agent (e.g., "translator", "analyzer")
        system_prompt: The system prompt that defines the agent's behavior
        model: Name of the model to use (e.g., "gpt-4")
        step: Type of step to create. Can be:
            - "think": For analysis and reasoning (default temp=0.7)
            - "act": For decisive actions (default temp=0.2)
            - "observe": For data gathering (default temp=0.1)
            - "synthesize": For summarizing (default temp=0.4)
            - Any custom string for custom step types with no defaults
        temperature: Optional temperature override for the model (0.0-1.0)
        top_p: Optional top_p override for the model (0.0-1.0)
        stream: Whether to stream the model's response
        tools: Optional list of tools to use. Can be either:
            - Functions with proper type hints and docstrings
            - ToolConfig objects for more control
        tool_selection_config: Complete tool selection configuration
        tool_confidence: Confidence threshold for tool selection (0.0-1.0)
        tool_model: Name of model to use for tool selection
        max_tools_per_step: Maximum number of tools to use in a single step
        **model_kwargs: Additional model configuration parameters

    Returns:
        Agent: A configured agent instance ready to process inputs

    Raises:
        AgentError: If any of the following occur:
            - Required parameters are missing or invalid
            - Configuration validation fails
            - Tool processing fails
            - Agent initialization fails
            - Any unexpected error during agent creation

    Examples:
        Basic translation agent:
        ```python
        translator = create_agent(
            client=client,
            role="translator",
            system_prompt="You are a helpful translation assistant. "
                        "Translate the input text to French.",
            model="gpt-4"
        )

        result = translator.run("Hello world!")
        print(result)  # Output: "Bonjour le monde!"
        ```

        Analysis agent with specific configuration:
        ```python
        analyzer = create_agent(
            client=client,
            role="analyzer",
            system_prompt="Analyze the input data and provide insights.",
            model="gpt-4",
            step="think",  # Use thinking step type
            temperature=0.4,
            tools=[add, multiply],
            tool_confidence=0.8
        )

        result = analyzer.run("Analyze sales data: [1000, 1200, 950]")
        ```

        Agent with tool configuration:
        ```python
        def add(x: int, y: int) -> int:
            '''Add two numbers.'''
            return x + y

        def multiply(x: int, y: int) -> int:
            '''Multiply two numbers.'''
            return x * y

        calculator = create_agent(
            client=client,
            role="calculator",
            system_prompt="You are an assistant. Use tools for calculations.",
            model="gpt-4",
            tools=[add, multiply]  # Just pass functions directly
        )

        result = calculator.run(
            "What is 5 plus 3, then multiplied by 2?"
        )
        ```

        Streaming agent with custom settings:
        ```python
        validator = create_agent(
            client=client,
            role="validator",
            system_prompt="Validate the input against specified rules.",
            model="gpt-4",
            temperature=0.1,
            top_p=0.95,
            stream=True
        )
        ```

    Note:
        - Temperature and top_p have step defaults that can be overridden
        - Custom step types default to ACT behavior without default parameters
        - Functions passed as tools should have type hints and docstrings
        - Tool configuration options are mutually exclusive (can't use both
          tool_selection_config and individual parameters)
    """
    try:
        if not role or not isinstance(role, str):
            raise ValueError("Role must be a non-empty string")
        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError("System prompt must be a non-empty string")
        if not model or not isinstance(model, str):
            raise ValueError("Model must be a non-empty string")

        logger.debug(f"Creating {role} agent with model {model}")

        step_lower = step.lower()
        allowed_steps: Set[str] = {"think", "act", "observe", "synthesize"}
        if step_lower not in allowed_steps and not step.isidentifier():
            raise ValueError(
                f"Step must be one of {allowed_steps} "
                "or a valid Python identifier"
            )

        step_config = DEFAULT_STEP_CONFIGS.get(step_lower, {})
        logger.debug(f"Using step type: {step_lower}")

        try:
            model_config = _create_model_config(
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                step_config=step_config,
                stream=stream,
                model_kwargs=model_kwargs,
            )
        except ValueError as e:
            raise ValueError(f"Invalid model configuration: {str(e)}")

        try:
            processed_tools = _process_tools(tools)
        except ToolError as e:
            raise ValueError(f"Invalid tool configuration: {str(e)}")

        try:
            tool_config = _create_tool_config(
                tool_selection_config=tool_selection_config,
                tool_confidence=tool_confidence,
                max_tools_per_step=max_tools_per_step,
            )
        except ValueError as e:
            raise ValueError(f"Invalid tool selection configuration: {str(e)}")

        if step_lower in DEFAULT_STEP_CONFIGS:
            step_decorator = {
                "think": think,
                "act": act,
                "observe": observe,
                "synthesize": synthesize,
            }[step_lower]
        else:
            step_decorator = create_step_decorator(StepType.ACT)

        class SingleStepAgent(Agent):
            """
            Agent that processes inputs using a single configured step.

            This specialized agent class implements a single processing step
            based on the configured step type. It inherits the full
            capabilities of the base Agent class while providing a
            simplified interface focused on a specific task.

            The step type and its configuration are determined by the factory
            function parameters, allowing the agent to be optimized for
            different types of tasks (thinking, acting, observing, or
            synthesizing).

            Attributes:
                Inherits all attributes from the base Agent class.

            Methods:
                process: The main processing step that handles input data.
                All other methods are inherited from the base Agent class.
            """

            @step_decorator(
                name=f"{role}_step",
                description=f"Execute {role} functionality",
                stream=stream,
            )
            def process(self, input_data: str) -> str:
                """
                Process input according to the configured behavior.

                This method implements the core functionality of the agent,
                processing input data according to the role and configuration
                specified during agent creation.

                Args:
                    input_data: The input string to process

                Returns:
                    str: The processed result

                Note:
                    The actual processing is determined by the agent's
                    configuration, including:
                    - The step type (think/act/observe/synthesize)
                    - The system prompt
                    - The selected model and its parameters
                    - Any available tools and their selection criteria
                """
                return input_data

        try:
            return SingleStepAgent(
                client=client,
                default_model=model_config,
                tools=processed_tools,
                tool_selection_config=tool_config,
                tool_model=tool_model,
                **model_kwargs,
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize agent: {str(e)}")

    except ValueError as e:
        logger.error(f"Agent creation error: {e}")
        raise AgentError(str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error creating agent: {e}")
        raise AgentError(f"Unexpected error creating agent: {str(e)}") from e
