import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union

from clientai._typing import AIProviderProtocol
from clientai.client_ai import ClientAI

from ..config.defaults import DEFAULT_STEP_CONFIGS
from ..config.tools import ToolConfig
from ..core import Agent
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

    Args:
        tools: List of tools, which can be either callables or
               ToolConfig objects

    Returns:
        List of ToolConfig objects, or None if no tools provided

    Example:
        ```python
        def add(x: int, y: int) -> int:
            return x + y

        configs = _process_tools([add, ToolConfig(multiply)])
        ```
    """
    if not tools:
        return None

    logger.debug(f"Processing {len(tools)} tools")
    return [
        tool if isinstance(tool, ToolConfig) else ToolConfig(tool=tool)
        for tool in tools
    ]


def _create_tool_config(
    tool_selection_config: Optional[ToolSelectionConfig],
    tool_confidence: Optional[float],
    max_tools_per_step: Optional[int],
) -> Optional[ToolSelectionConfig]:
    """
    Create tool selection configuration from provided parameters.

    Args:
        tool_selection_config: Complete configuration object, if provided
        tool_confidence: Confidence threshold for tool selection
        max_tools_per_step: Maximum number of tools to use per step

    Returns:
        ToolSelectionConfig object or None if no configuration needed

    Raises:
        ValueError: If both tool_selection_config and
                    individual parameters are provided

    Example:
        ```python
        config = _create_tool_config(
            tool_selection_config=None,
            tool_confidence=0.8,
            max_tools_per_step=2
        )
        ```
    """
    if tool_selection_config:
        if any(p is not None for p in [tool_confidence, max_tools_per_step]):
            raise ValueError(
                "Cannot specify both tool_selection_config and individual "
                "tool parameters (tool_confidence, max_tools_per_step)"
            )
        return tool_selection_config

    if any(p is not None for p in [tool_confidence, max_tools_per_step]):
        config_params = {}
        if tool_confidence is not None:
            config_params["confidence_threshold"] = tool_confidence
        if max_tools_per_step is not None:
            config_params["max_tools_per_step"] = max_tools_per_step
        logger.debug(f"Creating tool config with params: {config_params}")
        return ToolSelectionConfig.create(**config_params)

    return None


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
    Create model configuration dictionary with
    appropriate defaults and overrides.

    Args:
        model: Name of the model to use
        system_prompt: System prompt for the model
        temperature: Optional temperature override
        top_p: Optional top_p override
        step_config: Step-specific configuration defaults
        stream: Whether to enable response streaming
        model_kwargs: Additional model configuration parameters

    Returns:
        Dictionary containing complete model configuration

    Example:
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
    """
    config = {
        "name": model,
        "system_prompt": system_prompt,
        "stream": stream,
        **model_kwargs,
    }

    if temperature is not None:
        config["temperature"] = temperature
    elif "temperature" in step_config:
        config["temperature"] = step_config["temperature"]

    if top_p is not None:
        config["top_p"] = top_p
    elif "top_p" in step_config:
        config["top_p"] = step_config["top_p"]

    logger.debug(f"Created model config: {config}")
    return config


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

    This factory function simplifies the creation of specialized agents
    that perform a single task. It handles configuration of the model,
    step type, and tools while providing sensible defaults based on
    the intended use case.

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
        ValueError: If inputs are invalid or configuration is inconsistent

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

        Simple agent with function tools:
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

        result = calculator.run("What is 5 plus 3, then multiplied by 2?")
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
    if not role or not isinstance(role, str):
        raise ValueError("Role must be a non-empty string")
    if not system_prompt or not isinstance(system_prompt, str):
        raise ValueError("System prompt must be a non-empty string")
    if not model or not isinstance(model, str):
        raise ValueError("Model must be a non-empty string")

    if temperature is not None and not 0 <= temperature <= 1:
        raise ValueError("Temperature must be between 0 and 1")
    if top_p is not None and not 0 <= top_p <= 1:
        raise ValueError("Top_p must be between 0 and 1")

    logger.debug(f"Creating {role} agent with model {model}")

    step_lower = step.lower()
    allowed_steps: Set[str] = {"think", "act", "observe", "synthesize"}
    if step_lower not in allowed_steps and not step.isidentifier():
        raise ValueError(
            f"Step must be one of {allowed_steps} or a valid Python identifier"
        )

    step_config = DEFAULT_STEP_CONFIGS.get(step_lower, {})
    logger.debug(f"Using step type: {step_lower}")

    model_config = _create_model_config(
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        step_config=step_config,
        stream=stream,
        model_kwargs=model_kwargs,
    )

    processed_tools = _process_tools(tools)
    tool_config = _create_tool_config(
        tool_selection_config=tool_selection_config,
        tool_confidence=tool_confidence,
        max_tools_per_step=max_tools_per_step,
    )

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
        """Agent that processes inputs using a single configured step."""

        @step_decorator(
            name=f"{role}_step", description=f"Execute {role} functionality"
        )
        def process(self, input_data: str) -> str:
            """Process input according to the configured behavior."""
            return input_data

    return SingleStepAgent(
        client=client,
        default_model=model_config,
        tools=processed_tools,
        tool_selection_config=tool_config,
        tool_model=tool_model,
        **model_kwargs,
    )
