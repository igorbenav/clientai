from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    TypeVar,
    Union,
    overload,
)

from ..config import StepConfig
from ..config.defaults import STEP_TYPE_DEFAULTS
from ..config.models import ModelConfig
from ..tools import ToolSelectionConfig
from .base import Step
from .types import StepType

T = TypeVar("T")


class StepFunction:
    """
    A wrapper class for step functions that allows setting metadata attributes.

    This class wraps agent step functions and provides a way to attach
    additional information like step configuration while maintaining
    proper type checking.

    Attributes:
        func: The original step function being wrapped.
        _step_info: Metadata about the step, including its type,
                                   configuration, and other properties.

    Example:
        ```python
        def example_step(self, input: str) -> str:
            return f"Processing: {input}"

        wrapped = StepFunction(example_step)
        wrapped._step_info = Step.create(
            func=example_step,
            step_type=StepType.THINK,
            name="example"
        )
        ```
    """

    def __init__(self, func: Callable[..., str]) -> None:
        """
        Initialize the step function wrapper.

        Args:
            func: The step function to wrap.
        """
        self.func = func
        self._step_info: Optional[Step] = None
        wraps(func)(self)

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """
        Execute the wrapped step function.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            str: The result of the step function execution.
        """
        return self.func(*args, **kwargs)


class BoundRunFunction:
    """
    Represents a run function bound to a specific instance.

    This class maintains the binding between a run function and its instance,
    preserving method attributes and proper execution context. It ensures that
    when the bound function is called, it receives the correct instance as
    its first argument.

    Attributes:
        func: The original function to be bound.
        instance: The instance to bind the function to.
        _is_run: Flag indicating this is a run method.
        _run_description: Optional description of the run behavior.
    """

    def __init__(self, func: Callable[..., Any], instance: Any) -> None:
        """
        Initialize a bound run function.

        Args:
            func: The function to bind.
            instance: The instance to bind the function to.
        """
        self.func = func
        self.instance = instance
        self._is_run = True
        self._run_description: Optional[str] = None
        for attr in ["__name__", "__doc__", "_is_run", "_run_description"]:
            if hasattr(func, attr):
                setattr(self, attr, getattr(func, attr))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the function with the bound instance.

        Automatically prepends the bound instance as the first argument (self)
        when calling the wrapped function.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of executing the bound function.
        """
        return self.func(*args, **kwargs)


class RunFunction:
    """
    A wrapper class for custom run methods in agents.

    This class implements Python's descriptor protocol to enable proper
    method binding when the wrapped function is accessed as a class
    attribute. It ensures that the function behaves correctly as an
    instance method while maintaining its custom attributes and metadata.

    Attributes:
        func: The original run function being wrapped.
        _is_run: Flag indicating this is a run method.
        _run_description: Optional description of the run behavior.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        """
        Initialize the run function wrapper.

        Args:
            func: The run function to wrap.
        """
        self.func = func
        self._is_run = True
        self._run_description: Optional[str] = None
        wraps(func)(self)

    def __get__(
        self, obj: Any, objtype: Optional[type] = None
    ) -> BoundRunFunction:
        """
        Support descriptor protocol for instance method binding.

        Implements Python's descriptor protocol to create bound methods when
        the function is accessed through an instance.

        Args:
            obj: The instance that the method is being accessed from.
            objtype: The type of the instance (not used).

        Returns:
            BoundRunFunction: A bound version of the run function.

        Raises:
            TypeError: If accessed without an instance (obj is None).
        """
        if obj is None:
            raise TypeError("Cannot access run method without instance")
        return BoundRunFunction(self.func, obj)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the wrapped run function directly.

        Note: This method is typically only called when the descriptor is used
        without being bound to an instance, which should raise a TypeError
        through __get__.

        Args:
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of the run function execution.
        """
        return self.func(*args, **kwargs)


def create_step_decorator(step_type: StepType):
    """
    Generate a decorator for defining workflow steps of a specific type.

    This factory function creates decorators (like @think, @act, etc.)
    that mark methods as specific types of workflow steps. The generated
    decorators handle both tool selection and LLM interaction configuration.

    Args:
        step_type: The type of step (THINK, ACT, OBSERVE, SYNTHESIZE)
                   this decorator will create

    Returns:
        A decorator function that can be used to mark methods as workflow steps

    Example:
        Creating a custom step decorator:
        ```python
        from steps.types import StepType

        analyze_step = create_step_decorator(StepType.THINK)

        class CustomAgent(Agent):
            @analyze_step(
                "analyze_data",
                description="Analyzes input data",
                tool_confidence=0.8
            )
            def analyze(self, data: str) -> str:
                return f"Please analyze: {data}"
        ```

    Note:
        The generated decorator can be used in two ways:
        1. With parameters
        2. As a bare decorator
    """

    def decorator(
        name: Optional[Union[str, Callable[..., str]]] = None,
        *,
        model: Optional[Union[str, Dict[str, Any], ModelConfig]] = None,
        description: Optional[str] = None,
        send_to_llm: bool = True,
        stream: bool = False,
        json_output: bool = False,
        use_tools: bool = True,
        tool_selection_config: Optional[ToolSelectionConfig] = None,
        tool_confidence: Optional[float] = None,
        tool_model: Optional[Union[str, Dict[str, Any], ModelConfig]] = None,
        max_tools_per_step: Optional[int] = None,
        step_config: Optional[StepConfig] = None,
    ) -> Union[StepFunction, Callable[[Callable[..., str]], StepFunction]]:
        """
        Core decorator implementation for workflow steps.

        This decorator configures how a step function should be executed
        within the workflow, including its interaction with the LLM and
        tool selection behavior.

        Args:
            name: The name of the step. If omitted, the function name is used.
                  Can be the function itself when used as a bare decorator.
            model: Model configuration for this specific step. Can be:
                - A string (model name)
                - A dictionary of model parameters
                - A ModelConfig instance
                If not provided, uses the agent's default model.
            description: Human-readable description of what this step does.
            send_to_llm: Whether this step's output should be sent to the
                         language model. Set to False for steps that process
                         data locally.
            stream: Whether to stream the LLM's response
            json_output: Whether the LLM should format its response as JSON.
            use_tools: Whether this step should use automatic tool selection.
            tool_selection_config: Complete tool selection configuration for
                                   this step. Mutually exclusive with
                                   individual tool parameters.
            tool_confidence: Minimum confidence for tool selection (0.0-1.0).
                             Overrides agent's default threshold for this step.
            tool_model: Specific model to use for tool selection in this step.
                       Can be a string, dict, or ModelConfig.
            max_tools_per_step: Maximum number of tools to use in this step.
                               Overrides the agent's default for this step.

        Returns:
            A decorated step function that will be executed
            as part of the workflow

        Raises:
            ValueError: If both tool_selection_config and individual tool
                        parameters are provided or if the configuration is
                        otherwise invalid

        Example:
            Basic usage with tool selection:
            ```python
            class MyAgent(Agent):
                @think(
                    "analyze",
                    description="Analyzes input data",
                    use_tools=True,
                    tool_confidence=0.8,
                    tool_model="gpt-4"
                )
                def analyze_data(self, input_data: str) -> str:
                    return f"Please analyze this data: {input_data}"

                @act(
                    "process",
                    description="Processes analysis results",
                    use_tools=False  # Disable tool selection for this step
                )
                def process_results(self, analysis: str) -> str:
                    return f"Process these results: {analysis}"
            ```

            Using complete tool selection configuration:
            ```python
            class MyAgent(Agent):
                @think(
                    "analyze",
                    description="Complex analysis step",
                    tool_selection_config=ToolSelectionConfig(
                        confidence_threshold=0.8,
                        max_tools_per_step=3,
                        prompt_template="Custom tool selection prompt: {task}"
                    )
                )
                def complex_analysis(self, data: str) -> str:
                    return f"Perform complex analysis on: {data}"
            ```

            Using as a bare decorator:
            ```python
            class MyAgent(Agent):
                @think  # No parameters, uses defaults
                def simple_analysis(self, data: str) -> str:
                    return f"Analyze: {data}"
            ```

        Note:
            - Tool parameters are mutually exclusive with tool_selection_config
            - The step's model config takes precedence over the agent's default
            - When used as a bare decorator, parameters use the default values
        """

        def wrapper(func: Callable[..., str]) -> StepFunction:
            """Inner wrapper that creates the StepFunction instance."""
            wrapped = StepFunction(func)
            step_name = name if isinstance(name, str) else func.__name__

            if tool_selection_config and any(
                x is not None
                for x in [tool_confidence, tool_model, max_tools_per_step]
            ):
                raise ValueError(
                    "Cannot specify both tool_selection_config and "
                    "individual tool parameters "
                    "(tool_confidence, tool_model, max_tools_per_step)"
                )

            final_tool_config = None
            if tool_selection_config:
                final_tool_config = tool_selection_config
            elif any(
                x is not None for x in [tool_confidence, max_tools_per_step]
            ):
                config_params = {}
                if tool_confidence is not None:
                    config_params["confidence_threshold"] = tool_confidence
                if max_tools_per_step is not None:
                    config_params["max_tools_per_step"] = max_tools_per_step
                final_tool_config = ToolSelectionConfig.create(**config_params)

            tool_model_config = None
            if tool_model is not None:
                if isinstance(tool_model, str):
                    tool_model_config = ModelConfig(name=tool_model)
                elif isinstance(tool_model, dict):
                    tool_model_config = ModelConfig(**tool_model)
                else:
                    tool_model_config = tool_model

            wrapped._step_info = Step.create(
                func=func,
                step_type=step_type,
                name=step_name,
                description=description,
                llm_config=_create_model_config(model, step_type)
                if model and send_to_llm
                else None,
                send_to_llm=send_to_llm,
                stream=stream,
                json_output=json_output,
                use_tools=use_tools,
                tool_selection_config=final_tool_config,
                tool_model=tool_model_config,
                step_config=step_config,
            )
            return wrapped

        if callable(name):
            return wrapper(name)
        return wrapper

    return decorator


def _create_model_config(
    model: Union[str, Dict[str, Any], ModelConfig], step_type: StepType
) -> ModelConfig:
    """
    Create a ModelConfig for a step, applying step-type-specific defaults.

    Takes a model configuration in various formats and merges it with default
    settings based on the step type. This ensures consistent model behavior
    for different types of steps while allowing customization.

    Args:
        model: The base model configuration, which can be:
            - A string (model name)
            - A dictionary of configuration parameters
            - A ModelConfig instance
        step_type: The type of step to apply defaults for.

    Returns:
        ModelConfig: The finalized ModelConfig with applied defaults.

    Raises:
        ValueError: If the model configuration is invalid or
                    missing required fields.

    Examples:
        ```python
        # From string
        config = _create_model_config("gpt-4", StepType.THINK)
        print(config.temperature)  # Output: 0.7

        # From dict
        config = _create_model_config(
            {"name": "gpt-4", "temperature": 0.5},
            StepType.ACT
        )
        ```
    """
    type_defaults = STEP_TYPE_DEFAULTS[step_type]

    if isinstance(model, str):
        return ModelConfig(
            name=model,
            return_full_response=False,
            stream=False,
            json_output=False,
            temperature=type_defaults.get("temperature"),
            top_p=type_defaults.get("top_p"),
        )
    elif isinstance(model, dict):
        if "name" not in model:
            raise ValueError("Model configuration must include 'name'")

        core_params = {
            "name": model["name"],
            "return_full_response": model.get("return_full_response", False),
            "stream": model.get("stream", False),
            "json_output": model.get("json_output", False),
        }

        extra_params = {
            k: v for k, v in model.items() if k not in ModelConfig.CORE_ATTRS
        }

        merged_extra = type_defaults.copy()
        merged_extra.update(extra_params)

        return ModelConfig(
            **core_params,
            temperature=merged_extra.get("temperature"),
            top_p=merged_extra.get("top_p"),
        )
    elif isinstance(model, ModelConfig):
        return model.merge(**type_defaults)
    else:
        return ModelConfig(
            name="default",
            return_full_response=False,
            stream=False,
            json_output=False,
            temperature=type_defaults.get("temperature"),
            top_p=type_defaults.get("top_p"),
        )


think = create_step_decorator(StepType.THINK)
act = create_step_decorator(StepType.ACT)
observe = create_step_decorator(StepType.OBSERVE)
synthesize = create_step_decorator(StepType.SYNTHESIZE)


@overload
def run(
    *, description: Optional[str] = None
) -> Callable[[Callable[..., T]], RunFunction]:
    ...


@overload
def run(func: Callable[..., T]) -> RunFunction:
    ...


def run(
    func: Optional[Callable[..., T]] = None,
    *,
    description: Optional[str] = None,
) -> Union[Callable[[Callable[..., T]], RunFunction], RunFunction]:
    """
    Decorator for defining a custom `run` method in an agent class.

    This decorator marks a method as the custom run implementation for
    an agent, optionally providing a description of its behavior. The
    decorated method will properly handle instance binding and maintain
    its metadata through Python's descriptor protocol.

    Args:
        func: The function to decorate (when used without parameters).
        description: A description of the custom run behavior.

    Returns:
        Union[Callable[[Callable[..., T]], RunFunction], RunFunction]:
            Either a decorator function or the decorated function.

    Examples:
        ```python
        class CustomAgent(Agent):
            @run(description="Custom workflow execution.")
            def custom_run(self, input_data: Any) -> Any:
                # Custom implementation
                return f"Custom execution for {input_data}"

            # Or without parameters
            @run
            def another_run(self, data: str) -> str:
                # Another implementation
                return f"Processing: {data}"
        ```
    """

    def decorator(f: Callable[..., T]) -> RunFunction:
        wrapped = RunFunction(f)
        wrapped._run_description = description
        return wrapped

    if func is None:
        return decorator

    return decorator(func)
