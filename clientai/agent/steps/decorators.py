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

from ..config.models import ModelConfig
from .base import Step
from .types import StepType

T = TypeVar("T")

STEP_TYPE_DEFAULTS = {
    StepType.THINK: {
        "temperature": 0.7,
        "top_p": 0.9,
    },
    StepType.ACT: {
        "temperature": 0.2,
        "top_p": 0.8,
    },
    StepType.OBSERVE: {
        "temperature": 0.1,
        "top_p": 0.5,
    },
    StepType.SYNTHESIZE: {
        "temperature": 0.4,
        "top_p": 0.7,
    },
}


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
    Generate a decorator to define a step of a specific type (e.g., THINK).

    This function creates a decorator that can be used to mark methods as steps
    in the agent's workflow. The generated decorator captures metadata about
    the step, such as its name, type, description, and whether it should
    interact with an LLM.

    Args:
        step_type: The type of step (THINK, ACT, OBSERVE, SYNTHESIZE).

    Returns:
        Callable: A decorator function for defining a step.

    Examples:
        Create a custom decorator:
        ```python
        from steps.types import StepType

        custom_decorator = _create_step_decorator(StepType.THINK)

        @custom_decorator("custom_step", description="A custom THINK step.")
        def custom_function(self, input_data: str) -> str:
            return f"Custom processing: {input_data}"
        ```

        Use generated decorators:
        ```python
        @think("example", description="Local THINK step.", send_to_llm=False)
        def example_function(self, input_data: str) -> str:
            return f"Processing locally: {input_data}"

        @think
        def another_example(self, input_data: str) -> str:
            return f"Analyze: {input_data}"
        ```
    """

    def decorator(
        name: Optional[Union[str, Callable[..., str]]] = None,
        *,
        model: Optional[Union[str, Dict[str, Any], ModelConfig]] = None,
        description: Optional[str] = None,
        send_to_llm: bool = True,
        json_output: bool = False,
    ) -> Union[StepFunction, Callable[[Callable[..., str]], StepFunction]]:
        """
        Decorator function that wraps step methods.

        Args:
            name: Optional name for the step. If not provided, the function
                 name is used. Can also be the function itself when used
                 without parameters.
            model: Optional model configuration for LLM interaction.
            description: Optional description of what the step does.
            send_to_llm: Whether this step should send its output to the LLM.
            json_output: Whether the LLM should format its response as JSON.

        Returns:
            Union[StepFunction, Callable[[Callable[..., str]], StepFunction]]:
                Either the wrapped function or a wrapper function, depending on
                how the decorator is used.
        """

        def wrapper(func: Callable[..., str]) -> StepFunction:
            wrapped = StepFunction(func)
            step_name = name if isinstance(name, str) else func.__name__

            wrapped._step_info = Step.create(
                func=func,
                step_type=step_type,
                name=step_name,
                description=description,
                llm_config=_create_model_config(model, step_type)
                if model and send_to_llm
                else None,
                send_to_llm=send_to_llm,
                json_output=json_output,
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


# Export step decorators
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
