from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    Optional,
    Type,
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


class StepFunction(Generic[T]):
    """A wrapper class for step functions that maintains
    metadata and execution context.

    Wraps agent step functions while preserving their metadata and allowing
    attachment of additional step information through the _step_info attribute.

    Attributes:
        func: The original step function being wrapped
        _step_info: Optional Step instance containing step metadata

    Example:
        Create a wrapped step function:
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

    def __init__(self, func: Callable[..., T]) -> None:
        """
        Initialize the step function wrapper.

        Args:
            func: The step function to wrap.
        """
        self.func = func
        self._step_info: Optional[Step] = None
        wraps(func)(self)

    def __call__(
        self, instance: Optional[Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute the step function with instance binding.

        Args:
            instance: The agent instance the step is being called from.
                      If None, executes the raw function without engine
                      involvement.
            *args: Positional arguments to pass to the step
            **kwargs: Keyword arguments to pass to the step

        Returns:
            Any: Either:
                - Result from execution_engine if called from instance
                  with step info
                - Raw function result if called without instance or step info

        Example:
            When called from an agent instance:
            ```python
            result = step(
                agent_instance,
                "input data"
            )  # Uses execution engine
            ```

            When called directly:
            ```python
            result = step(
                None,
                "input data"
            )  # Calls raw function
            ```
        """
        if instance is None:
            return self.func(*args, **kwargs)

        if self._step_info:
            return instance.execution_engine.execute_step(
                self._step_info, *args, **kwargs
            )
        return self.func(instance, *args, **kwargs)

    def __get__(
        self, instance: Optional[object], owner: Optional[type]
    ) -> Union["StepFunction[T]", Callable[..., T]]:
        """
        Make steps behave like instance methods via the descriptor protocol.

        When a step is accessed on an agent instance, returns
        a bound method that automatically passes the instance through
        __call__ for engine execution. When accessed on the class,
        returns the StepFunction itself.

        Args:
            instance: The agent instance accessing the step.
                      None if accessed on class.
            owner: The agent class the step is defined on.

        Returns:
            Union[StepFunction, Callable[..., str]]:
                - If accessed on class (instance=None), returns
                  the StepFunction itself
                - If accessed on instance, returns a callable
                  that passes the instance through __call__
                  for engine-managed execution

        Example:
            ```python
            # On instance - returns bound method that uses engine
            agent.analyze_data("input")

            # On class - returns raw StepFunction
            AgentClass.analyze_data
            ```
        """
        if instance is None:
            return self
        return lambda *args, **kwargs: self(instance, *args, **kwargs)


class BoundRunFunction:
    """Run function bound to a specific agent instance.

    Maintains the binding between a run function and its agent instance while
    preserving method attributes and proper execution context.

    Attributes:
        func: The original function to be bound
        instance: The agent instance to bind to
        _is_run: Flag indicating this is a run method
        _run_description: Optional description of the run behavior

    Example:
        Create bound function:
        ```python
        bound = BoundRunFunction(run_method, agent_instance)
        result = bound("input data")  # Executes with proper agent binding
        ```
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
        return self.func(self.instance, *args, **kwargs)


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


def create_step_decorator(
    step_type: StepType,
) -> Callable[
    ...,
    Union[
        StepFunction[Union[str, Iterator[str], Any]],
        Callable[
            [Callable[..., Union[str, Iterator[str], Any]]],
            StepFunction[Union[str, Iterator[str], Any]],
        ],
    ],
]:
    """Generate a decorator for defining workflow steps of a specific type.

    Creates specialized decorators (like @think, @act) that mark methods as
    workflow steps with specific configurations and types.

    Args:
        step_type: The type of step (THINK, ACT, OBSERVE, SYNTHESIZE)
                   this decorator creates

    Returns:
        A decorator function that can be used to mark methods as workflow steps

    Example:
        Create custom step decorator:
        ```python
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

    Notes:
        - Generated decorators support both parameterized and bare usage
        - Decorators handle both tool selection and LLM configuration
        - Step type influences default configuration values
    """

    def decorator(
        name: Optional[
            Union[str, Callable[..., Union[str, Iterator[str], Any]]]
        ] = None,
        *,
        model: Optional[Union[str, Dict[str, Any], ModelConfig]] = None,
        description: Optional[str] = None,
        send_to_llm: bool = True,
        stream: bool = False,
        json_output: bool = False,
        return_type: Optional[Type[Any]] = None,
        return_full_response: bool = False,
        use_tools: bool = True,
        tool_selection_config: Optional[ToolSelectionConfig] = None,
        tool_confidence: Optional[float] = None,
        tool_model: Optional[Union[str, Dict[str, Any], ModelConfig]] = None,
        max_tools_per_step: Optional[int] = None,
        step_config: Optional[StepConfig] = None,
    ) -> Union[
        StepFunction[Union[str, Iterator[str], Any]],
        Callable[
            [Callable[..., Union[str, Iterator[str], Any]]],
            StepFunction[Union[str, Iterator[str], Any]],
        ],
    ]:
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
            stream: Whether to stream the LLM's response.
                Cannot be used with json_output.
            json_output: Whether to validate the output against return_type.
                Cannot be used with stream.
            return_type: Type to validate against when json_output=True.
                Must be a Pydantic model.
            return_full_response: Whether to return the complete API response.
                Cannot be used with json_output=True.
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
            ValueError: If:
                - Both stream and json_output are True
                - return_type is specified without json_output=True
                - Configuration parameters are invalid

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

            With validation:
            ```python
            from pydantic import BaseModel

            class Response(BaseModel):
                score: float
                summary: str

            class MyAgent(Agent):
                @think(
                    "analyze",
                    description="Analyzes data",
                    json_output=True,
                    return_type=Response
                )
                def analyze_data(self, input_data: str) -> Response:
                    return f"Please analyze: {input_data}"
            ```
        Note:
            - Tool parameters are mutually exclusive with tool_selection_config
            - The step's model config takes precedence over the agent's default
            - When used as a bare decorator, parameters use the default values
            - Streaming and validation are mutually exclusive
            - Validation requires a Pydantic model as return_type
        """

        def wrapper(
            func: Callable[..., Union[str, Iterator[str], Any]],
        ) -> StepFunction[Union[str, Iterator[str], Any]]:
            """Inner wrapper that creates the StepFunction instance."""
            step_name = name if isinstance(name, str) else func.__name__

            if return_type and stream:
                raise ValueError(
                    f"Step '{step_name}' cannot use both streaming and data "
                    "validation. These options are mutually exclusive."
                )

            if json_output and return_full_response:
                raise ValueError(
                    f"Step '{step_name}' cannot use both JSON validation and "
                    "full response return. These options are mutually "
                    "exclusive."
                )

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

            if return_type is not None and json_output is not True:
                raise ValueError(
                    "`json_output` should be set to `True` "
                    "when using `return_type`"
                )

            wrapped = StepFunction(func)

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
                return_type=return_type,
                return_full_response=return_full_response,
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

    Example:
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
"""Decorator for creating thinking/analysis steps.

Example:
    ```python
    @think("analyze", description="Analyzes input data")
    def analyze_data(self, data: str) -> str:
        return f"Analysis task: {data}"
    ```

    ```python
    # Standard usage
    @think("analyze", description="Analyzes input data")
    def analyze_data(self, data: str) -> str:
        return f"Analysis task: {data}"

    # With validation
    from pydantic import BaseModel

    class Analysis(BaseModel):
        score: float
        findings: str

    @think(
        "analyze",
        json_output=True,
        return_type=Analysis
    )
    def analyze_data(self, data: str) -> Analysis:
        return Analysis(score=0.8, findings="Found X")
    ```
"""

act = create_step_decorator(StepType.ACT)
"""Decorator for creating action/execution steps.

Example:
    ```python
    # Standard usage
    @act("process", description="Processes analyzed data")
    def process_data(self, analysis: str) -> str:
        return f"Processing results: {analysis}"

    # With validation
    from pydantic import BaseModel

    class ActionResult(BaseModel):
        action_taken: str
        success: bool
        details: str

    @act(
        "process",
        json_output=True,
        return_type=ActionResult
    )
    def process_data(self, analysis: str) -> ActionResult:
        return ActionResult(
            action_taken="data_cleanup",
            success=True,
            details="Processed and normalized data"
        )
    ```
"""

observe = create_step_decorator(StepType.OBSERVE)
"""Decorator for creating observation/data gathering steps.

Example:
    ```python
    # Standard usage
    @observe("gather", description="Gathers input data")
    def gather_data(self, query: str) -> str:
        return f"Gathering data for: {query}"

    # With validation
    from pydantic import BaseModel
    from typing import List

    class Observation(BaseModel):
        data_points: List[float]
        timestamp: str
        source: str

    @observe(
        "gather",
        json_output=True,
        return_type=Observation
    )
    def gather_data(self, query: str) -> Observation:
        return Observation(
            data_points=[1.2, 3.4, 5.6],
            timestamp="2024-01-01T12:00:00",
            source="sensor_array"
        )
    ```
"""

synthesize = create_step_decorator(StepType.SYNTHESIZE)
"""Decorator for creating synthesis/summary steps.

Example:
    ```python
    # Standard usage
    @synthesize("summarize", description="Summarizes results")
    def summarize_data(self, data: str) -> str:
        return f"Summary of: {data}"

    # With validation
    from pydantic import BaseModel
    from typing import List, Dict

    class Summary(BaseModel):
        key_findings: List[str]
        metrics: Dict[str, float]
        conclusion: str

    @synthesize(
        "summarize",
        json_output=True,
        return_type=Summary
    )
    def summarize_data(self, data: str) -> Summary:
        return Summary(
            key_findings=["Finding 1", "Finding 2"],
            metrics={"accuracy": 0.95, "confidence": 0.87},
            conclusion="Data shows positive trends"
        )
    ```
"""


@overload
def run(
    *, description: Optional[str] = None
) -> Callable[[Callable[..., T]], RunFunction]: ...


@overload
def run(func: Callable[..., T]) -> RunFunction: ...


def run(
    func: Optional[Callable[..., T]] = None,
    *,
    description: Optional[str] = None,
) -> Union[Callable[[Callable[..., T]], RunFunction], RunFunction]:
    """Decorator for defining a custom `run` method in an agent class.

    Marks a method as the custom run implementation for an agent,
    optionally with a description of its behavior.

    Args:
        func: The function to decorate (when used without parameters)
        description: Optional description of the custom run behavior

    Returns:
        Either a decorator function or the decorated function

    Example:
        Define custom run methods:
        ```python
        class CustomAgent(Agent):
            @run(description="Custom workflow execution")
            def custom_run(self, input_data: Any) -> Any:
                # Custom implementation
                return f"Custom execution for {input_data}"

            # Or without parameters
            @run
            def another_run(self, data: str) -> str:
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
