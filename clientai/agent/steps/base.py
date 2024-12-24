from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Type, get_type_hints

from ..config.models import ModelConfig
from ..config.steps import StepConfig
from ..tools import ToolCallDecision, ToolSelectionConfig
from .types import StepType


@dataclass(frozen=True)
class FunctionMetadata:
    """
    Metadata extracted from a function, including its name, return type,
    docstring, and argument types.

    Attributes:
        name: The name of the function.
        return_type: The return type of the function as a string.
        docstring: The docstring of the function.
        arg_types: A mapping of argument names to their types.

    Methods:
        from_function: Class method to extract metadata from a function.
    """

    name: str
    return_type: str
    docstring: Optional[str]
    arg_types: dict[str, Any]

    @classmethod
    def from_function(cls, func: Callable[..., Any]) -> "FunctionMetadata":
        """
        Extract metadata from a function, including its name, return type,
        docstring, and argument types.

        Args:
            func: The function to analyze.

        Returns:
            FunctionMetadata: An instance containing the extracted metadata.

        Example:
            Analyze a function:
            ```python
            def example_function(arg1: int, arg2: str) -> str:
                "An example function."
                return f"{arg1} and {arg2}"

            metadata = FunctionMetadata.from_function(example_function)
            print(metadata.name)  # Output: "example_function"
            print(metadata.return_type)  # Output: "str"
            print(metadata.arg_types)  # Output: {"arg1": int, "arg2": str}
            ```
        """
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        return_type = hints.pop("return", Any)
        return cls(
            name=func.__name__,
            return_type=str(return_type),
            docstring=func.__doc__,
            arg_types=hints,
        )


@dataclass(frozen=True)
class Step:
    """
    Represents a step in the agent's workflow, encapsulating its function,
    type, metadata, and configuration.

    Attributes:
        func: The function representing the step logic.
        step_type: The type of step (e.g., THINK, ACT).
        name: The name of the step.
        description: A description of the step's purpose.
        llm_config: LLM configuration if the step interacts with an LLM.
        send_to_llm: Whether the step sends its data to an LLM. Default True.
        stream: Whether to stream the LLM's response
        json_output: Whether the step's output should be validated as JSON.
            Cannot be used with stream=True.
        return_type: Optional type to validate output against when
            json_output=True. Must be a Pydantic model for validation.
        return_full_response: Whether to return the complete API response.
            Cannot be used with json_output=True.
        use_tools: Whether tool selection is enabled for step. Default True.
        tool_selection_config: Configuration for tool selection behavior.
        tool_model: Optional specific model to use for tool selection.
        config: Configuration settings for the step.
        metadata: Metadata extracted from the step function.
        tool_decisions: Records of tool selection and execution decisions.

    Example:
        Create a step with tool configuration:
        ```python
        step = Step.create(
            func=example_function,
            step_type=StepType.THINK,
            name="analyze",
            use_tools=True,
            tool_selection_config=ToolSelectionConfig(
                confidence_threshold=0.8
            ),
            tool_model=ModelConfig(
                name="llama-2",
                temperature=0.0
            )
        )
        ```

    Notes:
        - Streaming and validation (json_output, return_type)
          cannot be used together
        - Return type validation requires a Pydantic model type
    """

    func: Callable[..., Any]
    step_type: StepType
    name: str
    description: Optional[str] = None
    llm_config: Optional[ModelConfig] = None
    send_to_llm: bool = True
    stream: bool = False
    json_output: bool = False
    return_type: Optional[Type[Any]] = None
    return_full_response: bool = False
    use_tools: bool = True
    tool_selection_config: Optional[ToolSelectionConfig] = None
    tool_model: Optional[ModelConfig] = None
    metadata: Optional[FunctionMetadata] = None
    tool_decisions: Optional[List[ToolCallDecision]] = None
    config: StepConfig = field(default_factory=StepConfig)

    def __post_init__(self) -> None:
        """Validate step values after initialization."""
        self._validate_function(self.func)
        self._validate_name(self.name)

        if self.json_output and self.return_full_response:
            raise ValueError(
                f"Step '{self.name}' cannot use both JSON validation and "
                "full response return. These options are mutually exclusive."
            )

    @staticmethod
    def _validate_function(func: Callable[..., Any]) -> None:
        """Validate the step function's signature and return type.

        Ensures the function is callable. When json_output=True, also
        validates that return type hints are present and match the
        specified return_type.

        Args:
            func: The function to validate.

        Raises:
            ValueError: If the function is not callable
                        or has invalid type hints.

        Example:
            Validate a function:
            ```python
            def example_function(data: str) -> OutputFormat:
                return OutputFormat(value="test", score=0.5)

            validated_func = Step.validate_function(example_function)
            ```

        Notes:
            - Return type must be a Pydantic model for validation
        """
        if not callable(func):
            raise ValueError("func must be a callable")

    @staticmethod
    def _validate_name(name: str) -> None:
        """
        Validate the step's name to ensure it is non-empty
        and a valid Python identifier.

        Args:
            name: The name of the step to validate.

        Raises:
            ValueError: If the name is empty or not a valid identifier.

        Example:
            Validate a step name:
            ```python
            valid_name = Step.validate_name("valid_step_name")
            print(valid_name)  # Output: "valid_step_name"

            Step.validate_name("")  # Raises ValueError
            ```
        """
        if not name:
            raise ValueError("Step name cannot be empty")
        if not name.isidentifier():
            raise ValueError(
                f"Step name '{name}' must be a valid Python identifier"
            )

    def is_compatible_with(self, other: "Step") -> bool:
        """Check if this step's input is compatible with another step's output.

        Determines if steps can be connected in a workflow by comparing their
        input/output types.

        Args:
            other: The step to check compatibility with

        Returns:
            bool: True if this step can accept the other step's output type

        Example:
            Check step compatibility:
            ```python
            step1 = Step.create(
                func=process_text,  # returns str
                step_type=StepType.THINK,
                name="process"
            )
            step2 = Step.create(
                func=analyze_text,  # takes str input
                step_type=StepType.ACT,
                name="analyze"
            )

            if step2.is_compatible_with(step1):
                print("Can connect process -> analyze")
            ```
        """
        if not other.metadata or not self.metadata:
            return False

        arg_types = list(self.metadata.arg_types.values())
        if not arg_types:
            return False

        return str(arg_types[0]) == other.metadata.return_type

    def can_execute_with(self, input_data: Any) -> bool:
        """Check if the step can execute with the provided input.

        Args:
            input_data: The input data to validate

        Returns:
            bool: True if the input matches the step's expected input type

        Example:
            Validate input data:
            ```python
            step = Step.create(
                func=process_numbers,  # takes List[int]
                step_type=StepType.ACT,
                name="process"
            )

            data = [1, 2, 3]
            if step.can_execute_with(data):
                print("Input data is valid")
            ```
        """
        if not self.metadata or not self.metadata.arg_types:
            return True

        first_arg_type = next(iter(self.metadata.arg_types.values()))
        return isinstance(input_data, first_arg_type)

    def __str__(self) -> str:
        """
        Provide a human-readable string representation of the step.

        Returns:
            str: A description of the step, including its name, type,
                 and optional details.

        Example:
            Print a step's string representation:
            ```python
            step = Step.create(
                func=example_function,
                step_type=StepType.THINK,
                name="example_step"
            )
            print(str(step))
            # Output: "Step(example_step) | Type: THINK | Description: ..."
            ```
        """
        parts = [
            f"Step({self.name})",
            f"Type: {self.step_type.name}",
        ]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.llm_config:
            parts.append(f"Model: {self.llm_config.name}")
        if self.json_output:
            parts.append("JSON output enabled")
        return " | ".join(parts)

    @classmethod
    def create(
        cls,
        func: Callable[..., Any],
        step_type: StepType,
        name: Optional[str] = None,
        description: Optional[str] = None,
        llm_config: Optional[ModelConfig] = None,
        send_to_llm: Optional[bool] = None,
        stream: bool = False,
        json_output: bool = False,
        return_type: Optional[Type[Any]] = None,
        return_full_response: bool = False,
        use_tools: bool = True,
        tool_selection_config: Optional[ToolSelectionConfig] = None,
        tool_model: Optional[ModelConfig] = None,
        step_config: Optional[StepConfig] = None,
    ) -> "Step":
        """Create and validate a new step.

        Factory method for creating steps with
        comprehensive configuration options.

        Args:
            func: Function implementing the step logic
            step_type: Type classification for the step
            name: Optional custom name (defaults to function name)
            description: Optional step description
            llm_config: Optional LLM configuration
            send_to_llm: Whether to send step output to LLM
            stream: Whether to stream LLM responses
            json_output: Whether to validate the step's output as JSON.
                Cannot be used with stream=True.
            return_type: Type to validate output against when json_output=True.
                Must be a Pydantic model.
            return_full_response: Whether to return the complete API response.
                                  Cannot be used with json_output=True.
            json_output: Whether LLM should return JSON
            use_tools: Whether to enable tool selection
            tool_selection_config: Optional tool selection configuration
            tool_model: Optional specific model for tool selection
            step_config: Optional step-specific configuration

        Returns:
            Step: Validated step instance

        Raises:
            ValueError: If step name is invalid or function
                        lacks required annotations

        Example:
            Create step with tool selection:
            ```python
            step = Step.create(
                func=analyze_data,
                step_type=StepType.THINK,
                name="analyze",
                description="Analyzes input data",
                use_tools=True,
                tool_selection_config=ToolSelectionConfig(
                    confidence_threshold=0.8
                )
            )
            ```

            Create step with custom model:
            ```python
            step = Step.create(
                func=process_data,
                step_type=StepType.ACT,
                llm_config=ModelConfig(
                    name="gpt-4",
                    temperature=0.7
                ),
                stream=True
            )
            ```

            Create step with validation:
            ```python
            from pydantic import BaseModel

            class OutputFormat(BaseModel):
                value: str
                score: float

            step = Step.create(
                func=analyze_data,
                step_type=StepType.THINK,
                name="analyze",
                json_output=True,
                return_type=OutputFormat
            )
            ```
        """
        metadata = FunctionMetadata.from_function(func)
        return cls(
            func=func,
            step_type=step_type,
            name=name or func.__name__,
            description=description or func.__doc__,
            llm_config=llm_config,
            send_to_llm=send_to_llm if send_to_llm is not None else True,
            stream=stream,
            json_output=json_output,
            return_type=return_type,
            return_full_response=return_full_response,
            use_tools=use_tools,
            tool_selection_config=tool_selection_config,
            tool_model=tool_model,
            config=step_config or StepConfig(),
            metadata=metadata,
        )
