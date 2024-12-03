from dataclasses import dataclass
from typing import Any, Callable, Optional, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..config.models import ModelConfig
from ..config.steps import StepConfig
from .types import StepType


@dataclass(frozen=True)
class FunctionMetadata:
    """
    Metadata extracted from a function, including its name, return type,
    docstring, and argument types.

    Attributes:
        name (str): The name of the function.
        return_type (str): The return type of the function as a string.
        docstring (Optional[str]): The docstring of the function.
        arg_types (Dict[str, Any]): A mapping of argument names to their types.

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
            func (Callable[..., Any]): The function to analyze.

        Returns:
            FunctionMetadata: An instance containing the extracted metadata.

        Examples:
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
        hints = get_type_hints(func)
        return_type = hints.pop("return", Any)
        return cls(
            name=func.__name__,
            return_type=str(return_type),
            docstring=func.__doc__,
            arg_types=hints,
        )


class Step(BaseModel):
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
        json_output: Whether the LLM should return a JSON. Default False.
        config: Configuration settings for the step.
        metadata: Metadata extracted from the step function.

    Methods:
        validate_function: Ensures the step function is callable
                           and returns a string.
        validate_name: Ensures the step name is valid Python identifier.
        is_compatible_with: Checks if the step can receive
                            output from another step.
        can_execute_with: Determines if the step can execute with given input.
        __str__: Provides a human-readable string representation of the step.
        create: Factory method to create a validated Step instance.
    """

    func: Callable[..., Any]
    step_type: StepType
    name: str
    description: Optional[str] = None
    llm_config: Optional[ModelConfig] = None
    send_to_llm: bool = True
    json_output: bool = False
    config: StepConfig = Field(default_factory=StepConfig)
    metadata: Optional[FunctionMetadata] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )

    @field_validator("func")
    @classmethod
    def validate_function(cls, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Validate the step function's signature and return type.

        Ensures that the function is callable, has a return type annotation,
        and that the return type is a string.

        Args:
            func (Callable[..., Any]): The function to validate.

        Returns:
            Callable[..., Any]: The validated function.

        Raises:
            ValueError: If the function is not callable, lacks a return type
                        annotation, or does not return a string.

        Examples:
            Validate a function:
            ```python
            def example_function(input_data: str) -> str:
                return f"Processed: {input_data}"

            validated_func = Step.validate_function(example_function)
            ```
        """
        if not callable(func):
            raise ValueError("func must be a callable")

        hints = get_type_hints(func)
        if "return" not in hints:
            raise ValueError(
                f"Function {func.__name__} must have a return type annotation"
            )

        return_type = hints["return"]
        if return_type is not str:
            raise ValueError(
                f"Function {func.__name__} must return str (got {return_type})"
            )

        return func

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        """
        Validate the step's name to ensure it is non-empty
        and a valid Python identifier.

        Args:
            name (str): The name of the step to validate.

        Returns:
            str: The validated step name.

        Raises:
            ValueError: If the name is empty or not a valid identifier.

        Examples:
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
        return name

    def is_compatible_with(self, other: "Step") -> bool:
        """
        Check if this step's input is compatible with another step's output.

        Args:
            other (Step): The step to check compatibility with.

        Returns:
            bool: True if compatible, False otherwise.

        Examples:
            Check step compatibility:
            ```python
            step1 = Step.create(
                func=func1, step_type=StepType.THINK, name="step1"
            )
            step2 = Step.create(
                func=func2, step_type=StepType.ACT, name="step2"
            )
            print(step1.is_compatible_with(step2))  # Output: True or False
            ```
        """
        if not other.metadata or not self.metadata:
            return False

        arg_types = list(self.metadata.arg_types.values())
        if not arg_types:
            return False

        return str(arg_types[0]) == other.metadata.return_type

    def can_execute_with(self, input_data: Any) -> bool:
        """
        Determine if the step can execute with the provided input.

        Args:
            input_data (Any): The input data to test.

        Returns:
            bool: True if the step can execute with the input, False otherwise.

        Examples:
            Check if a step can execute with given input:
            ```python
            step = Step.create(
                func=example_function,
                step_type=StepType.ACT,
                name="example_step"
            )
            print(step.can_execute_with("test input"))  # Output: True or False
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

        Examples:
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
        json_output: bool = False,
        step_config: Optional[StepConfig] = None,
    ) -> "Step":
        """
        Factory method to create and validate a step.

        Args:
            func: The function representing the step logic.
            step_type: The type of step (THINK, ACT, etc.).
            name: A custom name for the step. Defaults to the function's name.
            description: A description of the step. Defaults to
                         the function's docstring.
            llm_config: Model configuration for LLM-based steps.
            send_to_llm: Whether the step sends its prompt to an LLM.
            json_output: Whether the LLM should format its response as JSON.
            step_config: Additional step-specific configuration.

        Returns:
            Step: A validated Step instance.

        Examples:
            Create a step:
            ```python
            step = Step.create(
                func=example_function,
                step_type=StepType.ACT,
                name="example_step",
                description="An example step.",
                llm_config=ModelConfig(name="gpt-4"),
                send_to_llm=True,
                json_output=True
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
            json_output=json_output,
            config=step_config or StepConfig(),
            metadata=metadata,
        )