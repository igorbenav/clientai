from enum import Enum, auto
from typing import Any, Protocol


class StepType(Enum):
    """Type classification for workflow steps.

    Defines the different types of steps that can exist
    in a workflow, each representing a different kind
    of operation or phase in the agent's processing.

    Attributes:
        THINK: Analysis and reasoning steps that process information
        ACT: Decision-making and action steps that perform operations
        OBSERVE: Data collection and observation steps that gather information
        SYNTHESIZE: Integration steps that combine or summarize information

    Example:
        Using step types:
        ```python
        # Reference step types
        step_type = StepType.THINK
        print(step_type.name)  # Output: "THINK"

        # Use in step decoration
        @think("analyze")  # Uses StepType.THINK internally
        def analyze_data(self, input_data: str) -> str:
            return f"Analysis of {input_data}"

        # Compare step types
        if step.step_type == StepType.ACT:
            print("This is an action step")
        ```

    Notes:
        - Each step type has default configurations (temperature, etc.)
        - Step types influence tool availability through scoping
        - Custom steps typically default to ACT type behavior
    """

    THINK = auto()
    ACT = auto()
    OBSERVE = auto()
    SYNTHESIZE = auto()


class StepFunction(Protocol):
    """Protocol defining the interface for step functions.

    Defines the required signature for callable objects that can serve as
    workflow steps, ensuring type safety and consistent behavior.

    Methods:
        __call__: Execute the step with provided input data

    Example:
        Implementing the protocol:
        ```python
        class MyStep:
            def __call__(self, input_data: Any) -> str:
                return f"Processed: {input_data}"

        # Function conforming to protocol
        def example_step(input_data: Any) -> str:
            return f"Handled: {input_data}"

        # Using with type checking
        def register_step(step: StepFunction) -> None:
            result = step("test input")
            print(result)
        ```

    Notes:
        - Step functions must return strings
        - Input can be any type but should be documented
        - Protocol enables static type checking
    """

    def __call__(self, input_data: Any) -> str:
        """Execute the step's processing logic.

        Args:
            input_data: Data to be processed by the step

        Returns:
            str: Result of step processing
        """
        ...
