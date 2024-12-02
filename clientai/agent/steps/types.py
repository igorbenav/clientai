from enum import Enum, auto
from typing import Any, Protocol


class StepType(Enum):
    """
    Enumeration of step types for workflow steps.

    Members:
        THINK: Represents a THINK step, typically involving
               analysis or reasoning.
        ACT: Represents an ACT step, involving execution
             or decision-making.
        OBSERVE: Represents an OBSERVE step, involving data
                 collection or observation.
        SYNTHESIZE: Represents a SYNTHESIZE step, involving
                    combining or summarizing information.

    Examples:
        ```python
        step_type = StepType.THINK
        print(step_type.name)  # Output: "THINK"
        ```
    """

    THINK = auto()
    ACT = auto()
    OBSERVE = auto()
    SYNTHESIZE = auto()


class StepFunction(Protocol):
    """
    Protocol defining the interface for step functions.

    A step function must accept input data and return a string.

    Methods:
        __call__: Execute the step logic with provided input data.

    Examples:
        Define a function matching the protocol:
        ```python
        def example_function(input_data: str) -> str:
            return f"Processed: {input_data}"
        ```
    """

    def __call__(self, input_data: Any) -> str:
        ...
