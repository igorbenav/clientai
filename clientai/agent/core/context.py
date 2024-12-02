from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentContext:
    """
    Maintains the agent's state, memory, and results across steps.

    Attributes:
        memory: Stores step-by-step memory for contextual data.
        state: Stores arbitrary state information.
        last_results: Tracks the results of the most recent steps.
        current_input: The current input data for the workflow.
        iteration: Tracks the number of workflow iterations.

    Methods:
        clear: Reset the context to its initial state.
        get_step_result: Retrieve the result of a specific step.
        set_step_result: Store the result of a specific step.
        increment_iteration: Increment the workflow iteration counter.

    Examples:
        Initialize and manipulate the context:
        ```python
        context = AgentContext()

        # Set step results
        context.set_step_result("step1", "output1")
        print(context.get_step_result("step1"))  # Output: "output1"

        # Manage memory and state
        context.memory.append({"key": "value"})
        context.state["session"] = "active"

        # Reset the context
        context.clear()
        print(context.memory)  # Output: []
        ```
    """

    memory: List[Dict[str, str]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    last_results: Dict[str, Any] = field(default_factory=dict)
    current_input: Any = None
    iteration: int = 0

    def clear(self) -> None:
        """
        Reset the context by clearing memory, state, results, and input.

        Examples:
            Reset the context:
            ```python
            context = AgentContext()
            context.set_step_result("step1", "output1")
            context.clear()
            print(context.get_step_result("step1"))  # Output: None
            ```
        """
        self.memory.clear()
        self.state.clear()
        self.last_results.clear()
        self.current_input = None
        self.iteration = 0

    def get_step_result(self, step_name: str) -> Any:
        """
        Retrieve the result of a specific step.

        Args:
            step_name (str): The name of the step.

        Returns:
            Any: The result of the step, or None if the step has no result.

        Examples:
            Access a step result:
            ```python
            context = AgentContext()
            context.set_step_result("step1", "output1")
            print(context.get_step_result("step1"))  # Output: "output1"
            ```
        """
        return self.last_results.get(step_name)

    def set_step_result(self, step_name: str, result: Any) -> None:
        """
        Store the result of a step execution.

        Args:
            step_name (str): The name of the step.
            result (Any): The result to store.

        Examples:
            Store a step result:
            ```python
            context = AgentContext()
            context.set_step_result("step1", "output1")
            print(context.get_step_result("step1"))  # Output: "output1"
            ```
        """
        self.last_results[step_name] = result

    def increment_iteration(self) -> int:
        """
        Increment the current iteration counter.

        Returns:
            int: The updated iteration count.

        Examples:
            Increment the iteration counter:
            ```python
            context = AgentContext()
            print(context.increment_iteration())  # Output: 1
            print(context.increment_iteration())  # Output: 2
            ```
        """
        self.iteration += 1
        return self.iteration
