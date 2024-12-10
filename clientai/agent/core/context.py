from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentContext:
    """Maintains state, memory, and execution results for an agent.

    This class serves as a central repository for maintaining agent state
    across workflow executions. It stores step results, iteration counts,
    and arbitrary state data that may be needed during workflow execution.

    Attributes:
        memory: List of dictionaries storing step-by-step execution memory.
        state: Dictionary storing arbitrary state information for the agent.
        last_results: Dictionary mapping steps to their most recent results.
        current_input: The current input being processed by the workflow.
        iteration: Counter tracking the number of workflow iterations.

    Example:
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
        print(len(context.memory))  # Output: 0
        ```
    """

    memory: List[Dict[str, str]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    last_results: Dict[str, Any] = field(default_factory=dict)
    current_input: Any = None
    iteration: int = 0

    def clear(self) -> None:
        """Reset the context to its initial state.

        Example:
            Reset the context:
            ```python
            context = AgentContext()
            context.state["key"] = "value"
            context.clear()
            print(context.state)  # Output: {}
            ```
        """
        self.memory.clear()
        self.state.clear()
        self.last_results.clear()
        self.current_input = None
        self.iteration = 0

    def get_step_result(self, step_name: str) -> Any:
        """Retrieve the result of a specific workflow step.

        Args:
            step_name: Name of the step whose result should be retrieved.

        Returns:
            Any: The stored result for the specified step,
                 or None if no result exists.

        Example:
            Access a step result:
            ```python
            context = AgentContext()
            context.set_step_result("step1", "output1")
            print(context.get_step_result("step1"))  # Output: "output1"
            ```
        """
        return self.last_results.get(step_name)

    def set_step_result(self, step_name: str, result: Any) -> None:
        """Store the result of a workflow step.

        Args:
            step_name: Name of the step whose result is being stored.
            result: The result value to store.

        Example:
            Store a step result:
            ```python
            context = AgentContext()
            context.set_step_result("step1", "output1")
            print(context.get_step_result("step1"))  # Output: "output1"
            ```
        """
        self.last_results[step_name] = result

    def increment_iteration(self) -> int:
        """Increment and return the workflow iteration counter.

        Returns:
            int: The new iteration count after incrementing.

        Example:
            Increment the iteration counter:
            ```python
            context = AgentContext()
            print(context.increment_iteration())  # Output: 1
            print(context.increment_iteration())  # Output: 2
            ```
        """
        self.iteration += 1
        return self.iteration
