from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class AgentContext:
    """Maintains state, memory, and execution results for an agent.

    This class serves as a central repository for maintaining agent
    state across workflow executions. It stores step results, tracks
    conversation history, maintains iteration counts, and stores
    arbitrary state data needed during workflow execution.

    Attributes:
        memory: List of dictionaries storing step-by-step execution memory.
        state: Dictionary storing arbitrary state information for the agent.
        last_results: Dictionary mapping steps to their most recent results.
        current_input: The current input being processed by the workflow.
        original_input: Original input stored separately from current_input.
        conversation_history: List of previous interactions with their results.
        max_history_size: Maximum number of previous interactions to maintain.
        iteration: Counter tracking the number of workflow iterations.

    Example:
        Initialize and manipulate the context:
        ```python
        context = AgentContext()

        # Set new input
        context.set_input("What is Python?")

        # Store a step result
        context.set_step_result("analyze", "Python is a programming language")

        # Access conversation history
        history = context.get_recent_history(n=2)
        print(history)  # Shows last 2 interactions

        # Reset the context but keep history
        context.clear()
        ```
    """

    memory: List[Dict[str, str]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    last_results: Dict[str, Any] = field(default_factory=dict)
    current_input: Any = None
    original_input: Any = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history_size: int = 10
    iteration: int = 0

    def set_input(self, input_data: Any) -> None:
        """Set new input and save previous interaction to history.

        Stores the current interaction in history (if exists) and sets up
        for a new interaction. Maintains maximum history size by removing
        oldest interactions when limit is reached.

        Args:
            input_data: The new input to process.

        Example:
            ```python
            context = AgentContext()
            context.set_input("What is Python?")
            context.set_step_result(
                "analyze",
                "Python is a programming language"
            )
            context.set_input(
                "How do I install Python?"
            )  # Previous interaction saved
            ```
        """
        if self.current_input is not None and self.last_results:
            interaction = {
                "input": self.original_input,
                "results": self.last_results.copy(),
                "iteration": self.iteration,
                "timestamp": datetime.now().isoformat(),
            }
            self.conversation_history.append(interaction)
            if len(self.conversation_history) > self.max_history_size:
                self.conversation_history = self.conversation_history[
                    -self.max_history_size :
                ]

        self.current_input = input_data
        self.original_input = input_data
        self.last_results.clear()

    def clear(self) -> None:
        """Reset the current interaction but preserve conversation history.

        Clears current state, memory, and results while maintaining the
        conversation history.

        Example:
            ```python
            context = AgentContext()
            context.state["key"] = "value"
            context.clear()
            print(context.state)  # Output: {}
            print(len(context.conversation_history))  # Preserved
            ```
        """
        self.memory.clear()
        self.state.clear()
        self.last_results.clear()
        self.current_input = None
        self.original_input = None
        self.iteration = 0

    def clear_all(self) -> None:
        """Reset everything including conversation history.

        Performs a complete reset of the context, including all history.

        Example:
            ```python
            context = AgentContext()
            context.set_input("Test")
            context.clear_all()
            print(len(context.conversation_history))  # Output: 0
            ```
        """
        self.clear()
        self.conversation_history.clear()

    def get_step_result(self, step_name: str) -> Any:
        """Retrieve the result of a specific workflow step.

        Args:
            step_name: Name of the step whose result should be retrieved.

        Returns:
            Any: The stored result for the specified step,
                 or None if no result exists.

        Example:
            ```python
            context = AgentContext()
            context.set_step_result("analyze", "Result")
            print(context.get_step_result("analyze"))  # Output: "Result"
            ```
        """
        return self.last_results.get(step_name)

    def set_step_result(self, step_name: str, result: Any) -> None:
        """Store the result of a workflow step.

        Args:
            step_name: Name of the step whose result is being stored.
            result: The result value to store.

        Example:
            ```python
            context = AgentContext()
            context.set_step_result("analyze", "Python analysis")
            print(context.get_step_result("analyze"))
            ```
        """
        self.last_results[step_name] = result

    def set_max_history_size(self, size: int) -> None:
        """Update the maximum history size and trim if necessary.

        Args:
            size: New maximum number of interactions to maintain.

        Raises:
            ValueError: If size is negative.

        Example:
            ```python
            context = AgentContext()
            context.set_max_history_size(5)  # Only keep last 5 interactions
            ```
        """
        if size < 0:
            raise ValueError("History size must be non-negative")

        self.max_history_size = size
        if len(self.conversation_history) > size:
            self.conversation_history = self.conversation_history[-size:]

    def get_recent_history(
        self, n: Optional[int] = None, raw: bool = False
    ) -> Union[str, List[Dict[str, Any]]]:
        """Get recent interactions with formatted context for LLM.

        Retrieves recent interactions either as formatted text for LLM context
        or as raw data structures for programmatic use.

        Args:
            n: Number of recent interactions to retrieve.
                Defaults to all within max_size.
            raw: If True, returns raw data structure.
                 If False, returns formatted string.

        Returns:
            Either a formatted string of conversation history suitable for LLM
            context, or the raw list of interaction dictionaries.

        Example:
            ```python
            context = AgentContext()

            # Get formatted history for LLM
            history = context.get_recent_history(n=2)

            # Get raw data for processing
            raw_history = context.get_recent_history(n=2, raw=True)
            ```
        """
        history = (
            self.conversation_history[-n:]
            if n and n < len(self.conversation_history)
            else self.conversation_history
        )

        if raw:
            return history

        if not history:
            if self.original_input is not None:
                return "No previous interactions. This is the first query."
            return "No interactions available."

        formatted_history = []
        for i, interaction in enumerate(history, 1):
            timestamp = datetime.fromisoformat(
                interaction["timestamp"]
            ).strftime("%Y-%m-%d %H:%M:%S")
            formatted_interaction = (
                f"Interaction {i}:\n"
                f"Time: {timestamp}\n"
                f"Input: {interaction['input']}\n"
                f"Results:"
            )

            for step_name, result in interaction["results"].items():
                formatted_interaction += f"\n- {step_name}: {result}"

            formatted_history.append(formatted_interaction)

        history_text = "Previous Interactions:\n" + "\n\n".join(
            formatted_history
        )

        if self.original_input is not None:
            history_text += "\n\nNow handling the current query."

        return history_text

    def increment_iteration(self) -> int:
        """Increment and return the workflow iteration counter.

        Returns:
            int: The new iteration count after incrementing.

        Example:
            ```python
            context = AgentContext()
            print(context.increment_iteration())  # Output: 1
            print(context.increment_iteration())  # Output: 2
            ```
        """
        self.iteration += 1
        return self.iteration
