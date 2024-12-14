from collections import defaultdict
from typing import Dict, List, Optional, Set

from .base import Step
from .types import StepType


class StepRegistry:
    """Registry for managing and organizing workflow steps.

    Maintains a registry of workflow steps with support for dependency
    tracking, type indexing, and step validation. Ensures steps are
    properly organized and can be executed in the correct order.

    Attributes:
        _steps: Dictionary mapping step names to their Step instances
        _type_index: Index mapping step types to sets of step names
        _dependency_graph: Graph tracking dependencies between steps

    Example:
        Basic registry usage:
        ```python
        registry = StepRegistry()

        # Register a step
        registry.register(analyze_step)

        # Get steps by type
        think_steps = registry.get_by_type(StepType.THINK)

        # Get step dependencies
        deps = registry.get_dependencies("process_step")
        print(deps)  # Output: {"analyze_step"}
        ```

    Notes:
        - Steps are stored with their dependencies for workflow ordering
        - Type indexing enables efficient step retrieval by type
        - Steps must have unique names within the registry
    """

    def __init__(self) -> None:
        """
        Initialize an empty step registry.

        Creates empty storage for tools and initializes scope indexing for
        all available tool scopes.
        """
        self._steps: Dict[str, Step] = {}
        self._type_index: Dict[StepType, Set[str]] = defaultdict(set)
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)

    def register(self, step: Step) -> None:
        """Register a step in the registry.

        Args:
            step: The step instance to register

        Raises:
            ValueError: If a step with the same name is already registered

        Example:
            Register a step:
            ```python
            registry = StepRegistry()
            try:
                registry.register(analyze_step)
                print("Step registered successfully")
            except ValueError as e:
                print(f"Registration failed: {e}")
            ```
        """
        if step.name in self._steps:
            raise ValueError(f"Step '{step.name}' already registered")

        self._steps[step.name] = step
        self._type_index[step.step_type].add(step.name)
        self._update_dependencies(step)

    def _update_dependencies(self, step: Step) -> None:
        """
        Update step dependencies based on compatibility.

        Analyzes the new step's compatibility with existing steps and updates
        the dependency graph accordingly.

        Args:
            step: The step whose dependencies need to be updated.
        """
        for existing_step in self._steps.values():
            if step.is_compatible_with(existing_step):
                self._dependency_graph[step.name].add(existing_step.name)

    def get(self, name: str) -> Optional[Step]:
        """Retrieve a step by its name.

        Args:
            name: The name of the step to retrieve

        Returns:
            Optional[Step]: The requested step if found, None otherwise

        Example:
            Retrieve a step:
            ```python
            step = registry.get("analyze_step")
            if step:
                print(f"Found step: {step.name}")
            else:
                print("Step not found")
            ```
        """
        return self._steps.get(name)

    def get_by_type(self, step_type: StepType) -> List[Step]:
        """Retrieve all steps of a specific type.

        Args:
            step_type: The type of steps to retrieve

        Returns:
            List[Step]: List of steps matching the specified type

        Example:
            Get steps by type:
            ```python
            think_steps = registry.get_by_type(StepType.THINK)
            print(f"Found {len(think_steps)} thinking steps:")
            for step in think_steps:
                print(f"- {step.name}")
            ```
        """
        step_names = self._type_index[step_type]
        return [self._steps[name] for name in step_names]

    def get_dependencies(self, step_name: str) -> Set[str]:
        """Get names of steps that a step depends on.

        Args:
            step_name: The name of the step

        Returns:
            Set[str]: Set of step names this step depends on

        Example:
            Check dependencies:
            ```python
            deps = registry.get_dependencies("final_step")
            print(f"Dependencies: {', '.join(deps)}")  # Output: "step1, step2"
            ```
        """
        return self._dependency_graph.get(step_name, set())

    def remove(self, name: str) -> None:
        """Remove a step from the registry.

        Args:
            name: The name of the step to remove

        Example:
            Remove a step:
            ```python
            registry.remove("old_step")
            print(registry.get("old_step"))  # Output: None
            ```
        """
        if name not in self._steps:
            return

        step = self._steps[name]
        del self._steps[name]
        self._type_index[step.step_type].discard(name)
        del self._dependency_graph[name]
        for deps in self._dependency_graph.values():
            deps.discard(name)

    def clear(self) -> None:
        """Clear all registered steps and indexes.

        Example:
            Clear registry:
            ```python
            registry.clear()
            print(len(registry.get_by_type(StepType.THINK)))  # Output: 0
            ```
        """
        self._steps.clear()
        self._type_index.clear()
        self._dependency_graph.clear()
