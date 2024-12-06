from collections import defaultdict
from typing import Dict, List, Optional, Set

from .base import Step
from .types import StepType


class StepRegistry:
    """
    A registry to manage and organize workflow steps.

    The registry stores steps by name, indexes them by type, and tracks
    dependencies between steps.

    Attributes:
        _steps: A dictionary of registered steps by their names.
        _type_index: An index mapping step types to step names.
        _dependency_graph: A graph tracking step dependencies.

    Methods:
        register: Add a step to the registry.
        get: Retrieve a step by its name.
        get_by_type: Retrieve all steps of a specific type.
        get_dependencies: Get names of steps that a step depends on.
        remove: Remove a step from the registry.
        clear: Clear all registered steps and indexes.
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
        """
        Register a step in the registry.

        Args:
            step (Step): The step to register.

        Raises:
            ValueError: If the step name is already registered.

        Examples:
            Register a step:
            ```python
            registry = StepRegistry()
            registry.register(step)
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
        """
        Retrieve a step by its name.

        Args:
            name (str): The name of the step to retrieve.

        Returns:
            Optional[Step]: The step if found, otherwise None.

        Examples:
            ```python
            step = registry.get("step_name")
            print(step)
            ```
        """
        return self._steps.get(name)

    def get_by_type(self, step_type: StepType) -> List[Step]:
        """
        Retrieve all steps of a specific type.

        Args:
            step_type (StepType): The type of steps to retrieve.

        Returns:
            List[Step]: A list of steps of the specified type.

        Examples:
            Get THINK steps:
            ```python
            think_steps = registry.get_by_type(StepType.THINK)
            print(think_steps)
            ```
        """
        step_names = self._type_index[step_type]
        return [self._steps[name] for name in step_names]

    def get_dependencies(self, step_name: str) -> Set[str]:
        """
        Retrieve the names of steps that the specified step depends on.

        Args:
            step_name (str): The name of the step.

        Returns:
            Set[str]: A set of step names this step depends on.

        Examples:
            ```python
            dependencies = registry.get_dependencies("step_name")
            print(dependencies)  # Output: {"step_a", "step_b"}
            ```
        """
        return self._dependency_graph.get(step_name, set())

    def remove(self, name: str) -> None:
        """
        Remove a step from the registry and update indexes.

        Removes the step and updates all related indexes and dependencies.
        If the step doesn't exist, silently returns.

        Args:
            name (str): The name of the step to remove.

        Examples:
            Remove a step:
            ```python
            registry.remove("step_name")
            print(registry.get("step_name"))  # Output: None
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
        """
        Clear all registered steps and indexes from the registry.

        Removes all steps and resets the registry to its initial empty state.

        Examples:
            Clear the registry:
            ```python
            registry.clear()
            print(registry.get("step_name"))  # Output: None
            ```
        """
        self._steps.clear()
        self._type_index.clear()
        self._dependency_graph.clear()
