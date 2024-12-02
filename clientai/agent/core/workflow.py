import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

from ..steps.base import Step
from ..steps.types import StepType
from ..types.protocols import StepExecutionProtocol

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manages the workflow of steps for an agent.

    This class registers steps, organizes them into a sequence, and
    executes them in the defined order.

    Attributes:
        _steps: Steps registered for execution.
        _custom_run: Custom workflow execution function.

    Methods:
        register_class_steps: Register steps from an agent class.
        execute: Execute the workflow starting with the provided input.
        get_step: Retrieve a step by its name.
        get_steps: Retrieve all registered steps in execution order.
        reset: Reset the workflow manager.

    Example:
        >>> manager = WorkflowManager()
        >>> manager.register_class_steps(agent)
        >>> result = manager.execute(agent, input_data, execution_engine)
    """

    def __init__(self) -> None:
        """
        Initialize the WorkflowManager.

        Sets up empty step registry and custom run function.

        Example:
            >>> manager = WorkflowManager()
        """
        self._steps: OrderedDict[str, Step] = OrderedDict()
        self._custom_run: Optional[Callable[[Any, Any], Any]] = None

    def register_class_steps(self, agent_instance: Any) -> None:
        """
        Register steps defined in an agent class.

        Scans the agent class for methods decorated as steps and registers them
        for execution. Also identifies any custom run method if present.

        Args:
            agent_instance: The agent containing the step definitions.

        Example:
            >>> manager.register_class_steps(agent)
        """
        steps: list[tuple[str, Step]] = []
        class_dict = agent_instance.__class__.__dict__

        for name, func in class_dict.items():
            if callable(func):
                if hasattr(func, "_step_info"):
                    logger.debug(f"Registering step: {name}")
                    steps.append((func._step_info.name, func._step_info))
                elif hasattr(func, "_is_run"):
                    logger.debug(f"Found custom run method: {name}")
                    self._custom_run = getattr(agent_instance, name)

        self._steps = OrderedDict(steps)
        logger.info(
            f"Registered {len(steps)} steps: {list(self._steps.keys())}"
        )

    def execute(
        self, agent: Any, input_data: Any, engine: StepExecutionProtocol
    ) -> Any:
        """
        Execute the workflow with the provided input data.

        Processes each step in sequence, passing results between steps
        as configured. Uses either the default sequential execution or
        a custom run method if defined.

        Args:
            agent: The agent executing the workflow.
            input_data: The initial input data.
            engine: The execution engine for processing steps.

        Returns:
            The final result of the workflow execution.

        Example:
            >>> result = manager.execute(agent, "input data", engine)
            >>> print(result)
        """
        logger.info(
            f"Starting workflow execution with {len(self._steps)} steps"
        )
        logger.debug(f"Input data: {input_data}")

        if self._custom_run:
            logger.info("Using custom run method")
            return self._custom_run(agent, input_data)

        last_result: Any = input_data
        for step in self._steps.values():
            logger.info(f"Executing step: {step.name} ({step.step_type})")
            logger.debug(f"Step input: {last_result}")

            result = engine.execute_step(step, agent, last_result)
            logger.debug(f"Step result: {result}")

            if step.config.pass_result:
                logger.debug(f"Passing result from {step.name} to next step")
                last_result = result

            logger.debug(f"Context after step: {agent.context.last_results}")

        logger.info("Workflow execution completed")
        return last_result

    def get_step(self, name: str) -> Optional[Step]:
        """
        Retrieve a registered step by its name.

        Args:
            name: The name of the step to retrieve.

        Returns:
            The requested step if found, None otherwise.

        Example:
            >>> step = manager.get_step("analyze")
            >>> if step:
            ...     print(f"Found step: {step.name}")
        """
        return self._steps.get(name)

    def get_steps(self) -> OrderedDict[str, Step]:
        """
        Retrieve all registered steps in execution order.

        Returns:
            An ordered dictionary of step names mapped to their instances.

        Example:
            >>> steps = manager.get_steps()
            >>> for name, step in steps.items():
            ...     print(f"{name}: {step.step_type}")
        """
        return self._steps.copy()

    def reset(self) -> None:
        """
        Reset the workflow manager to its initial state.

        Clears all registered steps and custom run method.

        Example:
            >>> manager.reset()
            >>> print(len(manager.get_steps()))
            0
        """
        self._steps.clear()
        self._custom_run = None

    def get_steps_by_type(self, step_type: StepType) -> Dict[str, Step]:
        """
        Retrieve all steps of a specific type.

        Args:
            step_type: The type of steps to retrieve.

        Returns:
            A dictionary of step names mapped to
            their instances for the given type.

        Example:
            >>> think_steps = manager.get_steps_by_type(StepType.THINK)
            >>> print(f"Found {len(think_steps)} thinking steps")
        """
        return {
            name: step
            for name, step in self._steps.items()
            if step.step_type == step_type
        }
