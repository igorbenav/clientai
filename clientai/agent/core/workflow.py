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

    This class registers steps, organizes them into a sequence,
    and executes them in the defined order. It supports two
    methods of passing data between steps:

    1. Automatic Parameter Binding:
       Steps can receive previous results through their parameters.
       Parameters are filled in reverse chronological order (most recent first)
       The number of parameters determines how many previous results are passed

       Example:
           >>> class MyAgent(Agent):
           ...     @think
           ...     def step1(self, input_text: str) -> str:
           ...         # Receives the initial input
           ...         return "First result"
           ...
           ...     @act
           ...     def step2(self, prev_result: str) -> str:
           ...         # Receives result from step1
           ...         return "Second result"
           ...
           ...     @synthesize
           ...     def step3(self, latest: str, previous: str) -> str:
           ...         # Receives results from step2 and step1 in that order
           ...         return "Final result"

    2. Context Access:
       Steps can access any previous result through the agent's context.
       This is useful for more complex workflows or when steps need to
       access results out of order:

       Example:
           >>> class MyAgent(Agent):
           ...     @think
           ...     def step1(self) -> str:
           ...         input_text = self.context.current_input
           ...         return "First result"
           ...
           ...     @act
           ...     def step2(self) -> str:
           ...         step1_result = self.context.get_step_result("step1")
           ...         return "Second result"

    Both methods can be used in the same agent, choosing the most appropriate
    approach for each step based on its needs.

    Parameter Validation:
    - Steps cannot declare more parameters than available results
    - Available results include the initial input and all previous step results
    - A ValueError is raised if a step requests more results than available

    Attributes:
        _steps: Steps registered for execution.
        _custom_run: Custom workflow execution function.
        _results_history: History of step execution results.

    Methods:
        register_class_steps: Register steps from an agent class.
        execute: Execute the workflow starting with the provided input.
        get_step: Retrieve a step by its name.
        get_steps: Retrieve all registered steps in execution order.
        reset: Reset the workflow manager.
        get_steps_by_type: Get all steps of a specific type.
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
        self,
        agent: Any,
        input_data: Any,
        engine: StepExecutionProtocol,
        stream_override: Optional[bool] = None,
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
            stream_override: Optional bool to override step's
                             stream configuration.

        Returns:
            The final result of the workflow execution.
        """
        logger.info(
            f"Starting workflow execution with {len(self._steps)} steps"
        )
        logger.debug(f"Input data: {input_data}")

        if self._custom_run:
            logger.info("Using custom run method")
            return self._custom_run(agent, input_data)

        agent.context.clear()
        agent.context.current_input = input_data
        last_result = input_data

        for step in self._steps.values():
            logger.info(f"Executing step: {step.name} ({step.step_type})")

            current_stream = (
                stream_override
                if stream_override is not None
                else getattr(step, "stream", False)
            )
            logger.debug(
                f"Using stream setting: {current_stream} for step {step.name}"
            )

            param_count = len(step.func.__code__.co_varnames) - 1
            available_results = len(agent.context.last_results) + 1

            if param_count > available_results:
                raise ValueError(
                    f"Step '{step.name}' declares {param_count} parameters, "
                    f"but only {available_results} previous results are "
                    f"available (including input data and results from steps: "
                    f"{', '.join(self._steps.keys())})"
                )

            try:
                if param_count == 0:
                    result = engine.execute_step(
                        step, agent, stream=current_stream
                    )
                elif param_count == 1:
                    result = engine.execute_step(
                        step, agent, last_result, stream=current_stream
                    )
                else:
                    previous_results = [
                        agent.context.last_results[s.name]
                        for s in reversed(list(self._steps.values()))
                        if s.name in agent.context.last_results
                    ][: param_count - 1]
                    args = [last_result] + previous_results
                    result = engine.execute_step(
                        step, agent, *args, stream=current_stream
                    )

                if result is not None:
                    agent.context.last_results[step.name] = result
                    if step.config.pass_result:
                        last_result = result

                logger.debug(f"Step {step.name} completed")

            except Exception as e:
                logger.error(f"Error executing step {step.name}: {e}")
                raise

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
