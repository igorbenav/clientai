import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..exceptions import StepError, WorkflowError
from ..steps.base import Step
from ..steps.types import StepType
from ..types.protocols import StepExecutionProtocol

logger = logging.getLogger(__name__)


class WorkflowManager:
    """Manages the workflow of steps for an agent.

    This class registers steps, organizes them into a sequence,
    and executes them in the defined order. It supports two methods
    of passing data between steps:

    1. Automatic Parameter Binding

        Steps can receive previous results through their parameters.
        Parameters are filled in reverse chronological order
        (most recent first). The number of parameters determines how
        many previous results are passed.

    Example:
        ```python
        class MyAgent(Agent):
            @think
            def step1(self, input_text: str) -> str:
                # Receives the initial input
                return "First result"

            @act
            def step2(self, prev_result: str) -> str:
                # Receives result from step1
                return "Second result"

            @synthesize
            def step3(self, latest: str, previous: str) -> str:
                # Receives results from step2 and step1 in that order
                return "Final result"
        ```

    2. Context Access

        Steps can access any previous result through the agent's context.
        This is useful for more complex workflows or when steps need to
        access results out of order.

    Example:
        ```python
        class MyAgent(Agent):
            @think
            def step1(self) -> str:
                input_text = self.context.current_input
                return "First result"

            @act
            def step2(self) -> str:
                step1_result = self.context.get_step_result("step1")
                return "Second result"
        ```

    Both methods can be used in the same agent, choosing the most appropriate
    approach for each step based on its needs.

    Parameter Validation:
        - Steps cannot declare more parameters than available results
        - Available results include the initial input and all previous results
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
            ```python
            manager = WorkflowManager()
            ```
        """
        try:
            self._steps: OrderedDict[str, Step] = OrderedDict()
            self._custom_run: Optional[Callable[[Any, Any], Any]] = None
            logger.debug("Initialized WorkflowManager")
        except Exception as e:
            logger.error(f"Failed to initialize WorkflowManager: {e}")
            raise WorkflowError(
                f"Failed to initialize WorkflowManager: {str(e)}"
            ) from e

    def _validate_agent_instance(self, agent_instance: Any) -> Dict[str, Any]:
        """Validate and get the agent class dictionary."""
        try:
            return agent_instance.__class__.__dict__  # type: ignore
        except AttributeError as e:
            raise WorkflowError(f"Invalid agent instance: {str(e)}")

    def _process_step_info(
        self, name: str, func: Any
    ) -> Optional[Tuple[str, Step]]:
        """Process a single step or run method from the class."""
        try:
            if not callable(func):
                return None

            if hasattr(func, "_step_info"):
                logger.debug(f"Processing step: {name}")
                step_info = func._step_info
                if not isinstance(step_info, Step):
                    raise StepError(
                        f"Invalid step info for {name}: "
                        f"expected Step, got {type(step_info)}"
                    )
                return (step_info.name, step_info)
            return None
        except Exception as e:
            logger.error(f"Error processing step '{name}': {e}")
            raise StepError(f"Error processing step '{name}': {str(e)}") from e

    def _process_run_method(
        self, name: str, func: Any, agent_instance: Any
    ) -> None:
        """Process a custom run method if found."""
        try:
            if callable(func) and hasattr(func, "_is_run"):
                logger.debug(f"Found custom run method: {name}")
                self._custom_run = getattr(agent_instance, name)
        except AttributeError as e:
            raise StepError(f"Failed to bind custom run method: {str(e)}")

    def register_class_steps(self, agent_instance: Any) -> None:
        """
        Register steps defined in an agent class.

        Scans the agent class for methods decorated as steps and registers them
        for execution. Also identifies any custom run method if present.

        Args:
            agent_instance: The agent containing the step definitions.

        Example:
            ```python
            manager.register_class_steps(agent)
            ```
        """
        try:
            class_dict = self._validate_agent_instance(agent_instance)
            steps: List[Tuple[str, Step]] = []

            for name, func in class_dict.items():
                step_info = self._process_step_info(name, func)
                if step_info:
                    steps.append(step_info)
                self._process_run_method(name, func, agent_instance)

            self._steps = OrderedDict(steps)
            logger.info(
                f"Registered {len(steps)} steps: {list(self._steps.keys())}"
            )

        except (StepError, WorkflowError):
            raise
        except Exception as e:
            logger.error(f"Failed to register class steps: {e}")
            raise WorkflowError(
                f"Failed to register class steps: {str(e)}"
            ) from e

    def _execute_custom_run(self, agent: Any, input_data: Any) -> Any:
        """Execute the custom run method if defined."""
        try:
            logger.info("Using custom run method")
            if self._custom_run is None:
                raise WorkflowError("Custom run method is None")
            return self._custom_run(agent, input_data)
        except Exception as e:
            raise WorkflowError(f"Custom run method failed: {str(e)}")

    def _initialize_execution(self, agent: Any, input_data: Any) -> None:
        """Initialize the execution context."""
        try:
            if not self._steps and not self._custom_run:
                raise WorkflowError(
                    "No steps registered and no custom run method defined"
                )

            agent.context.clear()
            agent.context.current_input = input_data
        except Exception as e:
            raise WorkflowError(f"Failed to initialize execution: {str(e)}")

    def _get_step_parameters(self, step: Step) -> int:
        """Get the number of parameters for a step."""
        try:
            params = [
                param for param in step.func.__code__.co_varnames[:step.func.__code__.co_argcount]
                if param != 'self'
            ]
            return len(params)
        except AttributeError as e:
            raise StepError(f"Invalid step function: {str(e)}")

    def _validate_parameter_count(
        self, step: Step, param_count: int, available_results: int
    ) -> None:
        """Validate the parameter count against available results."""
        if param_count > available_results:
            raise ValueError(
                f"Step '{step.name}' declares {param_count} parameters, "
                f"but only {available_results} previous results are "
                f"available (including input data and results from steps: "
                f"{', '.join(self._steps.keys())})"
            )

    def _get_previous_results(self, agent: Any, param_count: int) -> List[Any]:
        """Gather results from previous steps."""
        try:
            return [
                agent.context.last_results[s.name]
                for s in reversed(list(self._steps.values()))
                if s.name in agent.context.last_results
            ][: param_count - 1]
        except Exception as e:
            raise StepError(f"Failed to gather previous results: {str(e)}")

    def _execute_step(
        self,
        step: Step,
        agent: Any,
        last_result: Any,
        param_count: int,
        current_stream: bool,
        engine: StepExecutionProtocol,
    ) -> Any:
        """Execute a single step with proper parameter handling."""
        try:
            if param_count == 0:
                return engine.execute_step(step, agent, stream=current_stream)
            elif param_count == 1:
                return engine.execute_step(
                    step, agent, last_result, stream=current_stream
                )
            else:
                previous_results = self._get_previous_results(
                    agent, param_count
                )
                args = [last_result] + previous_results
                return engine.execute_step(
                    step, agent, *args, stream=current_stream
                )
        except Exception as e:
            raise StepError(
                f"Failed to execute step '{step.name}': {str(e)}"
            ) from e

    def _handle_step_result(
        self, step: Step, result: Any, agent: Any
    ) -> Optional[Any]:
        """Handle the result of a step execution."""
        if result is not None:
            agent.context.last_results[step.name] = result
            if step.config.pass_result:
                return result
        return None

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
        try:
            logger.info(
                f"Starting workflow execution with {len(self._steps)} steps"
            )
            logger.debug(f"Input data: {input_data}")

            self._initialize_execution(agent, input_data)

            if self._custom_run:
                return self._execute_custom_run(agent, input_data)

            last_result = input_data

            for step in self._steps.values():
                try:
                    logger.info(
                        f"Executing step: {step.name} ({step.step_type})"
                    )

                    current_stream = (
                        stream_override
                        if stream_override is not None
                        else getattr(step, "stream", False)
                    )
                    logger.debug(
                        f"Using stream setting: {current_stream} "
                        f"for step {step.name}"
                    )

                    param_count = self._get_step_parameters(step)
                    available_results = len(agent.context.last_results) + 1

                    self._validate_parameter_count(
                        step, param_count, available_results
                    )

                    result = self._execute_step(
                        step,
                        agent,
                        last_result,
                        param_count,
                        current_stream,
                        engine,
                    )

                    step_result = self._handle_step_result(step, result, agent)
                    if step_result is not None:
                        last_result = step_result

                    logger.debug(f"Step {step.name} completed")

                except (StepError, ValueError) as e:
                    logger.error(f"Error in step '{step.name}': {e}")
                    if step.config.required:
                        raise
                    logger.warning(
                        f"Continuing workflow after non-required "
                        f"step failure: {step.name}"
                    )
                    continue

            logger.info("Workflow execution completed")
            return last_result

        except (StepError, WorkflowError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Unexpected workflow execution error: {e}")
            raise WorkflowError(
                f"Unexpected workflow execution error: {str(e)}"
            ) from e

        except (StepError, WorkflowError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Unexpected workflow execution error: {e}")
            raise WorkflowError(
                f"Unexpected workflow execution error: {str(e)}"
            ) from e

    def get_step(self, name: str) -> Optional[Step]:
        """
        Retrieve a registered step by its name.

        Args:
            name: The name of the step to retrieve.

        Returns:
            The requested step if found, None otherwise.

        Example:
            ```python
            step = manager.get_step("analyze")
            if step:
                print(f"Found step: {step.name}")
            ```
        """
        try:
            return self._steps.get(name)
        except Exception as e:
            logger.error(f"Failed to retrieve step '{name}': {e}")
            raise WorkflowError(
                f"Failed to retrieve step '{name}': {str(e)}"
            ) from e

    def get_steps(self) -> OrderedDict[str, Step]:
        """
        Retrieve all registered steps in execution order.

        Returns:
            An ordered dictionary of step names mapped to their instances.

        Example:
            ```python
            steps = manager.get_steps()
            for name, step in steps.items():
                print(f"{name}: {step.step_type}")
            ```
        """
        try:
            return self._steps.copy()
        except Exception as e:
            logger.error("Failed to retrieve steps: {e}")
            raise WorkflowError(f"Failed to retrieve steps: {str(e)}") from e

    def reset(self) -> None:
        """
        Reset the workflow manager to its initial state.

        Clears all registered steps and custom run method.

        Example:
            ```python
            manager.reset()
            print(len(manager.get_steps()))

            # Output: 0
            ```
        """
        try:
            self._steps.clear()
            self._custom_run = None
            logger.debug("Workflow manager reset completed")
        except Exception as e:
            logger.error(f"Failed to reset workflow manager: {e}")
            raise WorkflowError(
                f"Failed to reset workflow manager: {str(e)}"
            ) from e

    def get_steps_by_type(self, step_type: StepType) -> Dict[str, Step]:
        """
        Retrieve all steps of a specific type.

        Args:
            step_type: The type of steps to retrieve.

        Returns:
            A dictionary of step names mapped to
            their instances for the given type.

        Example:
            ```python
            think_steps = manager.get_steps_by_type(StepType.THINK)
            print(f"Found {len(think_steps)} thinking steps")
            ```
        """
        try:
            if not isinstance(step_type, StepType):
                raise ValueError(
                    "Invalid step type: expected StepType, "
                    f"got {type(step_type)}"
                )

            return {
                name: step
                for name, step in self._steps.items()
                if step.step_type == step_type
            }
        except ValueError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to retrieve steps by type '{step_type}': {e}"
            )
            raise WorkflowError(
                f"Failed to retrieve steps by type '{step_type}': {str(e)}"
            ) from e
