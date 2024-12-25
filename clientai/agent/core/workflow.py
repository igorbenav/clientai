import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ..exceptions import StepError, WorkflowError
from ..steps.base import Step
from ..steps.types import StepType
from ..types.protocols import StepExecutionProtocol

logger = logging.getLogger(__name__)


class WorkflowManager:
    """Manages the workflow of steps for an agent.

    This class registers steps, organizes them into a sequence,
    and executes them in the defined order. It supports two
    methods of passing data between steps:

    1. Automatic Parameter Binding
        - Steps receive previous results through their parameters
        - Parameters filled in reverse chronological order (most recent first)
        - Number of parameters determines how many previous results are passed

    2. Context Access
        - Steps can access any previous result through the agent's context
        - Useful for complex workflows or accessing results out of order

    Attributes:
        _steps: Ordered dictionary of registered steps
        _custom_run: Optional custom workflow execution function
        _results_history: History of step execution results

    Example:
        Define an agent with different step parameter patterns:
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

        Using context access pattern:
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

    Notes:
        - Steps cannot declare more parameters than available results
        - Available results include the initial input and all previous results
        - Both parameter binding and context access can be used in an agent
        - A ValueError is raised if a step requests more results than available
        - Streaming and validation cannot be used together in the same step
        - Custom return types require explicit type hints and valid JSON output
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
            self._custom_run: Optional[Callable[[Any], Any]] = None
            logger.debug("Initialized WorkflowManager")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to initialize WorkflowManager: {e}")
            raise WorkflowError(
                f"Failed to initialize WorkflowManager: {str(e)}"
            ) from e

    def _validate_agent_instance(self, agent_instance: Any) -> Dict[str, Any]:
        """Validate and retrieve the agent class dictionary.

        Args:
            agent_instance: The agent instance to validate

        Returns:
            Dict[str, Any]: The validated agent class dictionary

        Raises:
            WorkflowError: If agent instance is invalid
        """
        try:
            return agent_instance.__class__.__dict__  # type: ignore
        except AttributeError as e:  # pragma: no cover
            raise WorkflowError(f"Invalid agent instance: {str(e)}")

    def _process_step_info(
        self, name: str, func: Any
    ) -> Optional[Tuple[str, Step]]:
        """Process a single step or run method from the class.

        Args:
            name: Name of the step/method
            func: Function to process

        Returns:
            Optional tuple of (step name, Step instance) if valid step

        Raises:
            StepError: If step info is invalid
        """
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
                    )  # pragma: no cover
                return (step_info.name, step_info)
            return None
        except Exception as e:  # pragma: no cover
            logger.error(f"Error processing step '{name}': {e}")
            raise StepError(f"Error processing step '{name}': {str(e)}") from e

    def _process_run_method(
        self, name: str, func: Any, agent_instance: Any
    ) -> None:
        """Process a custom run method if found.

        Args:
            name: Name of the method
            func: Function to process
            agent_instance: Agent instance owning the method

        Raises:
            StepError: If run method binding fails
        """
        try:
            if callable(func) and hasattr(func, "_is_run"):
                logger.debug(f"Found custom run method: {name}")
                self._custom_run = getattr(agent_instance, name)
        except AttributeError as e:  # pragma: no cover
            raise StepError(f"Failed to bind custom run method: {str(e)}")

    def register_class_steps(self, agent_instance: Any) -> None:
        """Register steps defined in an agent class.

        Scans the agent class for methods decorated as steps
        and registers them for execution. Also identifies
        any custom run method if present.

        Args:
            agent_instance: The agent instance containing step definitions

        Raises:
            StepError: If step registration fails
            WorkflowError: If class scanning fails

        Example:
            Register steps in an agent class:
            ```python
            class MyAgent(Agent):
                @think("analyze")
                def analyze_step(self, data: str) -> str:
                    return f"Analyzing: {data}"

                @act("process")
                def process_step(self, analysis: str) -> str:
                    return f"Processing: {analysis}"

            agent = MyAgent(...)
            workflow = WorkflowManager()
            workflow.register_class_steps(agent)
            ```

        Notes:
            - Steps must be decorated with appropriate step decorators
            - Steps are registered in the order they're defined
            - Custom run methods are detected and stored separately
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

        except (StepError, WorkflowError):  # pragma: no cover
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to register class steps: {e}")
            raise WorkflowError(
                f"Failed to register class steps: {str(e)}"
            ) from e

    def _execute_custom_run(
        self,
        agent: Any,
        input_data: Any,
        engine: StepExecutionProtocol,
        stream_override: Optional[bool] = None,
    ) -> Any:
        """Execute the custom run method with proper stream handling."""
        try:
            logger.info("Using custom run method")
            if self._custom_run is None:  # pragma: no cover
                raise WorkflowError("Custom run method is None")

            try:
                if stream_override is not None:
                    original_steps = {}
                    for step in self._steps.values():
                        original_steps[step.name] = getattr(
                            step, "stream", False
                        )
                        object.__setattr__(step, "stream", stream_override)

                result = self._custom_run(input_data)
                return result

            finally:
                if (
                    stream_override is not None
                    and "original_steps" in locals()
                ):
                    for step in self._steps.values():
                        object.__setattr__(
                            step, "stream", original_steps[step.name]
                        )

        except Exception as e:  # pragma: no cover
            logger.error(f"Custom run method failed: {e}")
            raise WorkflowError(f"Custom run method failed: {str(e)}") from e

    def _initialize_execution(self, agent: Any, input_data: Any) -> None:
        """Process a custom run method if found.

        Args:
            name: Name of the method
            func: Function to process
            agent_instance: Agent instance owning the method

        Raises:
            StepError: If run method binding fails
        """
        try:
            if not self._steps and not self._custom_run:  # pragma: no cover
                raise WorkflowError(
                    "No steps registered and no custom run method defined"
                )

            agent.context.clear()
            agent.context.current_input = input_data
        except Exception as e:  # pragma: no cover
            raise WorkflowError(f"Failed to initialize execution: {str(e)}")

    def _get_step_parameters(self, step: Step) -> int:
        """Get number of parameters for a step.

        Args:
            step: Step to analyze

        Returns:
            Number of non-self parameters

        Raises:
            StepError: If step function is invalid
        """
        try:
            params = [
                param
                for param in step.func.__code__.co_varnames[
                    : step.func.__code__.co_argcount
                ]
                if param != "self"
            ]
            return len(params)
        except AttributeError as e:  # pragma: no cover
            raise StepError(f"Invalid step function: {str(e)}")

    def _validate_parameter_count(
        self, step: Step, param_count: int, available_results: int
    ) -> None:
        """Validate parameter count against available results.

        Args:
            step: Step to validate
            param_count: Number of parameters required
            available_results: Number of results available

        Raises:
            ValueError: If step requires more parameters than available results
        """
        if param_count > available_results:
            raise ValueError(
                f"Step '{step.name}' declares {param_count} parameters, "
                f"but only {available_results} previous results are "
                f"available (including input data and results from steps: "
                f"{', '.join(self._steps.keys())})"
            )

    def _get_previous_results(self, agent: Any, param_count: int) -> List[Any]:
        """Gather results from previous steps.

        Args:
            agent: Agent instance
            param_count: Number of results needed

        Returns:
            List of previous results in reverse chronological order

        Raises:
            StepError: If result gathering fails
        """
        try:
            results = []

            step_results = [
                agent.context.last_results[step.name]
                for step in self._steps.values()
                if step.name in agent.context.last_results
            ][: param_count - 1]

            if len(step_results) < param_count:
                results = step_results + [agent.context.original_input]
            else:
                results = step_results

            return results

        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to gather previous results: {e}")
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
        """Execute a single step with proper parameter handling.

        Args:
            step: Step to execute
            agent: Agent instance
            last_result: Most recent result
            param_count: Number of parameters needed
            current_stream: Whether to stream output
            engine: Execution engine to use

        Returns:
            Any: Step execution result which could be:
                - str: For standard text responses
                - Iterator[str]: For streaming responses
                - Any: A validated type when json_output=True and
                       return type is specified
                Note: A step cannot use both streaming and validation.

        Raises:
            StepError: If step execution fails or validation fails
        """
        try:
            if param_count == 0:
                return engine.execute_step(step, stream=current_stream)
            elif param_count == 1:
                input_data = (
                    agent.context.original_input
                    if len(agent.context.last_results) == 0
                    else last_result
                )

                if isinstance(input_data, Iterator):
                    try:
                        collected = list(input_data)
                        if collected and isinstance(collected[0], str):
                            input_data = "".join(collected)
                        else:
                            input_data = collected[-1] if collected else None
                    except Exception as e:  # pragma: no cover
                        raise StepError(
                            f"Failed to collect streaming data: {str(e)}"
                        )

                return engine.execute_step(
                    step, input_data, stream=current_stream
                )
            else:
                previous_results = self._get_previous_results(
                    agent, param_count - 1
                )

                processed_results = []
                for result in [last_result] + previous_results:
                    if isinstance(result, Iterator):
                        try:
                            collected = list(result)
                            if collected and isinstance(collected[0], str):
                                result = "".join(collected)
                            else:
                                result = collected[-1] if collected else None
                        except Exception as e:
                            raise StepError(
                                f"Failed to collect streaming data: {str(e)}"
                            )
                    processed_results.append(result)

                return engine.execute_step(
                    step, *processed_results, stream=current_stream
                )
        except Exception as e:
            raise StepError(
                f"Failed to execute step '{step.name}': {str(e)}"
            ) from e

    def _handle_step_result(
        self, step: Step, result: Any, agent: Any
    ) -> Optional[Any]:
        """Handle the result of a step execution.

        Args:
            step: Executed step
            result: Step execution result
            agent: Agent instance

        Returns:
            Result to pass to next step if step.config.pass_result is True
        """
        if result is not None:
            agent.context.last_results[step.name] = result
            if step.config.pass_result:
                agent.context.current_input = result
                return result
        return None  # pragma: no cover

    def _get_step_stream_setting(
        self,
        step: Step,
        stream_override: Optional[bool],
        is_intermediate_step: bool,
    ) -> bool:
        """Determine the appropriate streaming setting for a step.

        Args:
            step: The step being executed
            stream_override: Optional streaming override from run()
            is_intermediate_step: Whether this is an intermediate step

        Returns:
            bool: The determined streaming setting
        """
        if stream_override is True and is_intermediate_step:
            return False

        if stream_override is not None:
            return stream_override

        return getattr(step, "stream", False)

    def execute(
        self,
        agent: Any,
        input_data: Any,
        engine: StepExecutionProtocol,
        stream_override: Optional[bool] = None,
    ) -> Any:
        """Execute the workflow with provided input data.

        Processes each step in sequence, passing results between steps
        as configured. Uses either default sequential execution or
        custom run method if defined.

        Args:
            agent: The agent executing the workflow
            input_data: The initial input data
            engine: The execution engine for processing steps
            stream_override: Optional bool to override steps'
                            stream configuration

        Returns:
            Any: The final result of workflow execution, which could be:
                - str: A standard text response
                - Iterator[str]: When streaming is enabled
                - Any: A validated type when json_output=True
                       and a return type is specified
                Note: Streaming and validation cannot be used together.

        Raises:
            StepError: If a required step fails
            WorkflowError: If workflow execution fails
            ValueError: If step parameter validation fails
            ValidationError: If output validation fails for a json_output step
        """
        try:
            logger.info(
                f"Starting workflow execution with {len(self._steps)} steps"
            )
            logger.debug(f"Input data: {input_data}")

            self._initialize_execution(agent, input_data)
            agent.context.set_input(input_data)
            agent.context.increment_iteration()

            engine._current_agent = agent  # type: ignore

            try:
                if self._custom_run:
                    return self._execute_custom_run(
                        agent=agent,
                        input_data=input_data,
                        engine=engine,
                        stream_override=stream_override,
                    )

                last_result = input_data
                steps = list(self._steps.values())

                for step in steps[:-1]:
                    try:
                        logger.info(
                            f"Executing step: {step.name} ({step.step_type})"
                        )

                        param_count = self._get_step_parameters(step)
                        available_results = len(agent.context.last_results) + 1
                        self._validate_parameter_count(
                            step, param_count, available_results
                        )

                        current_stream = self._get_step_stream_setting(
                            step=step,
                            stream_override=stream_override,
                            is_intermediate_step=True,
                        )

                        result = self._execute_step(
                            step=step,
                            agent=agent,
                            last_result=last_result,
                            param_count=param_count,
                            current_stream=current_stream,
                            engine=engine,
                        )

                        step_result = self._handle_step_result(
                            step, result, agent
                        )
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

                if steps:
                    final_step = steps[-1]

                    param_count = self._get_step_parameters(final_step)
                    available_results = len(agent.context.last_results) + 1
                    self._validate_parameter_count(
                        final_step, param_count, available_results
                    )

                    current_stream = self._get_step_stream_setting(
                        step=final_step,
                        stream_override=stream_override,
                        is_intermediate_step=False,
                    )

                    result = self._execute_step(
                        step=final_step,
                        agent=agent,
                        last_result=last_result,
                        param_count=param_count,
                        current_stream=current_stream,
                        engine=engine,
                    )

                    if not current_stream:
                        self._handle_step_result(final_step, result, agent)

                    return result

                logger.info("Workflow execution completed")
                return last_result

            finally:
                engine._current_agent = None  # type: ignore

        except (StepError, WorkflowError, ValueError):
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Unexpected workflow execution error: {e}")
            raise WorkflowError(
                f"Unexpected workflow execution error: {str(e)}"
            ) from e

    def get_step(self, name: str) -> Optional[Step]:
        """Retrieve a registered step by its name.

        Args:
            name: The name of the step to retrieve

        Returns:
            Optional[Step]: The requested step if found, None otherwise

        Example:
            Retrieve and check a step:
            ```python
            workflow = WorkflowManager()
            step = workflow.get_step("analyze")
            if step:
                print(f"Found step: {step.name}")
                print(f"Step type: {step.step_type}")
            ```
        """
        try:
            return self._steps.get(name)
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to retrieve step '{name}': {e}")
            raise WorkflowError(
                f"Failed to retrieve step '{name}': {str(e)}"
            ) from e

    def get_steps(self) -> OrderedDict[str, Step]:
        """Retrieve all registered steps in execution order.

        Returns:
            OrderedDict[str, Step]: Dictionary mapping step names
                                    to their instances

        Example:
            Get and inspect all steps:
            ```python
            workflow = WorkflowManager()
            steps = workflow.get_steps()

            for name, step in steps.items():
                print(f"Step: {name}")
                print(f"Type: {step.step_type}")
                print(f"Uses tools: {step.use_tools}")
            ```
        """
        try:
            return self._steps.copy()
        except Exception as e:  # pragma: no cover
            logger.error("Failed to retrieve steps: {e}")
            raise WorkflowError(f"Failed to retrieve steps: {str(e)}") from e

    def reset(self) -> None:
        """Reset the workflow manager to its initial state.

        Clears all registered steps and custom run method.

        Example:
            Reset workflow state:
            ```python
            workflow = WorkflowManager()
            # ... register steps and execute ...

            workflow.reset()
            print(len(workflow.get_steps()))  # Output: 0
            ```

        Notes:
            - Removes all registered steps
            - Clears custom run method if set
            - Does not affect step configurations
        """
        try:
            self._steps.clear()
            self._custom_run = None
            logger.debug("Workflow manager reset completed")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to reset workflow manager: {e}")
            raise WorkflowError(
                f"Failed to reset workflow manager: {str(e)}"
            ) from e

    def get_steps_by_type(self, step_type: StepType) -> Dict[str, Step]:
        """Retrieve all steps of a specific type.

        Args:
            step_type: The type of steps to retrieve

        Returns:
            Dict[str, Step]: Dictionary mapping step names to their instances
                for the given type

        Raises:
            ValueError: If step_type is invalid
            WorkflowError: If step retrieval fails

        Example:
            Get steps by type:
            ```python
            workflow = WorkflowManager()
            think_steps = workflow.get_steps_by_type(StepType.THINK)

            print(f"Found {len(think_steps)} thinking steps:")
            for name, step in think_steps.items():
                print(f"- {name}: {step.description}")
            ```

        Notes:
            - Returns empty dict if no steps of the type exist
            - Maintains original step order within type
            - Steps are returned by reference
        """
        try:
            if not isinstance(step_type, StepType):
                raise ValueError(
                    "Invalid step type: expected StepType, "
                    f"got {type(step_type)}"
                )  # pragma: no cover

            return {
                name: step
                for name, step in self._steps.items()
                if step.step_type == step_type
            }
        except ValueError:  # pragma: no cover
            raise
        except Exception as e:  # pragma: no cover
            logger.error(
                f"Failed to retrieve steps by type '{step_type}': {e}"
            )
            raise WorkflowError(
                f"Failed to retrieve steps by type '{step_type}': {str(e)}"
            ) from e
