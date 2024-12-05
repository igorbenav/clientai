import logging
from typing import Any, Dict, Optional, Union

from clientai.client_ai import ClientAI
from clientai.exceptions import ClientAIError

from ..config.models import ModelConfig
from ..steps.base import Step
from ..tools import ToolSelectionConfig, ToolSelector

logger = logging.getLogger(__name__)


class StepExecutionEngine:
    """
    Handles the execution of workflow steps with integrated tool
    selection and LLM interaction.

    The StepExecutionEngine is responsible for executing individual
    workflow steps, managing tool selection, and coordinating LLM
    interactions. It handles:
    - Step execution with proper configuration
    - Automated tool selection and execution
    - LLM interaction with retry logic
    - Prompt building and result processing

    Key Features:
        - Configurable tool selection with confidence thresholds
        - Separate models for main workflow and tool selection
        - Built-in retry logic for LLM calls
        - Comprehensive error handling
        - Flexible prompt building

    Attributes:
        _client: The AI client for model interactions
        _default_model: Default model configuration for steps
        _default_kwargs: Default keyword arguments for model calls
        _default_tool_selection_config: Default tool selection settings
        _default_tool_model: Default model for tool selection
        _tool_selector: Instance handling tool selection logic

    Example:
        ```python
        engine = StepExecutionEngine(
            client=client,
            default_model=ModelConfig(name="gpt-4"),
            default_kwargs={"temperature": 0.7},
            tool_selection_config=ToolSelectionConfig(
                confidence_threshold=0.8,
                max_tools_per_step=3
            ),
            tool_model=ModelConfig(name="llama-2")
        )

        result = engine.execute_step(step, agent, "input data")
        ```
    """

    def __init__(
        self,
        client: ClientAI,
        default_model: Union[str, ModelConfig],
        default_kwargs: Dict[str, Any],
        tool_selection_config: Optional[ToolSelectionConfig] = None,
        tool_model: Optional[Union[str, ModelConfig]] = None,
    ) -> None:
        """
        Initialize the execution engine with specified configurations.

        Args:
            client: The AI client for model interactions
            default_model: Default model configuration
                           (string name or ModelConfig)
            default_kwargs: Default parameters for model calls
            tool_selection_config: Configuration for tool selection behavior
            tool_model: Model to use for tool selection
                        (default default_model)

        Example:
            ```python
            engine = StepExecutionEngine(
                client=my_client,
                default_model="gpt-4",
                default_kwargs={"temperature": 0.7},
                tool_selection_config=ToolSelectionConfig(
                    confidence_threshold=0.8
                )
            )
            ```
        """
        self._client = client
        self._default_model = default_model
        self._default_kwargs = default_kwargs
        self._default_tool_selection_config = (
            tool_selection_config or ToolSelectionConfig()
        )
        self._default_tool_model = tool_model or default_model
        self._tool_selector = ToolSelector()
        logger.debug(
            f"Initialized StepExecutionEngine with model: {default_model}"
        )

    def _get_tool_selection_config(self, step: Step) -> ToolSelectionConfig:
        """
        Get the effective tool selection configuration for a step.

        Determines the tool selection configuration to use by checking for
        step-specific settings and falling back to defaults if needed.

        Args:
            step: The step being executed

        Returns:
            The tool selection configuration to use for this step

        Example:
            ```python
            config = engine._get_tool_selection_config(step)
            print(config.confidence_threshold)  # Output: 0.8
            ```
        """
        step_config = getattr(step, "tool_selection_config", None)
        return step_config or self._default_tool_selection_config

    def _get_tool_model(self, step: Step) -> Union[str, ModelConfig]:
        """
        Get the effective model to use for tool selection in a step.

        Determines which model should be used for tool selection by checking
        step-specific settings and falling back to defaults if needed.

        Args:
            step: The step being executed

        Returns:
            The model configuration to use for tool selection

        Example:
            ```python
            model = engine._get_tool_model(step)
            print(model.name if isinstance(model, ModelConfig) else model)
            ```
        """
        step_model = getattr(step, "tool_model", None)
        return step_model or self._default_tool_model

    def _build_prompt(
        self, step: Step, agent: Any, *args: Any, **kwargs: Any
    ) -> str:
        """
        Build the prompt for a step, including tool selection if enabled.

        Creates a comprehensive prompt by executing the step's function and
        incorporating tool selection results if tools are enabled for the step.

        Args:
            step: The step being executed
            agent: The agent instance
            *args: Additional positional arguments for the step
            **kwargs: Additional keyword arguments for the step

        Returns:
            The complete prompt string to send to the LLM

        Raises:
            ValueError: If the step function returns an invalid type
            ClientAIError: If tool selection or execution fails

        Example:
            ```python
            prompt = engine._build_prompt(step, agent, "input data")
            print(prompt)  # Shows complete prompt with tool results if any
            ```
        """
        logger.debug(f"Building prompt for step '{step.name}'")
        logger.debug(f"Step use_tools setting: {step.use_tools}")

        try:
            result = step.func(agent, *args, **kwargs)
            if not isinstance(result, str):
                raise ValueError(
                    f"Step function must return str, got {type(result)}"
                )
            prompt = result

            if step.use_tools:
                logger.debug("Tool usage is enabled for this step")
                available_tools = agent.get_tools(step.step_type.name.lower())
                logger.debug(f"Found {len(available_tools)} available tools")

                if available_tools:
                    logger.debug("Starting tool selection process")
                    try:
                        decisions = self._tool_selector.select_tools(
                            task=prompt,
                            tools=available_tools,
                            context=agent.context.state,
                            client=self._client,
                        )

                        logger.debug(
                            f"Tool selector returned "
                            f"{len(decisions)} decisions"
                        )

                        if decisions:
                            logger.debug("Executing tool decisions")
                            tools_by_name = {
                                t.name: t for t in available_tools
                            }
                            updated_decisions = (
                                self._tool_selector.execute_tool_decisions(
                                    decisions=decisions, tools=tools_by_name
                                )
                            )

                            object.__setattr__(
                                step, "tool_decisions", updated_decisions
                            )

                            prompt += "\n\nTool Execution Results:\n"
                            for decision in updated_decisions:
                                prompt += f"\n{decision.tool_name}:"
                                if decision.error:
                                    prompt += f"\nError: {decision.error}"
                                else:
                                    prompt += f"\nResult: {decision.result}"
                                prompt += (
                                    f"\nConfidence: {decision.confidence}"
                                )
                                prompt += (
                                    f"\nReasoning: {decision.reasoning}\n"
                                )

                            agent.context.state["last_tool_decisions"] = [
                                {
                                    "tool_name": d.tool_name,
                                    "arguments": d.arguments,
                                    "result": d.result,
                                    "error": d.error,
                                    "confidence": d.confidence,
                                    "reasoning": d.reasoning,
                                }
                                for d in updated_decisions
                            ]
                    except Exception as e:
                        logger.error(
                            f"Error during tool selection/execution: {e}",
                            exc_info=True,
                        )
                        raise ClientAIError(
                            f"Tool selection/execution failed: {str(e)}"
                        ) from e
                else:
                    logger.debug("No tools available for this step")

            logger.debug(f"Final prompt: {prompt}")
            return prompt

        except Exception as e:
            logger.error(f"Error building prompt for step '{step.name}': {e}")
            raise

    def _execute_single_call(self, step: Step, prompt: str) -> str:
        """
        Execute a single LLM API call with proper configuration.

        Makes a single call to the LLM with the specified prompt and
        appropriate model configuration.

        Args:
            step: The step being executed
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response as a string

        Raises:
            ClientAIError: If the API call fails or returns
                           invalid response type

        Example:
            ```python
            try:
                response = engine._execute_single_call(
                    step,
                    "Analyze this text"
                )
                print(response)
            except ClientAIError as e:
                print(f"API call failed: {e}")
            ```
        """
        model_config = step.llm_config or self._default_model
        api_kwargs = {
            **self._default_kwargs,
            **(
                model_config.get_parameters()
                if isinstance(model_config, ModelConfig)
                else {}
            ),
        }

        result = self._client.generate_text(
            prompt,
            model=self._get_model_name(model_config),
            **api_kwargs,
        )

        if not isinstance(result, str):
            raise ClientAIError(
                f"Expected string response, got {type(result)}"
            )
        return result

    def _execute_with_retry(self, step: Step, prompt: str) -> str:
        """
        Execute an LLM call with configurable retry logic.

        Attempts the LLM call multiple times based on the step's retry
        configuration, logging warnings for failed attempts.

        Args:
            step: The step being executed
            prompt: The prompt to send to the LLM

        Returns:
            The successful LLM response

        Raises:
            ClientAIError: If all retry attempts fail

        Example:
            ```python
            try:
                response = engine._execute_with_retry(step, "Analyze this")
                print(response)
            except ClientAIError:
                print("All retry attempts failed")
            ```
        """
        for attempt in range(step.config.retry_count + 1):
            try:
                return self._execute_single_call(step, prompt)
            except ClientAIError as e:
                if attempt >= step.config.retry_count:
                    raise
                logger.warning(
                    f"Retry {attempt + 1}/{step.config.retry_count} "
                    f"for step '{step.name}': {e}"
                )
                continue
        raise ClientAIError("All retry attempts failed")

    def _execute_llm_call(self, step: Step, prompt: str) -> str:
        """
        Execute an LLM call with appropriate retry handling.

        Determines whether to use retry logic based on the step's
        configuration and executes the call accordingly.

        Args:
            step: The step being executed
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response

        Raises:
            ClientAIError: If the LLM call fails

        Example:
            ```python
            response = engine._execute_llm_call(step, "Generate a summary")
            print(response)
            ```
        """
        if step.config.use_internal_retry:
            return self._execute_with_retry(step, prompt)
        return self._execute_single_call(step, prompt)

    def _get_model_name(
        self, model_config: Union[str, ModelConfig, None]
    ) -> str:
        """
        Extract the model name from various configuration formats.

        Args:
            model_config: The model configuration to process

        Returns:
            The model name as a string

        Raises:
            ValueError: If no valid model name can be found

        Example:
            ```python
            name = engine._get_model_name(ModelConfig(name="gpt-4"))
            print(name)  # Output: "gpt-4"
            ```
        """
        if isinstance(model_config, str):
            return model_config
        if isinstance(model_config, ModelConfig):
            return model_config.name
        if isinstance(self._default_model, str):
            return self._default_model
        if isinstance(self._default_model, ModelConfig):
            return self._default_model.name
        raise ValueError("No valid model configuration found")

    def execute_step(
        self, step: Step, agent: Any, *args: Any, **kwargs: Any
    ) -> Optional[str]:
        """
        Execute a single workflow step with full configuration.

        This is the main entry point for step execution. It handles:
        - Tool selection and execution if enabled
        - LLM interaction if required
        - Error handling and retries
        - Result storage in context

        Args:
            step: The step to execute
            agent: The agent instance
            *args: Additional positional arguments for the step
            **kwargs: Additional keyword arguments for the step

        Returns:
            The step execution result, or None if step is disabled/failed

        Raises:
            ClientAIError: If a required step fails
            ValueError: If step returns invalid type

        Example:
            ```python
            result = engine.execute_step(
                step=my_step,
                agent=my_agent,
                input_data="Process this text"
            )
            if result:
                print(f"Step succeeded: {result}")
            ```

        Note:
            - Steps marked as required will raise errors on failure
            - Non-required steps return None on failure
            - Results are automatically stored in agent context
        """
        logger.info(f"Executing step '{step.name}'")
        logger.debug(
            f"Step configuration: use_tools={step.use_tools}, "
            f"send_to_llm={step.send_to_llm}"
        )

        if not step.config.enabled:
            logger.info(f"Step '{step.name}' is disabled, skipping")
            return None

        try:
            if step.send_to_llm:
                logger.debug("Building prompt")
                prompt = self._build_prompt(step, agent, *args, **kwargs)
                logger.debug(f"Built prompt: {prompt}")

                logger.debug("Executing LLM call")
                result = self._execute_llm_call(step, prompt)
                logger.debug(f"LLM result: {result}")
            else:
                result = step.func(agent, *args, **kwargs)
                if not isinstance(result, str):
                    raise ValueError(
                        f"Step function must return str, got {type(result)}"
                    )

            logger.debug(f"Storing result in context with key: {step.name}")
            agent.context.last_results[step.name] = result

            return result

        except ClientAIError as e:
            logger.error(f"Error executing step '{step.name}': {e}")
            if step.config.required:
                raise
            return None
