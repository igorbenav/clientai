import logging
from typing import Any, Dict, Iterator, Optional, Union, cast

from clientai.client_ai import ClientAI
from clientai.exceptions import ClientAIError

from ..config.models import ModelConfig
from ..steps.base import Step
from ..tools import ToolSelectionConfig, ToolSelector
from ..utils.exceptions import StepError, ToolError

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

    Raises:
        StepError: When there's an error in step execution or configuration
        ToolError: When there's an error in tool execution or selection
        ClientAIError: When there's an error in LLM API interaction
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

        Raises:
            StepError: If initialization fails or configuration is invalid
        """
        try:
            if not client:
                raise ValueError("Client must be specified")
            if not default_model:
                raise ValueError("Default model must be specified")

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

        except ValueError as e:
            logger.error(f"Initialization error: {e}")
            raise StepError(str(e)) from e
        except Exception as e:
            logger.error(f"Unexpected initialization error: {e}")
            raise StepError(
                f"Unexpected initialization error: {str(e)}"
            ) from e

    def _get_tool_selection_config(self, step: Step) -> ToolSelectionConfig:
        """
        Get the effective tool selection configuration for a step.

        Determines the tool selection configuration to use by checking for
        step-specific settings and falling back to defaults if needed.

        Args:
            step: The step being executed

        Returns:
            The tool selection configuration to use for this step

        Raises:
            StepError: If configuration access fails

        Example:
            ```python
            config = engine._get_tool_selection_config(step)
            print(config.confidence_threshold)  # Output: 0.8
            ```
        """
        try:
            step_config = getattr(step, "tool_selection_config", None)
            return step_config or self._default_tool_selection_config
        except AttributeError as e:
            logger.error(f"Invalid step configuration: {e}")
            raise StepError(f"Invalid step configuration: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error accessing tool selection config: {e}")
            raise StepError(
                f"Error accessing tool selection config: {str(e)}"
            ) from e

    def _get_tool_model(self, step: Step) -> Union[str, ModelConfig]:
        """
        Get the effective model to use for tool selection in a step.

        Determines which model should be used for tool selection by checking
        step-specific settings and falling back to defaults if needed.

        Args:
            step: The step being executed

        Returns:
            The model configuration to use for tool selection

        Raises:
            StepError: If model configuration access fails

        Example:
            ```python
            model = engine._get_tool_model(step)
            print(model.name if isinstance(model, ModelConfig) else model)
            ```
        """
        try:
            step_model = getattr(step, "tool_model", None)
            return step_model or self._default_tool_model
        except AttributeError as e:
            logger.error(f"Invalid step model configuration: {e}")
            raise StepError(
                f"Invalid step model configuration: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Error accessing tool model: {e}")
            raise StepError(f"Error accessing tool model: {str(e)}") from e

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
            StepError: If prompt building fails
            ToolError: If tool execution fails

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
                try:
                    prompt = self._handle_tool_execution(step, agent, prompt)
                except Exception as e:
                    raise ToolError(f"Tool execution failed: {str(e)}") from e

            logger.debug(f"Final prompt: {prompt}")
            return prompt

        except ValueError as e:
            logger.error(f"Prompt building error: {e}")
            raise StepError(str(e)) from e
        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            raise StepError(f"Failed to build prompt: {str(e)}") from e

    def _handle_tool_execution(
        self, step: Step, agent: Any, base_prompt: str
    ) -> str:
        """
        Handle tool selection and execution for a step.

        Coordinates tool selection, execution, and result formatting
        for inclusion in the step's prompt.

        Args:
            step: The step being executed
            agent: The agent instance
            base_prompt: The initial prompt before tool execution

        Returns:
            The updated prompt including tool execution results

        Raises:
            ToolError: If tool selection or execution fails
        """
        logger.debug("Tool usage is enabled for this step")
        available_tools = agent.get_tools(step.step_type.name.lower())
        logger.debug(f"Found {len(available_tools)} available tools")

        if not available_tools:
            logger.debug("No tools available for this step")
            return base_prompt

        try:
            decisions = self._tool_selector.select_tools(
                task=base_prompt,
                tools=available_tools,
                context=agent.context.state,
                client=self._client,
            )

            logger.debug(f"Tool selector returned {len(decisions)} decisions")

            if not decisions:
                return base_prompt

            tools_by_name = {t.name: t for t in available_tools}
            updated_decisions = self._tool_selector.execute_tool_decisions(
                decisions=decisions, tools=tools_by_name
            )

            object.__setattr__(step, "tool_decisions", updated_decisions)

            prompt = base_prompt + "\n\nTool Execution Results:\n"
            for decision in updated_decisions:
                prompt += self._format_tool_result(decision)

            agent.context.state["last_tool_decisions"] = [
                self._create_decision_dict(d) for d in updated_decisions
            ]

            return prompt

        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            raise ToolError(f"Tool execution failed: {str(e)}") from e

    def _format_tool_result(self, decision: Any) -> str:
        """Format a single tool execution decision into a string."""
        if decision.error:
            result_line = f"Error: {decision.error}"
        else:
            result_line = f"Result: {str(decision.result)}"

        return (
            f"\n{decision.tool_name}:"
            f"\n{result_line}"
            f"\nConfidence: {decision.confidence}"
            f"\nReasoning: {decision.reasoning}\n"
        )

    def _create_decision_dict(self, decision: Any) -> Dict[str, Any]:
        """Create a dictionary representation of a tool execution decision."""
        return {
            "tool_name": decision.tool_name,
            "arguments": decision.arguments,
            "result": decision.result,
            "error": decision.error,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
        }

    def _execute_single_call(
        self,
        step: Step,
        prompt: str,
        api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Execute a single LLM API call with proper configuration.

        Args:
            step: The step being executed
            prompt: The prompt to send to the LLM
            api_kwargs: Optional dictionary of API call arguments

        Returns:
            Either a string or an iterator of strings for streaming responses

        Raises:
            ClientAIError: If the LLM call fails
            StepError: If there's an unexpected error
        """
        try:
            if api_kwargs is None:
                model_config = step.llm_config or self._default_model
                api_kwargs = self._prepare_api_kwargs(model_config)

            result = self._client.generate_text(
                prompt,
                model=self._get_model_name(
                    step.llm_config or self._default_model
                ),
                **api_kwargs,
            )

            return cast(Union[str, Iterator[str]], result)

        except ClientAIError:
            # Preserve the original ClientAIError
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM call: {e}")
            raise StepError(
                f"Unexpected error during LLM call: {str(e)}"
            ) from e

    def _prepare_api_kwargs(
        self, model_config: Union[str, ModelConfig]
    ) -> Dict[str, Any]:
        """
        Prepare keyword arguments for the LLM API call.

        Args:
            model_config: The model configuration to use

        Returns:
            A dictionary of API keyword arguments

        Raises:
            StepError: If API argument preparation fails
        """
        try:
            return {
                **self._default_kwargs,
                **(
                    model_config.get_parameters()
                    if isinstance(model_config, ModelConfig)
                    else {}
                ),
            }
        except Exception as e:
            logger.error(f"Error preparing API arguments: {e}")
            raise StepError(
                f"Failed to prepare API arguments: {str(e)}"
            ) from e

    def _execute_with_retry(
        self,
        step: Step,
        prompt: str,
        api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Execute an LLM call with configurable retry logic.

        Args:
            step: The step being executed
            prompt: The prompt to send to the LLM
            api_kwargs: Optional dictionary of API call arguments

        Returns:
            Either a string or an iterator of strings for streaming responses

        Raises:
            ClientAIError: If all retry attempts fail
        """
        for attempt in range(step.config.retry_count + 1):
            try:
                return self._execute_single_call(step, prompt, api_kwargs)
            except ClientAIError as e:
                if attempt >= step.config.retry_count:
                    logger.error(
                        f"All retry attempts failed for "
                        f"step '{step.name}': {e}"
                    )
                    raise
                logger.warning(
                    f"Retry {attempt + 1}/{step.config.retry_count} "
                    f"for step '{step.name}': {e}"
                )
                continue
        raise ClientAIError("All retry attempts failed")

    def _execute_llm_call(
        self,
        step: Step,
        prompt: str,
        api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Iterator[str]]:
        """
        Execute an LLM call with appropriate retry handling.

        Args:
            step: The step being executed
            prompt: The prompt to send to the LLM
            api_kwargs: Optional dictionary of API call arguments

        Returns:
            Either a string or an iterator of strings for streaming responses

        Raises:
            ClientAIError: If the LLM call fails
        """
        try:
            if step.config.use_internal_retry:
                return self._execute_with_retry(step, prompt, api_kwargs)
            return self._execute_single_call(step, prompt, api_kwargs)
        except ClientAIError:
            raise
        except Exception as e:
            logger.error(f"LLM execution failed: {e}")
            raise StepError(f"LLM execution failed: {str(e)}") from e

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
            StepError: If model name extraction fails

        Example:
            ```python
            name = engine._get_model_name(ModelConfig(name="gpt-4"))
            print(name)  # Output: "gpt-4"
            ```
        """
        try:
            if isinstance(model_config, str):
                return model_config
            if isinstance(model_config, ModelConfig):
                return model_config.name
            if isinstance(self._default_model, str):
                return self._default_model
            if isinstance(self._default_model, ModelConfig):
                return self._default_model.name
            raise ValueError("No valid model configuration found")
        except Exception as e:
            logger.error(f"Error getting model name: {e}")
            raise StepError(f"Failed to get model name: {str(e)}") from e

    def execute_step(
        self,
        step: Step,
        agent: Any,
        *args: Any,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> Optional[Union[str, Iterator[str]]]:
        """
        Execute a single workflow step with full configuration.

        This is the main entry point for step execution. It handles:
        - Tool selection and execution if enabled
        - LLM interaction if required
        - Error handling and retries
        - Result storage in context
        - Stream override handling

        Args:
            step: The step to execute
            agent: The agent instance
            *args: Additional positional arguments for the step
            stream: Optional bool to override step's stream configuration.
                If provided, overrides any step-level streaming settings.
            **kwargs: Additional keyword arguments for the step

        Returns:
            Optional[Union[str, Iterator[str]]]: The step execution result,
                which can be:
                - None if step is disabled or failed
                - A complete string if streaming is disabled
                - An iterator of string chunks if streaming is enabled

        Raises:
            StepError: If step execution fails and the step is required
            ToolError: If tool execution fails
            ClientAIError: If LLM interaction fails

        Example:
            ```python
            # Execute with default stream setting from step
            result = engine.execute_step(step, agent, "input data")

            # Force streaming on for this execution
            result = engine.execute_step(
                step, agent, "input data", stream=True
            )

            # Force streaming off for this execution
            result = engine.execute_step(
                step, agent, "input data", stream=False
            )
            ```
        """
        logger.info(f"Executing step '{step.name}'")
        logger.debug(
            f"Step configuration: use_tools={step.use_tools}, "
            f"send_to_llm={step.send_to_llm}, "
            f"stream={stream}"
        )

        if not step.config.enabled:
            logger.info(f"Step '{step.name}' is disabled, skipping")
            return None

        try:
            result = None

            if step.send_to_llm:
                try:
                    result = self._handle_llm_step(
                        step, agent, stream, args, kwargs
                    )
                except (ClientAIError, StepError, ToolError):
                    raise
                except Exception as e:
                    raise StepError(
                        f"LLM step execution failed: {str(e)}"
                    ) from e
            else:
                try:
                    result = self._handle_non_llm_step(
                        step, agent, args, kwargs
                    )
                except (ValueError, TimeoutError):
                    raise
                except Exception as e:
                    raise StepError(
                        f"Non-LLM step execution failed: {str(e)}"
                    ) from e

            self._update_context(step, agent, result)
            return result

        except (ClientAIError, StepError, ToolError, ValueError, TimeoutError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing step '{step.name}': {e}")
            if step.config.required:
                raise StepError(
                    f"Required step '{step.name}' failed: {str(e)}"
                ) from e
            return None

    def _handle_llm_step(
        self,
        step: Step,
        agent: Any,
        stream: Optional[bool],
        args: Any,
        kwargs: Any,
    ) -> Union[str, Iterator[str]]:
        """
        Handle execution of a step that involves LLM interaction.

        Args:
            step: The step being executed
            agent: The agent instance
            stream: Optional stream configuration override
            args: Positional arguments for the step
            kwargs: Keyword arguments for the step

        Returns:
            Either a string or an iterator of strings for streaming responses

        Raises:
            StepError: If step execution fails
            ToolError: If tool execution fails
            ClientAIError: If LLM interaction fails
        """
        prompt = self._build_prompt(step, agent, *args, **kwargs)
        model_config = self._prepare_model_config(step, stream)
        api_kwargs = self._prepare_api_kwargs(model_config)

        logger.debug(f"Executing LLM call with streaming={stream}")
        return self._execute_llm_call(step, prompt, api_kwargs)

    def _handle_non_llm_step(
        self, step: Step, agent: Any, args: Any, kwargs: Any
    ) -> Union[str, Iterator[str]]:
        """
        Handle execution of a step that doesn't involve LLM interaction.

        Args:
            step: The step being executed
            agent: The agent instance
            args: Positional arguments for the step
            kwargs: Keyword arguments for the step

        Returns:
            Either a string or an iterator of strings

        Raises:
            StepError: If step execution fails or returns invalid type
        """
        result = step.func(agent, *args, **kwargs)
        if not isinstance(result, (str, Iterator)):  # noqa: UP038
            raise StepError(
                f"Step function must return str or Iterator, "
                f"got {type(result)}"
            )
        return result

    def _prepare_model_config(
        self, step: Step, stream: Optional[bool]
    ) -> Union[str, ModelConfig]:
        """
        Prepare the model configuration for a step execution.

        Args:
            step: The step being executed
            stream: Optional stream configuration override

        Returns:
            The prepared model configuration

        Raises:
            StepError: If model configuration preparation fails
        """
        try:
            model_config = step.llm_config or self._default_model

            if isinstance(model_config, ModelConfig):
                if stream is not None:
                    return model_config.merge(stream=stream)
                return model_config

            return ModelConfig(
                name=model_config,
                stream=stream if stream is not None else False,
            )
        except Exception as e:
            logger.error(f"Error preparing model configuration: {e}")
            raise StepError(
                f"Failed to prepare model configuration: {str(e)}"
            ) from e

    def _update_context(
        self,
        step: Step,
        agent: Any,
        result: Optional[Union[str, Iterator[str]]],
    ) -> None:
        """
        Update the agent's context with step execution results.

        Args:
            step: The executed step
            agent: The agent instance
            result: The step execution result

        Raises:
            StepError: If context update fails
        """
        try:
            if result is not None and not isinstance(result, Iterator):
                agent.context.last_results[step.name] = result
                if step.config.pass_result:
                    agent.context.current_input = result
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            raise StepError(f"Failed to update context: {str(e)}") from e
