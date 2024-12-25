import logging
from typing import Any, Dict, Iterator, Optional, Union, cast

from clientai.client_ai import ClientAI
from clientai.exceptions import ClientAIError

from ..config.models import ModelConfig
from ..steps.base import Step
from ..tools import ToolSelectionConfig, ToolSelector
from ..utils.exceptions import StepError, ToolError
from ..validation import (
    OutputFormat,
    StepValidator,
    ValidationError,
    ValidatorContext,
)

logger = logging.getLogger(__name__)


class StepExecutionEngine:
    """Handles the execution of workflow steps with
    integrated tool selection and LLM interaction.

    Manages all aspects of step execution including tool selection,
    LLM interaction, and error handling. Provides configurable retry
    logic and streaming support.

    Attributes:
        _client: The AI client for model interactions
        _default_model: Default model configuration for steps
        _default_kwargs: Default keyword arguments for model calls
        _default_tool_selection_config: Default tool selection settings
        _default_tool_model: Default model for tool selection
        _tool_selector: Instance handling tool selection logic

    Example:
        Basic execution engine setup:
        ```python
        engine = StepExecutionEngine(
            client=client,
            default_model=ModelConfig(name="gpt-4"),
            default_kwargs={"temperature": 0.7}
        )

        result = engine.execute_step(step, agent, "input data")
        ```

        Configure tool selection:
        ```python
        engine = StepExecutionEngine(
            client=client,
            default_model="gpt-4",
            default_kwargs={},
            tool_selection_config=ToolSelectionConfig(
                confidence_threshold=0.8,
                max_tools_per_step=3
            ),
            tool_model=ModelConfig(name="llama-2")
        )
        ```

    Notes:
        - Manages automatic tool selection with confidence thresholds
        - Supports separate models for workflow and tool selection
        - Implements retry logic for LLM calls
        - Handles streaming configurations
        - Provides comprehensive error handling
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
            if not client:  # pragma: no cover
                raise ValueError("Client must be specified")
            if not default_model:  # pragma: no cover
                raise ValueError("Default model must be specified")

            self._client = client
            self._default_model = default_model
            self._default_kwargs = default_kwargs
            self._current_agent: Optional[Any] = None
            self._default_tool_selection_config = (
                tool_selection_config or ToolSelectionConfig()
            )

            self._default_tool_model = self._create_tool_model_config(
                tool_model if tool_model is not None else default_model
            )

            logger.debug(
                f"Initialized StepExecutionEngine with model: {default_model}"
            )

        except ValueError as e:  # pragma: no cover
            logger.error(f"Initialization error: {e}")
            raise StepError(str(e)) from e
        except Exception as e:  # pragma: no cover
            logger.error(f"Unexpected initialization error: {e}")
            raise StepError(
                f"Unexpected initialization error: {str(e)}"
            ) from e

    def _create_tool_model_config(
        self, model: Union[str, ModelConfig]
    ) -> ModelConfig:
        """Create a ModelConfig instance for tool selection.

        Args:
            model: Model name or configuration

        Returns:
            Configured ModelConfig for tool selection

        Raises:
            StepError: If configuration is invalid
        """
        try:
            if isinstance(model, str):
                return ModelConfig(
                    name=model,
                    temperature=0.0,
                    json_output=True,
                )
            elif isinstance(model, ModelConfig):
                return model.merge(
                    temperature=0.0,
                    json_output=True,
                )
            else:  # pragma: no cover
                raise ValueError(
                    f"Invalid model type: {type(model)}. "
                    "Must be string or ModelConfig"
                )
        except Exception as e:  # pragma: no cover
            logger.error(f"Error creating tool model config: {e}")
            raise StepError(f"Invalid tool model configuration: {str(e)}")

    def _get_effective_tool_model(self, step: Step) -> ModelConfig:
        """Get the model to use for tool selection based on priority order.

        Args:
            step: Step being executed

        Returns:
            ModelConfig for tool selection

        Raises:
            StepError: If model configuration fails
        """
        try:
            if step.tool_model is not None:
                return self._create_tool_model_config(step.tool_model)

            return self._default_tool_model

        except Exception as e:  # pragma: no cover
            logger.error(f"Error determining tool model: {e}")
            raise StepError(f"Failed to determine tool model: {str(e)}")

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
        except AttributeError as e:  # pragma: no cover
            logger.error(f"Invalid step configuration: {e}")
            raise StepError(f"Invalid step configuration: {str(e)}") from e
        except Exception as e:  # pragma: no cover
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
        except AttributeError as e:  # pragma: no cover
            logger.error(f"Invalid step model configuration: {e}")
            raise StepError(
                f"Invalid step model configuration: {str(e)}"
            ) from e
        except Exception as e:  # pragma: no cover
            logger.error(f"Error accessing tool model: {e}")
            raise StepError(f"Error accessing tool model: {str(e)}") from e

    def _build_prompt(
        self, step: Step, agent: Any, *args: Any, **kwargs: Any
    ) -> str:
        """Build the prompt for a step, including tool selection if enabled.

        Args:
            step: Step being executed
            agent: Agent instance
            *args: Step arguments
            **kwargs: Step keyword arguments

        Returns:
            Complete prompt string

        Raises:
            StepError: If prompt building fails
            ToolError: If tool execution fails
        """
        logger.debug(f"Building prompt for step '{step.name}'")
        logger.debug(f"Step use_tools setting: {step.use_tools}")

        try:
            result = step.func(agent, *args, **kwargs)
            if not isinstance(result, str):  # pragma: no cover
                raise ValueError(
                    f"Step function must return str, got {type(result)}"
                )
            prompt = result

            if step.use_tools:
                try:
                    prompt = self._handle_tool_execution(step, agent, prompt)
                except Exception as e:  # pragma: no cover
                    raise ToolError(f"Tool execution failed: {str(e)}") from e

            logger.debug(f"Final prompt: {prompt}")
            return prompt

        except ValueError as e:  # pragma: no cover
            logger.error(f"Prompt building error: {e}")
            raise StepError(str(e)) from e
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to build prompt: {e}")
            raise StepError(f"Failed to build prompt: {str(e)}") from e

    def _handle_tool_execution(
        self, step: Step, agent: Any, base_prompt: str
    ) -> str:
        """Handle tool selection and execution for a step.

        Args:
            step: The step being executed
            agent: The agent instance
            base_prompt: Initial prompt before tool execution

        Returns:
            Updated prompt including tool execution results

        Raises:
            ToolError: If tool selection or execution fails

        Example:
            ```python
            prompt = engine._handle_tool_execution(
                step=analyze_step,
                agent=agent,
                base_prompt="Analyze the following data..."
            )
            # Returns prompt enhanced with tool execution results
            ```
        """
        logger.debug("Tool usage is enabled for this step")
        available_tools = agent.get_tools(step.step_type.name.lower())
        logger.debug(f"Found {len(available_tools)} available tools")

        if not available_tools:
            logger.debug("No tools available for this step")
            return base_prompt

        try:
            tool_model = self._get_effective_tool_model(step)
            logger.debug(f"Using tool model: {tool_model.name}")

            tool_selector = ToolSelector(
                model_config=tool_model,
                config=self._get_tool_selection_config(step),
            )

            decisions = tool_selector.select_tools(
                task=base_prompt,
                tools=available_tools,
                context=agent.context.state,
                client=self._client,
            )

            logger.debug(f"Tool selector returned {len(decisions)} decisions")

            if not decisions:
                return base_prompt

            tools_by_name = {t.name: t for t in available_tools}
            updated_decisions = tool_selector.execute_tool_decisions(
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

        except Exception as e:  # pragma: no cover
            logger.error(f"Tool execution error: {e}", exc_info=True)
            raise ToolError(f"Tool execution failed: {str(e)}") from e

    def _format_tool_result(self, decision: Any) -> str:
        """Format a single tool execution decision into a string.

        Args:
            decision: Tool execution decision with results

        Returns:
            Formatted string representing the tool execution result

        Example:
            ```python
            formatted = engine._format_tool_result(decision)
            # Output format:
            # CalculatorTool:
            # Result: 42
            # Confidence: 0.95
            # Reasoning: Used for precise calculation
            ```
        """
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
        """Create a dictionary representation of a tool execution decision.

        Args:
            decision: Tool execution decision to convert

        Returns:
            Dictionary containing all decision information

        Example:
            ```python
            decision_dict = engine._create_decision_dict(decision)
            # Output structure:
            # {
            #     "tool_name": "calculator",
            #     "arguments": {"x": 5, "y": 3},
            #     "result": 8,
            #     "error": None,
            #     "confidence": 0.95,
            #     "reasoning": "Required for calculation"
            # }
            ```
        """
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
        """Execute a single LLM API call with proper configuration."""
        try:
            model_config = step.llm_config or self._default_model

            if isinstance(model_config, str):
                model_config = ModelConfig(name=model_config)

            param_attrs = ModelConfig.CORE_ATTRS - {"name"}

            step_params = {
                k: getattr(step, k) for k in param_attrs if hasattr(step, k)
            }

            effective_params = model_config.get_parameters()
            effective_params.update(self._default_kwargs)
            effective_params.update(step_params)

            if api_kwargs:
                additional_kwargs = {
                    k: v
                    for k, v in api_kwargs.items()
                    if k not in ModelConfig.CORE_ATTRS
                }
                effective_params.update(additional_kwargs)

            result = self._client.generate_text(
                prompt, model=model_config.name, **effective_params
            )

            return cast(Union[str, Iterator[str]], result)

        except ClientAIError:  # pragma: no cover
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Unexpected error during LLM call: {e}")
            raise StepError(
                f"Unexpected error during LLM call: {str(e)}"
            ) from e

    def _prepare_api_kwargs(
        self, model_config: Union[str, ModelConfig]
    ) -> Dict[str, Any]:
        """Prepare keyword arguments for the LLM API call.

        Args:
            model_config: The model configuration to use

        Returns:
            Dictionary of API keyword arguments

        Raises:
            StepError: If API argument preparation fails
        """
        try:
            kwargs = self._default_kwargs.copy()

            if isinstance(model_config, ModelConfig):
                model_params = model_config.get_parameters()
                kwargs.update(model_params)

            return kwargs
        except Exception as e:  # pragma: no cover
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
        Execute an LLM call using the appropriate execution strategy.

        Args:
            step: The step being executed
            prompt: The prompt to send to the LLM
            api_kwargs: Optional dictionary of API call arguments

        Returns:
            Either a string or an iterator of strings for streaming responses

        Raises:
            ClientAIError: If the LLM call fails
        """
        for attempt in range(step.config.retry_count + 1):
            try:
                return self._execute_single_call(step, prompt, api_kwargs)
            except ClientAIError as e:  # pragma: no cover
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
        except ClientAIError:  # pragma: no cover
            raise
        except Exception as e:
            logger.error(f"LLM execution failed: {e}")
            raise StepError(f"LLM execution failed: {str(e)}") from e

    def _get_model_name(
        self, model_config: Union[str, ModelConfig, None]
    ) -> str:
        """Extract the model name from various configuration formats.

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

    def _validate_step_output(
        self,
        step: Step,
        result: Union[str, Iterator[str]],
    ) -> Any:
        """Validate step output if validation is enabled.

        Args:
            step: The executed step
            result: Raw result from step execution. May be:
                - String: Direct response
                - Iterator[str]: Streamed response chunks
            stream: Whether this is a streaming response (currently unused)

        Returns:
            Any: Either:
                - The validated data if validation is enabled and successful
                - The original result if validation is not enabled

        Raises:
            ValidationError: If validation fails for the output data
            StepError: If an unexpected error occurs during validation
        """
        validator = StepValidator.from_step(step)
        if not validator:
            return result

        try:
            context: ValidatorContext = ValidatorContext(
                data=result,
                format=OutputFormat.JSON,
                partial=False,
                metadata={
                    "step_name": step.name,
                    "step_type": step.step_type,
                },
            )
            validation_result = validator.validate(result, context)
            if validation_result.is_valid:
                return validation_result.data

            error_str = "\n".join(
                f"{k}: {v}" for k, v in validation_result.errors.items()
            )
            raise ValidationError(error_str)

        except ValidationError:  # pragma: no cover
            raise
        except Exception as e:  # pragma: no cover
            raise StepError(f"Unexpected validation error: {str(e)}")

    def execute_step(
        self,
        step: Step,
        *args: Any,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> Optional[Union[str, Iterator[str]]]:
        """Execute a single workflow step with full configuration.

        Main entry point for step execution, handling tool selection,
        LLM interaction, validation, error handling, and result management.

        Args:
            step: The step to execute
            *args: Additional positional arguments for the step
            stream: Optional bool to override step's stream configuration
            **kwargs: Additional keyword arguments for the step

        Returns:
            Returns:
                Optional[Union[str, Iterator[str], Any]]:
                    The step execution result:
                    - None if step is disabled or failed
                    - Complete string if streaming is disabled
                    - Iterator of string chunks if streaming is enabled
                    - Any type for validated outputs from json_output steps

        Raises:
            StepError: If step execution fails and step is required
            ToolError: If tool execution fails
            ClientAIError: If LLM interaction fails
            ValidationError: If output validation fails for json_output steps

        Example:
            Basic step execution:
            ```python
            # Execute with default stream setting
            result = engine.execute_step(step, agent, "input data")

            # Force streaming on
            result = engine.execute_step(
                step,
                agent,
                "input data",
                stream=True
            )

            # Handle streaming results
            if isinstance(result, Iterator):
                for chunk in result:
                    print(chunk, end="")
            else:
                print(result)
            ```

            With validation:
            ```python
            class OutputModel(BaseModel):
                value: str
                score: float

            @think("analyze", json_output=True)
            def analyze(self, data: str) -> OutputModel:
                return "..."
            ```

        Notes:
            - Handles both streaming and non-streaming responses
            - Manages tool selection if enabled for step
            - Updates agent context with results
            - Supports retry logic for failed steps
            - Validates output against Pydantic models when json_output=True
            - Supports partial validation during streaming
        """
        if self._current_agent is None:  # pragma: no cover
            raise StepError("No agent context available for step execution")

        logger.info(f"Executing step '{step.name}'")
        if stream is None:  # pragma: no cover
            stream = getattr(step, "stream", False)

        logger.debug(
            f"Step configuration: use_tools={step.use_tools}, "
            f"send_to_llm={step.send_to_llm}, "
            f"json_output={getattr(step, 'json_output', False)}"
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
                        step, self._current_agent, stream, args, kwargs
                    )
                except (ClientAIError, StepError, ToolError):
                    raise
                except Exception as e:  # pragma: no cover
                    raise StepError(
                        f"LLM step execution failed: {str(e)}"
                    ) from e
            else:
                try:
                    result = self._handle_non_llm_step(
                        step, self._current_agent, args, kwargs
                    )
                except (ValueError, TimeoutError):  # pragma: no cover
                    raise
                except Exception as e:  # pragma: no cover
                    raise StepError(
                        f"Non-LLM step execution failed: {str(e)}"
                    ) from e

            if result is not None:
                result = self._validate_step_output(step, result)

            self._update_context(step, self._current_agent, result)
            return result

        except (
            ClientAIError,
            StepError,
            ToolError,
            ValueError,
            TimeoutError,
            ValidationError,
        ):
            raise
        except Exception as e:  # pragma: no cover
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
        """Handle execution of a step that involves LLM interaction.

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
        """Handle execution of a step that doesn't involve LLM interaction.

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
        """Prepare the model configuration for a step execution.

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
        except Exception as e:  # pragma: no cover
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
        """Update the agent's context with step execution results.

        Args:
            step: Executed step
            agent: Agent instance
            result: Step execution result

        Raises:
            StepError: If context update fails
        """
        try:
            if result is not None and not isinstance(result, Iterator):
                agent.context.set_step_result(step.name, result)

                if step.config.pass_result:
                    agent.context.current_input = result

        except Exception as e:  # pragma: no cover
            logger.error(f"Error updating context: {e}")
            raise StepError(f"Failed to update context: {str(e)}") from e
