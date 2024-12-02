import logging
from typing import Any, Dict, Optional, Union

from ...client_ai import ClientAI
from ...exceptions import ClientAIError
from ..config.models import ModelConfig
from ..steps.base import Step

logger = logging.getLogger(__name__)


class StepExecutionEngine:
    """
    Handles the execution of individual workflow steps in an agent system.

    This class manages the execution of workflow steps, including
    LLM interactions, retry logic, and prompt construction.
    It serves as the core execution engine for processing steps
    within an agent's workflow.

    Attributes:
        _client: The AI client interface for API communication.
        _default_model: Default model configuration for LLM calls.
        _default_kwargs: Default parameters for LLM calls.

    Example:
        >>> client = ClientAI(provider="openai")
        >>> engine = StepExecutionEngine(
        ...     client,
        ...     default_model="gpt-4",
        ...     default_kwargs={"temperature": 0.7}
        ... )
        >>> result = engine.execute_step(step, agent, "input data")
    """

    def __init__(
        self,
        client: ClientAI,
        default_model: Union[str, ModelConfig],
        default_kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize the execution engine with necessary configurations.

        Args:
            client: The AI client interface for making API calls.
            default_model: Default model configuration or name to use for
                           LLM calls.
            default_kwargs: Additional default parameters for LLM API calls.

        Example:
            >>> engine = StepExecutionEngine(
            ...     client=ClientAI(provider="openai"),
            ...     default_model="gpt-4",
            ...     default_kwargs={"temperature": 0.7}
            ... )
        """
        self._client = client
        self._default_model = default_model
        self._default_kwargs = default_kwargs
        logger.debug(
            f"Initialized StepExecutionEngine with model: {default_model}"
        )

    def _build_prompt(
        self, step: Step, agent: Any, *args: Any, **kwargs: Any
    ) -> str:
        """
        Construct a prompt for LLM-based steps by combining
        step output with tool descriptions.

        This method processes the step's function output and augments it with
        descriptions of available tools if any exist for the current step type.

        Args:
            step: The workflow step requiring a prompt.
            agent: The agent instance executing the step.
            *args: Variable positional arguments for the step function.
            **kwargs: Variable keyword arguments for the step function.

        Returns:
            A formatted prompt string combining the
            step output and tool descriptions.

        Raises:
            ValueError: If the step function does not return a string.
            Exception: If prompt construction fails for any other reason.

        Example:
            >>> prompt = engine._build_prompt(step, agent, "input")
            >>> print(prompt)
            'Analyze this input\nAvailable tools:\n- Calculator(x: int) -> ...'
        """
        logger.debug(f"Building prompt for step '{step.name}'")
        logger.debug(f"Step type: {step.step_type}")
        logger.debug(f"Input args: {args}")
        logger.debug(f"Input kwargs: {kwargs}")
        logger.debug(f"Current context: {agent.context.last_results}")

        try:
            result = step.func(agent, *args, **kwargs)
            if not isinstance(result, str):
                raise ValueError(
                    f"Step function must return str, got {type(result)}"
                )
            prompt = result
            logger.debug(f"Base prompt: {prompt}")

            available_tools = agent.get_tools(step.step_type.name.lower())
            if available_tools:
                logger.debug(
                    f"Available tools: {[t.name for t in available_tools]}"
                )
                tool_descriptions = self._format_tool_descriptions(
                    available_tools
                )
                prompt = f"{prompt}\n{tool_descriptions}"
                logger.debug(f"Final prompt with tools: {prompt}")

            return prompt

        except Exception as e:
            logger.error(f"Error building prompt for step '{step.name}': {e}")
            raise

    def _format_tool_descriptions(self, tools: list) -> str:
        """
        Format a list of tools into a string
        description suitable for LLM prompts.

        Creates a formatted string containing each tool's
        signature and description, organized in a clear,
        readable format.

        Args:
            tools: List of Tool instances to describe.

        Returns:
            A formatted string containing tool descriptions,
            or an empty string if no tools.

        Example:
            >>> tools = [Calculator(), TextProcessor()]
            >>> descriptions = engine._format_tool_descriptions(tools)
            >>> print(descriptions)
            'Available tools:\n- Calculator(x: int) -> int\n  Performs calc...'
        """
        if not tools:
            return ""

        tool_lines = ["Available tools:"]
        for tool in tools:
            tool_lines.append(f"- {tool.signature_str}")
            tool_lines.append(f"  {tool.description}")

        return "\n".join(tool_lines)

    def _get_model_name(
        self, model_config: Union[str, ModelConfig, None]
    ) -> str:
        """
        Extract the model name from various possible configuration formats.

        Handles different types of model configurations and falls back to
        default model if necessary.

        Args:
            model_config: The model configuration to process, which can be a
                          string, ModelConfig instance, or None.

        Returns:
            The extracted model name as a string.

        Raises:
            ValueError: If no valid model configuration can be found.

        Example:
            >>> name = engine._get_model_name(ModelConfig(name="gpt-4"))
            >>> print(name)
            'gpt-4'
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

    def _execute_single_call(self, step: Step, prompt: str) -> str:
        """
        Execute a single LLM API call without retry logic.

        Prepares the API call parameters and executes the call, ensuring
        proper type checking of the response.

        Args:
            step: The workflow step requesting the LLM call.
            prompt: The prepared prompt to send to the LLM.

        Returns:
            The LLM's response as a string.

        Raises:
            ClientAIError: If the API call fails or returns an invalid
                           response type.

        Example:
            >>> response = engine._execute_single_call(step, "Analyze text")
            >>> print(response)
            'The text contains...'
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
        Execute an LLM call with retry logic for handling transient failures.

        Attempts the call multiple times based on the step's retry
        configuration, logging warnings for failed attempts.

        Args:
            step: The workflow step requiring the LLM call.
            prompt: The prompt to send to the LLM.

        Returns:
            The successful LLM response as a string.

        Raises:
            ClientAIError: If all retry attempts fail.

        Example:
            >>> response = engine._execute_with_retry(step, "Analyze this")
            >>> print(response)
            'Analysis complete...'
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
        Execute an LLM call with appropriate retry handling
        based on step configuration.

        Determines whether to use retry logic based on the step's
        configuration and executes the call accordingly.

        Args:
            step: The workflow step requiring the LLM call.
            prompt: The prompt to send to the LLM.

        Returns:
            The LLM's response as a string.

        Raises:
            ClientAIError: If the LLM call fails.

        Example:
            >>> response = engine._execute_llm_call(step, "Generate a summary")
            >>> print(response)
            'Summary: The text discusses...'
        """
        if step.config.use_internal_retry:
            return self._execute_with_retry(step, prompt)
        return self._execute_single_call(step, prompt)

    def execute_step(
        self, step: Step, agent: Any, *args: Any, **kwargs: Any
    ) -> Optional[str]:
        """
        Execute a single workflow step with appropriate error handling
        and logging.

        This is the main entry point for step execution. It handles both
        LLM-based and local function execution, manages results storage,
        and provides comprehensive error handling.

        Args:
            step: The workflow step to execute.
            agent: The agent instance executing the step.
            *args: Variable positional arguments for the step.
            **kwargs: Variable keyword arguments for the step.

        Returns:
            The step execution result as a string, or None if the
            step is disabled or fails non-critically.

        Raises:
            ClientAIError: If a required step fails to execute.
            ValueError: If the step function returns an invalid type.

        Example:
            >>> result = engine.execute_step(step, agent, "input data")
            >>> print(result)
            'Processed result: input data has been analyzed...'
        """
        logger.info(f"Executing step '{step.name}'")

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
