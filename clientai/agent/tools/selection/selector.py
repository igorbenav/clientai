import json
import logging
from typing import Any, Dict, List, Optional

from ...config.models import ModelConfig
from ...tools.base import Tool
from .config import ToolCallDecision, ToolSelectionConfig

logger = logging.getLogger(__name__)


class ToolSelector:
    """
    Manages the automatic selection and execution
    of tools using LLM-based decision making.

    The ToolSelector uses a language model to analyze tasks and determine
    which tools would be most appropriate to use, considering both the
    task requirements and the current context. It provides a complete
    pipeline for tool selection, validation, and execution with
    comprehensive error handling and logging.

    The selector's key responsibilities include:
    1. Analyzing tasks and context to determine tool requirements
    2. Selecting appropriate tools based on capabilities and confidence
    3. Validating tool arguments before execution
    4. Managing tool execution and error handling
    5. Providing detailed logging and error reporting

    Key Features:
        - LLM-based tool selection with context awareness
        - Configurable confidence thresholds for tool selection
        - Automatic argument validation against tool signatures
        - Comprehensive error handling and recovery
        - Detailed execution logging and debugging support
        - Context-aware decision making

    Attributes:
        config: Configuration for tool selection behavior,
            including confidence thresholds and tool limits
        model_config: Configuration for the LLM used in selection,
            including model name and parameters

    Example:
        ```python
        # Initialize selector with custom configuration
        selector = ToolSelector(
            config=ToolSelectionConfig(
                confidence_threshold=0.8,
                max_tools_per_step=3
            ),
            model_config=ModelConfig(
                name="gpt-4",
                temperature=0.0
            )
        )

        # Select tools for a task with context
        decisions = selector.select_tools(
            task="Calculate the average daily sales",
            tools=[calculator, aggregator],
            context={"sales_data": [100, 200, 300]},
            client=llm_client
        )

        # Execute the selected tools
        results = selector.execute_tool_decisions(
            decisions=decisions,
            tools={"calculator": calculator, "aggregator": aggregator}
        )

        # Process results
        for result in results:
            if result.error:
                print(f"Error in {result.tool_name}: {result.error}")
            else:
                print(f"Result from {result.tool_name}: {result.result}")
        ```
    """

    def __init__(
        self,
        model_config: ModelConfig,
        config: Optional[ToolSelectionConfig] = None,
    ) -> None:
        """
        Initialize the ToolSelector with the specified configurations.

        Sets up the selector with either provided configurations or defaults.
        Initializes logging and validates configuration parameters.

        Args:
            model_config: Configuration for the LLM used in selection.
            config: Configuration for tool selection behavior. If None,
                   uses default configuration with standard thresholds
                   and limits.

        Raises:
            ValueError: If provided configurations contain invalid values.

        Example:
            ```python
            # With default configuration
            selector = ToolSelector()

            # With custom configuration
            selector = ToolSelector(
                config=ToolSelectionConfig(confidence_threshold=0.8),
                model_config=ModelConfig(name="gpt-4", temperature=0.0)
            )
            ```
        """
        self.config = config or ToolSelectionConfig()

        self.model_config = model_config.merge(stream=False, json_output=True)

        logger.debug("Initialized ToolSelector")
        logger.debug(f"Using model: {self.model_config.name}")
        logger.debug(
            f"Confidence threshold: {self.config.confidence_threshold}"
        )

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format the context dictionary into a structured string for LLM prompts.

        Converts a context dictionary into a formatted string representation
        that clearly presents the available context information to the LLM.
        The format is designed to be both human-readable and easily parseable
        by the LLM.

        Args:
            context: Dictionary containing contextual information that may be
                    relevant for tool selection. Can include any serializable
                    data relevant to the task.

        Returns:
            A formatted string representation of the context, organized in a
            hierarchical structure with clear labeling.

        Example:
            ```python
            context_str = selector._format_context({
                "user": "John Doe",
                "current_data": [1, 2, 3],
                "preferences": {"format": "json", "units": "metric"}
            })
            # Output:
            # Current Context:
            # - user: John Doe
            # - current_data: [1, 2, 3]
            # - preferences: {"format": "json", "units": "metric"}
            ```

        Note:
            - The output format is designed to be clear and consistent
            - Complex nested structures are preserved in their
              string representation
            - Empty context is handled with a clear "no context" message
        """
        if not context:
            return "No additional context available."

        formatted = ["Current Context:"]
        for key, value in context.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

    def _format_tools(self, tools: List[Tool]) -> str:
        """
        Format a list of tools into a structured string for LLM prompts.

        Creates a consolidated string representation of all available tools
        using each tool's standardized format_tool_info() method. This ensures
        consistent formatting throughout the application and provides clear,
        complete tool information to the LLM.

        Args:
            tools: List of Tool instances to format. Each tool should provide
                  its name, signature, and description.

        Returns:
            A formatted string containing the complete information for
            all tools, organized in a clear, hierarchical structure.

        Example:
            ```python
            tool_str = selector._format_tools([
                calculator_tool,
                text_processor_tool
            ])
            # Output:
            # - Calculator
            #   Signature: add(x: int, y: int) -> int
            #   Description: Adds two numbers together
            # - TextProcessor
            #   Signature: process(text: str, uppercase: bool = False) -> str
            #   Description: Processes text with optional case conversion
            ```

        Note:
            - Uses the standard Tool.format_tool_info() method for consistency
            - Maintains proper indentation and structure
            - Separates tools with newlines for clarity
            - Preserves complete signature information including defaults
        """
        return "\n".join(tool.format_tool_info() for tool in tools)

    def select_tools(
        self,
        task: str,
        tools: List[Tool],
        context: Dict[str, Any],
        client: Any,
    ) -> List[ToolCallDecision]:
        """Use LLM to select appropriate tools for a given task.

        Analyzes the task description, available tools, and current context
        to determine which tools would be most appropriate to use.
        Makes selections based on confidence thresholds,
        validates arguments, and provides reasoning.

        Args:
            task: Description of the task to accomplish.
            tools: List of available tools to choose from.
            context: Current execution context and state information.
            client: LLM client for making selection decisions.

        Returns:
            List of ToolCallDecision objects containing selected tools,
            arguments, confidence levels, and reasoning.

        Raises:
            StepError: If LLM interaction fails.
            ToolError: If tool validation fails.

        Example:
            ```python
            decisions = selector.select_tools(
                task="Calculate average daily sales increase",
                tools=[calculator, aggregator],
                context={"sales_data": [100, 200, 300]},
                client=llm_client
            )

            for decision in decisions:
                print(f"Selected: {decision.tool_name}")
                print(f"Arguments: {decision.arguments}")
                print(f"Confidence: {decision.confidence}")
            ```
        """
        logger.debug(f"Selecting tools for task: {task}")
        logger.debug(f"Available tools: {[t.name for t in tools]}")
        logger.debug(f"Context keys: {list(context.keys())}")

        if not tools:
            logger.debug("No tools available, returning empty list")
            return []

        prompt = f"""
        You are a helpful AI that uses tools to solve problems.

        Task: {task}

        {self._format_context(context)}

        Available Tools:
        {self._format_tools(tools)}

        Respond ONLY with a valid JSON object.
        Do not include any comments or placeholders.
        The JSON must follow this exact structure:
        {{
            "tool_calls": [
                {{
                    "tool_name": "name_of_tool",
                    "arguments": {{
                        "param1": "value1",
                        "param2": "value2"
                    }},
                    "confidence": 0.0,
                    "reasoning": "Clear explanation of why the tool was chosen"
                }}
            ]
        }}

        All values must be concrete and complete.
        Do not use placeholders or temporary values.
        Each tool_name must exactly match one of the available tools.
        All required parameters for the chosen tool must be provided.
        Confidence must be a number between 0.0 and 1.0.
        """

        logger.debug(f"Generated tool selection prompt:\n{prompt}")

        try:
            logger.debug("Requesting tool selection from LLM")
            response = client.generate_text(
                prompt,
                model=self.model_config.name,
                **self.model_config.get_parameters(),
            )
            logger.debug(f"Raw LLM response: {response}")

            try:
                decisions = json.loads(response)
                logger.debug(f"Parsed decisions: {decisions}")
            except json.JSONDecodeError:  # pragma: no cover
                logger.warning(
                    "Failed to parse JSON response, attempting to extract JSON"
                )
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    decisions = json.loads(json_str)
                    logger.debug(
                        f"Successfully extracted and parsed JSON: {decisions}"
                    )
                else:
                    logger.error("Could not find valid JSON in response")
                    return []

        except Exception as e:  # pragma: no cover
            logger.error(
                f"Failed to get or parse LLM tool selection response: {e}"
            )
            return []

        tool_decisions = []
        tools_by_name = {tool.name: tool for tool in tools}

        for call in decisions.get("tool_calls", []):
            tool_name = call.get("tool_name")
            if tool_name not in tools_by_name:
                logger.warning(
                    f"Tool '{tool_name}' not found in available tools"
                )
                continue

            arguments = call.get("arguments", {})
            confidence = call.get("confidence", 0.0)
            reasoning = call.get("reasoning", "No reasoning provided")

            logger.debug(f"Processing decision for tool '{tool_name}':")
            logger.debug(f"  Arguments: {arguments}")
            logger.debug(f"  Confidence: {confidence}")
            logger.debug(f"  Reasoning: {reasoning}")

            if confidence < self.config.confidence_threshold:
                logger.debug(
                    f"Skipping tool {tool_name} "
                    "due to low confidence: {confidence}"
                )
                continue

            if not self._validate_tool_arguments(
                tools_by_name[tool_name], arguments
            ):
                logger.warning(
                    f"Invalid arguments for tool {tool_name}: {arguments}"
                )
                continue

            tool_decisions.append(
                ToolCallDecision(
                    tool_name=tool_name,
                    arguments=arguments,
                    confidence=confidence,
                    reasoning=reasoning,
                )
            )
            logger.debug(f"Added decision for tool '{tool_name}'")

        logger.debug(
            f"Final tool decisions: {[d.tool_name for d in tool_decisions]}"
        )
        return tool_decisions

    def _validate_tool_arguments(
        self, tool: Tool, arguments: Dict[str, Any]
    ) -> bool:
        """
        Validate that provided arguments match a tool's signature.

        Performs comprehensive validation of tool arguments against the tool's
        signature, ensuring both required arguments are present and no unknown
        arguments are included.

        Args:
            tool: The tool whose arguments to validate, containing signature
                 information.
            arguments: Dictionary of argument names to values to validate.

        Returns:
            bool: True if all arguments are valid, False otherwise.

        Example:
            ```python
            # Valid argument check
            is_valid = selector._validate_tool_arguments(
                tool=calculator,
                arguments={"x": 5, "y": 3}
            )
            if not is_valid:
                print("Invalid arguments provided")

            # Invalid argument check (missing required)
            is_valid = selector._validate_tool_arguments(
                tool=calculator,
                arguments={"x": 5}  # missing 'y'
            )
            # Returns False, logs warning about missing parameter

            # Invalid argument check (unknown argument)
            is_valid = selector._validate_tool_arguments(
                tool=calculator,
                arguments={"x": 5, "y": 3, "z": 10}  # 'z' is unknown
            )
            # Returns False, logs warning about unknown parameter
            ```

        Note:
            - Validates presence of all required parameters
            - Checks for unknown parameters
            - Logs specific validation failures for debugging
            - Does not validate argument types (done at execution)
        """
        param_info = tool.signature.parameters

        required_params = {
            name for name, info in param_info if info.default is None
        }
        if not all(param in arguments for param in required_params):
            logger.warning(
                f"Missing required parameters for {tool.name}: "
                f"needs {required_params}, got {list(arguments.keys())}"
            )
            return False

        valid_params = {name for name, _ in param_info}
        if not all(arg in valid_params for arg in arguments):
            logger.warning(
                f"Unknown parameters for {tool.name}: "
                f"valid parameters are {valid_params}, "
                f"got {list(arguments.keys())}"
            )
            return False

        return True

    def execute_tool_decisions(
        self, decisions: List[ToolCallDecision], tools: Dict[str, Tool]
    ) -> List[ToolCallDecision]:
        """
        Execute a series of tool decisions and capture their results.

        Takes a list of validated tool selection decisions and executes each
        tool with its specified arguments. Updates the decision objects with
        either results or error messages from the execution.

        Args:
            decisions: List of ToolCallDecision objects containing tool
                      selections and their arguments.
            tools: Dictionary mapping tool names to Tool instances that
                  will be executed.

        Returns:
            The same list of decisions, updated with execution results or
            error messages in case of failures.

        Example:
            ```python
            # Execute multiple tool decisions
            updated_decisions = selector.execute_tool_decisions(
                decisions=[
                    ToolCallDecision(
                        tool_name="calculator",
                        arguments={"x": 5, "y": 3},
                        confidence=0.9,
                        reasoning="Need to add numbers"
                    ),
                    ToolCallDecision(
                        tool_name="formatter",
                        arguments={"text": "hello"},
                        confidence=0.8,
                        reasoning="Need to format text"
                    )
                ],
                tools={
                    "calculator": calculator_tool,
                    "formatter": formatter_tool
                }
            )

            # Process results
            for decision in updated_decisions:
                if decision.error:
                    print(
                        f"Error in {decision.tool_name}: {decision.error}"
                    )
                else:
                    print(
                        f"Result from {decision.tool_name}: {decision.result}"
                    )
            ```

        Notes:
            - Each decision is executed independently
            - Execution errors in one decision don't prevent others
              from executing
            - Original decision objects are modified with results/errors
            - All execution attempts are logged for debugging
            - Tools are executed with their provided arguments
              without modification
            - Results can be any type that the tools return
            - Errors are captured as strings in the decision object

        Raises:
            KeyError: If a tool name in decisions isn't found in the tools
                      dictionary
            Exception: Any exception from tool execution is caught and stored
                      in the decision's error field
        """
        for decision in decisions:
            logger.debug(f"Executing tool '{decision.tool_name}'")
            logger.debug(f"Arguments: {decision.arguments}")

            try:
                tool = tools[decision.tool_name]
                result = tool(**decision.arguments)
                decision.result = result
                logger.debug(f"Tool execution successful: {result}")
            except Exception as e:  # pragma: no cover
                logger.error(
                    f"Failed to execute tool {decision.tool_name}: {e}",
                    exc_info=True,
                )
                decision.error = str(e)

        return decisions
