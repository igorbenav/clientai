import json
import logging
from typing import Any, Dict, List, Optional

from ...config.models import ModelConfig
from ...tools.base import Tool
from .config import ToolCallDecision, ToolSelectionConfig

logger = logging.getLogger(__name__)


class ToolSelector:
    """
    Manages the automatic selection and execution of tools using LLM-based decision making.

    The ToolSelector uses a language model to analyze tasks and determine which tools
    would be most appropriate to use. It handles:
    - Tool selection based on task requirements
    - Confidence-based filtering
    - Argument validation
    - Tool execution and error handling

    Key Features:
        - LLM-based tool selection
        - Configurable confidence thresholds
        - Automatic argument validation
        - Comprehensive error handling
        - Detailed execution logging

    Attributes:
        config: Configuration for tool selection behavior
        model_config: Configuration for the LLM used in selection

    Example:
        ```python
        selector = ToolSelector(
            config=ToolSelectionConfig(confidence_threshold=0.8),
            model_config=ModelConfig(name="gpt-4")
        )

        # Select tools for a task
        decisions = selector.select_tools(
            task="Add 5 and 3, then multiply by 2",
            tools=[calculator, multiplier],
            context={},
            client=llm_client
        )

        # Execute the selected tools
        results = selector.execute_tool_decisions(
            decisions=decisions,
            tools={"calculator": calculator, "multiplier": multiplier}
        )
        ```
    """

    def __init__(
        self,
        config: Optional[ToolSelectionConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        """
        Initialize the ToolSelector with the specified configurations.

        Args:
            config: Configuration for tool selection behavior. If None,
                   uses default configuration.
            model_config: Configuration for the LLM used in selection.
                        If None, uses default model.
        """
        self.config = config or ToolSelectionConfig()
        self.model_config = model_config or ModelConfig(
            name="llama-3.2-3b-preview", temperature=0.0, json_output=True
        )
        logger.debug("Initialized ToolSelector")
        logger.debug(f"Using model: {self.model_config.name}")
        logger.debug(
            f"Confidence threshold: {self.config.confidence_threshold}"
        )

    def _format_tool_descriptions(self, tools: List[Tool]) -> str:
        """
        Format a list of tools into a string description suitable for LLM prompts.

        Creates a formatted string containing each tool's signature and description,
        organized in a clear, readable format for the LLM to understand.

        Args:
            tools: List of tools to describe

        Returns:
            A formatted string containing tool descriptions

        Example:
            ```python
            descriptions = selector._format_tool_descriptions([
                calculator,
                text_processor
            ])
            print(descriptions)
            # Output:
            # - Calculator
            #   Signature: add(x: int, y: int) -> int
            #   Description: Adds two numbers
            # ...
            ```
        """
        descriptions = []
        for tool in tools:
            descriptions.extend(
                [
                    f"- {tool.name}",
                    f"  Signature: {tool.signature_str}",
                    f"  Description: {tool.description}",
                    "",
                ]
            )
        formatted = "\n".join(descriptions)
        logger.debug(f"Formatted tool descriptions:\n{formatted}")
        return formatted

    def select_tools(
        self,
        task: str,
        tools: List[Tool],
        context: Dict[str, Any],
        client: Any,
    ) -> List[ToolCallDecision]:
        """
        Use LLM to select appropriate tools for a given task.

        Analyzes the task description and available tools to determine which
        tools would be most appropriate to use. Filters selections based on
        confidence thresholds and validates tool arguments.

        Args:
            task: Description of the task to accomplish
            tools: List of available tools to choose from
            context: Current execution context and state
            client: LLM client for making selection decisions

        Returns:
            List of tool selection decisions, including confidence levels
            and reasoning

        Example:
            ```python
            decisions = selector.select_tools(
                task="Calculate the sum of 5 and 3",
                tools=[calculator, text_processor],
                context={},
                client=llm_client
            )
            for decision in decisions:
                print(f"Selected: {decision.tool_name}")
                print(f"Confidence: {decision.confidence}")
                print(f"Reasoning: {decision.reasoning}")
            ```
        """
        logger.debug(f"Selecting tools for task: {task}")
        logger.debug(f"Available tools: {[t.name for t in tools]}")

        if not tools:
            logger.debug("No tools available, returning empty list")
            return []

        prompt = f"""
            You are a helpful AI that uses tools to solve problems.

            Task: {task}

            Available Tools:
            {self._format_tool_descriptions(tools)}

            Respond ONLY with a JSON object in this format:
            {{
                "tool_calls": [
                    {{
                        "tool_name": "<name of tool>",
                        "arguments": {{
                            "x": <first number>,
                            "y": <second number>
                        }},
                        "confidence": <0.0-1.0>,
                        "reasoning": "<why you chose this tool>"
                    }}
                ]
            }}
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
            except json.JSONDecodeError:
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

        except Exception as e:
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
                    f"Skipping tool {tool_name} due to low confidence: {confidence}"
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

        Checks that all required parameters are provided and that no
        unknown parameters are included.

        Args:
            tool: The tool whose arguments to validate
            arguments: The arguments to validate

        Returns:
            True if arguments are valid, False otherwise

        Example:
            ```python
            is_valid = selector._validate_tool_arguments(
                tool=calculator,
                arguments={"x": 5, "y": 3}
            )
            if not is_valid:
                print("Invalid arguments provided")
            ```
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
                f"valid parameters are {valid_params}, got {list(arguments.keys())}"
            )
            return False

        return True

    def execute_tool_decisions(
        self, decisions: List[ToolCallDecision], tools: Dict[str, Tool]
    ) -> List[ToolCallDecision]:
        """
        Execute a series of tool decisions and capture their results.

        Takes a list of tool selection decisions and executes each tool
        with its specified arguments. Updates the decisions with results
        or error messages.

        Args:
            decisions: List of tool call decisions to execute
            tools: Dictionary mapping tool names to Tool instances

        Returns:
            The same decisions, updated with execution results or errors

        Example:
            ```python
            # Execute decisions
            results = selector.execute_tool_decisions(
                decisions=decisions,
                tools={"calculator": calculator}
            )

            # Check results
            for result in results:
                if result.error:
                    print(f"Error: {result.error}")
                else:
                    print(f"Result: {result.result}")
            ```

        Note:
            - Each decision is executed independently
            - Errors in one execution don't prevent others
            - Original decision objects are modified with results
        """
        for decision in decisions:
            logger.debug(f"Executing tool '{decision.tool_name}'")
            logger.debug(f"Arguments: {decision.arguments}")

            try:
                tool = tools[decision.tool_name]
                result = tool(**decision.arguments)
                decision.result = result
                logger.debug(f"Tool execution successful: {result}")
            except Exception as e:
                logger.error(
                    f"Failed to execute tool {decision.tool_name}: {e}",
                    exc_info=True,
                )
                decision.error = str(e)

        return decisions
