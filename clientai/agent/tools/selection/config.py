from dataclasses import dataclass, field
from typing import Any, Dict, Optional

DEFAULT_PROMPT_TEMPLATE = """
    Given the current task and available tools,
    determine if and how to use tools to help accomplish the task.

    Task: {task}
    Current Context: {context}

    Available Tools:
    {tool_descriptions}

    Analyze the task and determine which tools would be helpful.
    Respond in JSON format with this structure:
    {{
        "tool_calls": [
            {{
                "tool_name": "name of the tool to use",
                "arguments": {{
                    "param_name": "param_value"
                }},
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation of why this tool is useful"
            }}
        ]
    }}
"""


@dataclass(frozen=True)
class ToolSelectionConfig:
    """Configuration settings for automatic tool selection behavior.

    This class defines parameters that control how tools are automatically
    selected and used during workflow execution. It includes settings for
    confidence thresholds, tool limits, and prompt customization.

    Attributes:
        confidence_threshold: Minimum confidence level (0.0-1.0) required
                              for a tool to be selected. Higher values mean
                              more selective tool usage.
        max_tools_per_step: Maximum number of tools that can be used in a
                            single workflow step. Prevents excessive tool
                            usage.
        prompt_template: Template string for generating tool selection prompts.
            Uses {task}, {context}, and {tool_descriptions} placeholders.

    Example:
        ```python
        # Basic configuration
        config = ToolSelectionConfig(
            confidence_threshold=0.8,
            max_tools_per_step=2
        )

        # Custom configuration with template
        config = ToolSelectionConfig(
            confidence_threshold=0.9,
            max_tools_per_step=5,
            prompt_template="Custom prompt: {task}\nTools: {tool_descriptions}"
        )
        ```

    Raises:
        ValueError: If configuration values are invalid (e.g., confidence
                    threshold outside 0-1 range).

    Notes:
        - The confidence threshold should be tuned based on your use case
        - The prompt template must maintain the required response format
        - Tool descriptions are automatically formatted in the prompt
    """

    confidence_threshold: float = field(default=0.7)
    max_tools_per_step: int = field(default=3)
    prompt_template: str = field(default=DEFAULT_PROMPT_TEMPLATE)

    @classmethod
    def create(cls, **kwargs: Any) -> "ToolSelectionConfig":
        """Create a ToolSelectionConfig instance from keyword arguments.

        Args:
            **kwargs: Configuration parameters (confidence_threshold,
                max_tools_per_step, prompt_template).

        Returns:
            ToolSelectionConfig: A new configuration instance.

        Example:
            ```python
            config = ToolSelectionConfig.create(
                confidence_threshold=0.8,
                max_tools_per_step=3
            )
            ```
        """
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "confidence_threshold",
                "max_tools_per_step",
                "prompt_template",
            ]
        }
        return cls(**filtered_kwargs)

    def __post_init__(self) -> None:  # pragma: no cover
        """Validate configuration values after initialization."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0"
            )
        if self.max_tools_per_step < 1:
            raise ValueError("max_tools_per_step must be at least 1")
        if not isinstance(self.prompt_template, str):
            raise ValueError("prompt_template must be a string")


@dataclass
class ToolCallDecision:
    """Represents a decision about calling a specific tool,
    including execution results.

    This class tracks both the decision to use a tool (including confidence and
    reasoning) and the results of executing that decision. It provides a
    complete record of a tool's selection and usage.

    Attributes:
        tool_name: Name of the selected tool.
        arguments: Dictionary of arguments to pass to the tool.
        confidence: Confidence level (0.0-1.0) in this tool selection.
        reasoning: Explanation of why this tool was selected.
        error: Error message if tool execution failed, None otherwise.
        result: Result from tool execution if successful, None otherwise.

    Example:
        ```python
        # Create a tool decision
        decision = ToolCallDecision(
            tool_name="calculator",
            arguments={"x": 5, "y": 3},
            confidence=0.95,
            reasoning="Need to add two numbers precisely"
        )

        # After execution, results are added
        decision.result = 8  # Successful execution
        # Or in case of error
        decision.error = "Invalid arguments"  # Failed execution
        ```

    Notes:
        - Error and result are mutually exclusive - only one should be set
        - Confidence should match the ToolSelectionConfig requirements
        - Arguments should match the tool's expected signature
    """

    tool_name: str
    arguments: Dict[str, Any]
    confidence: float
    reasoning: str
    error: Optional[str] = None
    result: Optional[Any] = None

    def __post_init__(self) -> None:  # pragma: no cover
        """Validate decision values after initialization."""
        if not isinstance(self.tool_name, str):
            raise ValueError("tool_name must be a string")
        if not isinstance(self.arguments, dict):
            raise ValueError("arguments must be a dictionary")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not isinstance(self.reasoning, str):
            raise ValueError("reasoning must be a string")
