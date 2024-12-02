class AgentError(Exception):
    """Base exception for agent-related errors."""


class StepError(AgentError):
    """Raised when there's an error in step execution."""


class ToolError(AgentError):
    """Raised when there's an error in tool execution."""
