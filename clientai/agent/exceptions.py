from clientai.exceptions import ClientAIError


class AgentError(ClientAIError):
    """Base exception for agent-related errors."""

    pass


class StepError(AgentError):
    """Raised when there's an error in step execution."""

    pass


class WorkflowError(AgentError):
    """Raised when there's an error in workflow execution."""

    pass


class ToolError(AgentError):
    """Raised when there's an error in tool execution."""

    pass
