from clientai.exceptions import ClientAIError


class AgentError(ClientAIError):
    """Base exception class for agent-related errors.

    This exception is raised when there are general errors in agent operations,
    such as initialization failures or invalid configurations.
    """

    pass


class StepError(AgentError):
    """Exception raised for errors in step execution.

    This exception is raised when there are errors in executing workflow steps,
    such as invalid step configurations or execution failures.
    """

    pass


class WorkflowError(AgentError):
    """Exception raised for errors in workflow execution.

    This exception is raised when there are errors in managing
    or executing the overall workflow, such as invalid step
    sequences or workflow state issues.
    """

    pass


class ToolError(AgentError):
    """Exception raised for errors in tool execution.

    This exception is raised when there are errors in tool operations,
    such as invalid tool configurations, registration failures, or
    execution errors.
    """

    pass
