class AgentError(Exception):
    """Base exception class for agent-related errors.

    This exception serves as the base class for agent-specific errors
    in the utils module. All other agent utility exceptions inherit from this.
    """


class StepError(AgentError):
    """Exception raised for errors in step validation and execution.

    This exception is raised during step validation or execution in utility
    functions, such as when validating step signatures or handling step data.
    """


class ToolError(AgentError):
    """Exception raised for errors in tool validation and execution.

    This exception is raised during tool validation or execution in utility
    functions, such as when validating tool signatures or processing tool data.
    """
