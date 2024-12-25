from ..exceptions import AgentError


class ValidationError(AgentError):
    """Base exception for validation errors."""

    pass


class SchemaValidationError(ValidationError):
    """Exception raised when schema validation fails."""

    pass
