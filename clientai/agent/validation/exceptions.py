from ..exceptions import AgentError


class ValidationError(AgentError):
    """Base exception for validation errors."""

    pass


class PartialValidationError(ValidationError):
    """Exception raised when partial validation fails during streaming."""

    pass


class SchemaValidationError(ValidationError):
    """Exception raised when schema validation fails."""

    pass
