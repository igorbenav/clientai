from typing import Optional, Type


class ClientAIError(Exception):
    """Base exception class for ClientAI errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.original_error = original_error

    def __str__(self):
        error_msg = super().__str__()
        if self.status_code:
            error_msg = f"[{self.status_code}] {error_msg}"
        return error_msg

    @property
    def original_exception(self) -> Optional[Exception]:
        """Returns the original exception object if available."""
        return self.original_error


class AuthenticationError(ClientAIError):
    """Raised when there's an authentication problem with the AI provider."""


class APIError(ClientAIError):
    """Raised when there's an API-related error from the AI provider."""


class RateLimitError(ClientAIError):
    """Raised when the AI provider's rate limit is exceeded."""


class InvalidRequestError(ClientAIError):
    """Raised when the request to the AI provider is invalid."""


class ModelError(ClientAIError):
    """Raised when there's an issue with the specified model."""


class ProviderNotInstalledError(ClientAIError):
    """Raised when the required provider package is not installed."""


class TimeoutError(ClientAIError):
    """Raised when a request to the AI provider times out."""


def map_status_code_to_exception(
    status_code: int,
) -> Type[ClientAIError]:  # pragma: no cover
    """
    Maps an HTTP status code to the appropriate ClientAI exception class.

    Args:
        status_code (int): The HTTP status code.

    Returns:
        Type[ClientAIError]: The appropriate ClientAI exception class.
    """
    if status_code == 401:
        return AuthenticationError
    elif status_code == 429:
        return RateLimitError
    elif status_code == 400:
        return InvalidRequestError
    elif status_code == 404:
        return ModelError
    elif status_code == 408:
        return TimeoutError
    elif status_code >= 500:
        return APIError
    else:
        return APIError


def raise_clientai_error(
    status_code: int, message: str, original_error: Optional[Exception] = None
) -> None:
    """
    Raises the appropriate ClientAI exception based on the status code.

    Args:
        status_code (int): The HTTP status code.
        message (str): The error message.
        original_error (Exception, optional): The original exception caught.

    Raises:
        ClientAIError: The appropriate ClientAI exception.
    """
    exception_class = map_status_code_to_exception(
        status_code,
    )
    raise exception_class(message, status_code, original_error)
