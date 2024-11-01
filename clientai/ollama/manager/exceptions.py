from typing import Optional, Type


class OllamaManagerError(Exception):
    """Base exception class for Ollama manager errors."""

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.original_error = original_error

    def __str__(self) -> str:
        """Format the error message."""
        error_msg = super().__str__()
        if self.original_error:
            error_msg = f"{error_msg}\nCaused by: {str(self.original_error)}"
        return error_msg

    @property
    def original_exception(self) -> Optional[Exception]:
        """Returns the original exception that caused this error, if any."""
        return self.original_error


class ExecutableNotFoundError(OllamaManagerError):
    """
    Raised when the Ollama executable cannot be found.

    This typically means Ollama is not installed or not in the system PATH.
    """


class ServerStartupError(OllamaManagerError):
    """
    Raised when the Ollama server fails to start.

    This can happen due to:
    - Port already in use
    - Insufficient permissions
    - Invalid configuration
    - System resource constraints
    """


class ServerShutdownError(OllamaManagerError):
    """
    Raised when there's an error stopping the Ollama server.

    This can happen when:
    - The server process cannot be terminated
    - The server is in an inconsistent state
    - The system prevents process termination
    """


class ServerTimeoutError(OllamaManagerError):
    """
    Raised when the server operation times out.

    This can happen during:
    - Server startup
    - Health checks
    - Server shutdown

    The timeout duration is configurable through OllamaServerConfig.
    """


class UnsupportedPlatformError(OllamaManagerError):
    """
    Raised when running on an unsupported platform or configuration.

    This can happen when:
    - The operating system is not supported (e.g., BSD, Solaris)
    - Required system features are missing
    - GPU configuration is incompatible
    """


class ResourceError(OllamaManagerError):
    """
    Raised when there are issues with system resources.

    This can happen due to:
    - Insufficient memory
    - GPU memory allocation failures
    - CPU thread allocation issues
    - Disk space constraints
    """


class ConfigurationError(OllamaManagerError):
    """
    Raised when there are issues with the Ollama configuration.

    This can happen when:
    - Invalid configuration values are provided
    - Incompatible settings are combined
    - Required configuration is missing
    - Platform-specific settings are invalid
    """


def raise_ollama_error(
    error_class: Type[OllamaManagerError],
    message: str,
    original_error: Optional[Exception] = None,
) -> None:
    """
    Helper function to raise Ollama manager errors with consistent formatting.

    Args:
        error_class: The specific error class to raise
        message: The error message
        original_error: The original exception that caused this error, if any

    Raises:
        OllamaManagerError: The specified error class with formatted message
    """
    raise error_class(message, original_error)
