from enum import Enum
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class OutputFormat(str, Enum):
    """Format of the output to be validated."""

    STRING = "string"
    JSON = "json"


class Validator(Protocol[T_co]):
    """Protocol defining the interface for validators."""

    def validate(self, data: str, partial: bool = False) -> T_co:
        """Validate output data.

        Args:
            data: The data to validate
            partial: Whether to allow partial validation for streaming

        Returns:
            Validated data of type T

        Raises:
            ValidationError: If validation fails
        """
        ...


class ValidatorContext(Generic[T]):
    """Context for validation operations."""

    data: Any
    format: OutputFormat
    partial: bool
    metadata: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        data: Any,
        format: OutputFormat = OutputFormat.STRING,
        partial: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data = data
        self.format = format
        self.partial = partial
        self.metadata = metadata or {}
