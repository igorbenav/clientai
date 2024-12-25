from dataclasses import dataclass, field
from typing import Any, Dict, Generic, TypeVar

T = TypeVar("T")


@dataclass
class ValidationResult(Generic[T]):
    """Result of validation operation."""

    data: T
    """The validated data."""

    is_partial: bool = False
    """Whether this is a partial validation result."""

    errors: Dict[str, Any] = field(default_factory=dict)
    """Any validation errors encountered."""

    warnings: Dict[str, Any] = field(default_factory=dict)
    """Any validation warnings generated."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the validation."""

    @property
    def is_valid(self) -> bool:
        """Whether validation was successful."""
        return not bool(self.errors)  # pragma: no cover

    @property
    def is_complete(self) -> bool:
        """Whether this is a complete validation result."""
        return not self.is_partial  # pragma: no cover

    def __bool__(self) -> bool:
        """Boolean representation of validation success."""
        return self.is_valid  # pragma: no cover
