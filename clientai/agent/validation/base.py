import json
import logging
from inspect import isclass
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
)

from pydantic import (
    BaseModel,
    TypeAdapter,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

from .exceptions import SchemaValidationError
from .result import ValidationResult
from .types import ValidatorContext

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelValidator(Generic[T]):
    """Validator for Pydantic model outputs."""

    def __init__(self, model_type: Type[T]) -> None:
        """Initialize validator with a model type.

        Args:
            model_type: The Pydantic model type to validate against
        """
        self.model_type = model_type
        self.type_adapter = TypeAdapter(model_type)

    def validate(
        self,
        data: Union[str, Iterator[str]],
        partial: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult[T]:
        """Validate data against the model.

        Args:
            data: String data to validate (JSON)
            partial: Whether to allow partial validation
            context: Optional validation context

        Returns:
            ValidationResult containing the validated data

        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            if isinstance(data, str):
                data = data.strip()
                if data.startswith("```json"):
                    code_block_end = data.rfind("```")
                    if code_block_end != -1:
                        data = data[7:code_block_end].strip()
                elif data.startswith("```"):
                    code_block_end = data.rfind("```")
                    if code_block_end != -1:
                        data = data[3:code_block_end].strip()

                try:
                    parsed_data = json.loads(data)
                except json.JSONDecodeError as e:
                    if partial:
                        return ValidationResult(
                            data=data,  # type: ignore
                            is_partial=True,
                            errors={"json_decode": str(e)},
                        )
                    raise SchemaValidationError(f"Invalid JSON: {str(e)}")
            else:
                parsed_data = data

            validated = self.type_adapter.validate_python(
                parsed_data,
                strict=False,
                from_attributes=True,
                context={"partial": partial, **(context or {})},
            )

            return ValidationResult(
                data=validated,
                is_partial=partial,
                metadata={"model": self.model_type.__name__},
            )

        except PydanticValidationError as e:
            if partial:
                error_dict = {}
                for error in e.errors():
                    location = ".".join(
                        str(loc) for loc in error.get("loc", [])
                    )
                    error_dict[location] = {
                        "msg": error.get("msg", ""),
                        "type": error.get("type", ""),
                    }
                return ValidationResult(
                    data=data,  # type: ignore
                    is_partial=True,
                    errors=error_dict,
                )
            raise SchemaValidationError(str(e))
        except SchemaValidationError:  # pragma: no cover
            raise
        except Exception as e:  # pragma: no cover
            raise SchemaValidationError(
                f"Unexpected validation error: {str(e)}"
            )


class StepValidator(Generic[T]):
    """Main validator for step outputs."""

    def __init__(self, return_type: Type[T]) -> None:
        """Initialize with a return type.

        Args:
            return_type: The type to validate against
        """
        self.return_type = return_type
        self.validator = self._create_validator()

    def _create_validator(self) -> Optional[ModelValidator[T]]:
        """Create appropriate validator for the return type."""
        if self._should_validate():
            return ModelValidator(self.return_type)
        return None

    def _should_validate(self) -> bool:
        """Determine if validation should be performed."""
        if not self.return_type or isinstance(self.return_type, str):
            return False

        if isclass(self.return_type) and issubclass(
            self.return_type, BaseModel
        ):
            return True

        args = get_args(self.return_type)
        return any(isclass(arg) and issubclass(arg, BaseModel) for arg in args)

    def validate(
        self, data: Union[str, Iterator[str]], context: ValidatorContext[T]
    ) -> ValidationResult[T]:
        """Validate step output data.

        Args:
            data: The data to validate
            context: Validation context

        Returns:
            ValidationResult containing the validated data
        """
        if not self.validator:
            return ValidationResult(
                data=data,  # type: ignore
                is_partial=context.partial,
            )

        return self.validator.validate(
            data, partial=context.partial, context=context.metadata
        )

    @classmethod
    def from_step(cls, step: Any) -> Optional["StepValidator[Any]"]:
        """Create a validator from a step instance.

        Args:
            step: The step to create validator for

        Returns:
            StepValidator if validation is needed, None otherwise
        """
        if not getattr(step, "json_output", False):
            return None

        return_type = getattr(step, "return_type", None)
        if return_type:
            return cls(return_type)

        try:
            hints = get_type_hints(step.func)
            return_type = hints.get("return")
            if return_type:
                return cls(return_type)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to create validator for {step.name}: {e}")

        return None
