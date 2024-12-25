from dataclasses import dataclass
from typing import Optional


@dataclass
class StepConfig:
    """Configuration settings for workflow step execution behavior.

    Controls how individual workflow steps are executed,
    including retry behavior, error handling, and result passing.

    Attributes:
        enabled: Whether the step is active and should be executed
        retry_count: Number of times to retry failed step execution
        timeout: Optional timeout in seconds for step execution
        required: Whether step failure should halt workflow execution
        pass_result: Whether step result should be passed to next step
        use_internal_retry: Whether to use internal retry mechanism

    Example:
        Create step configuration:
        ```python
        # Basic configuration
        config = StepConfig(
            enabled=True,
            retry_count=3,
            timeout=30.0,
            required=True
        )

        # Configuration for optional step
        optional_config = StepConfig(
            enabled=True,
            required=False,
            pass_result=False
        )

        # Configuration with retries disabled
        no_retry_config = StepConfig(
            retry_count=0,
            use_internal_retry=False
        )
        ```
    """

    enabled: bool = True
    retry_count: int = 0
    timeout: Optional[float] = None
    required: bool = True
    pass_result: bool = True
    use_internal_retry: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> "StepConfig":  # pragma: no cover
        """Create step configuration from dictionary.

        Args:
            config: Dictionary containing configuration parameters.
                   Only recognized parameters are used.

        Returns:
            StepConfig: New configuration instance
                        with parameters from dictionary.

        Example:
            Create from dictionary:
            ```python
            config = StepConfig.from_dict({
                "enabled": True,
                "retry_count": 3,
                "timeout": 30.0,
                "required": True,
                "unknown_param": "ignored"  # This will be ignored
            })
            ```
        """
        return cls(
            **{
                k: v
                for k, v in config.items()
                if k in cls.__dataclass_fields__
            }
        )
