from dataclasses import dataclass
from typing import Optional


@dataclass
class StepConfig:
    enabled: bool = True
    retry_count: int = 0
    timeout: Optional[float] = None
    required: bool = True
    pass_result: bool = True
    use_internal_retry: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> "StepConfig":
        return cls(
            **{
                k: v
                for k, v in config.items()
                if k in cls.__dataclass_fields__
            }
        )
