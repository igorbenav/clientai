from typing import Any, Dict

from ..steps.types import StepType

DEFAULT_STEP_CONFIGS: Dict[str, Dict[str, Any]] = {
    "think": {
        "temperature": 0.7,
        "top_p": 0.9,
        "step_type": StepType.THINK,
    },
    "act": {
        "temperature": 0.2,
        "top_p": 0.8,
        "step_type": StepType.ACT,
    },
    "observe": {
        "temperature": 0.1,
        "top_p": 0.5,
        "step_type": StepType.OBSERVE,
    },
    "synthesize": {
        "temperature": 0.4,
        "top_p": 0.7,
        "step_type": StepType.SYNTHESIZE,
    },
}

STEP_TYPE_DEFAULTS = {
    StepType.THINK: {
        "temperature": 0.7,
        "top_p": 0.9,
    },
    StepType.ACT: {
        "temperature": 0.2,
        "top_p": 0.8,
    },
    StepType.OBSERVE: {
        "temperature": 0.1,
        "top_p": 0.5,
    },
    StepType.SYNTHESIZE: {
        "temperature": 0.4,
        "top_p": 0.7,
    },
}

__all__ = ["DEFAULT_STEP_CONFIGS", "STEP_TYPE_DEFAULTS"]
