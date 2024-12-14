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
"""Default configurations for different step types.

A mapping of step types to their default configuration parameters.
Each step type has optimized settings for its specific purpose:

- think: Higher temperature for creative analysis
- act: Lower temperature for decisive actions
- observe: Lowest temperature for accurate observations
- synthesize: Moderate temperature for balanced summarization

Example:
    Apply default configuration:
    ```python
    step_type = "think"
    config = DEFAULT_STEP_CONFIGS[step_type]
    print(config["temperature"])  # Output: 0.7
    print(config["top_p"])  # Output: 0.9
    ```

Notes:
    - Temperature controls randomness (0.0-1.0)
    - Top_p controls nucleus sampling (0.0-1.0)
    - Each type has settings optimized for its purpose
    - These defaults can be overridden in step configuration
"""

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
"""Default parameters for each StepType enum value.

Similar to DEFAULT_STEP_CONFIGS but keyed by StepType enum values instead
of strings. Used internally when working directly with StepType enums.

Example:
    Access defaults by step type:
    ```python
    from clientai.agent.steps.types import StepType

    config = STEP_TYPE_DEFAULTS[StepType.THINK]
    print(config["temperature"])  # Output: 0.7
    print(config["top_p"])  # Output: 0.9
    ```

Notes:
    - Matches DEFAULT_STEP_CONFIGS values
    - Used with StepType enum values
    - Provides type-safe access to defaults
    - Does not include step_type field (redundant with key)
"""

__all__ = ["DEFAULT_STEP_CONFIGS", "STEP_TYPE_DEFAULTS"]
