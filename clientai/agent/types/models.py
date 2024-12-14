from typing import Any, Dict, TypedDict, Union


class ModelParameters(TypedDict, total=False):
    """Configuration parameters for language model execution.

    A TypedDict class defining the standard configuration parameters
    accepted by language models in the system. All fields are optional.

    Attributes:
        name: The name/identifier of the model.
        temperature: Sampling temperature between 0 and 1.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling parameter between 0 and 1.
        frequency_penalty: Penalty for token frequency between -2.0 and 2.0.
        presence_penalty: Penalty for token presence between -2.0 and 2.0.
        stop: Stop sequences for generation, either a single string or list.
        stream: Whether to stream responses or return complete.
        extra: Additional model-specific parameters.

    Example:
        ```python
        params: ModelParameters = {
            "name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True
        }
        ```

    Notes:
        - All fields are optional (total=False)
        - temperature and top_p affect randomness of outputs
        - frequency_penalty and presence_penalty affect repetition
        - stop can be either a single string or list of strings
        - extra allows for model-specific parameters not in standard config
    """

    name: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: Union[str, list[str]]
    stream: bool
    extra: Dict[str, Any]
