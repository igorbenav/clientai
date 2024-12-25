from typing import Any, ClassVar, Dict, Optional


class ModelConfig:
    """Configuration class for language model parameters and settings.

    Manages both core model parameters (name, streaming, etc.)
    and additional model-specific parameters. Provides methods
    for parameter merging and serialization.

    Attributes:
        CORE_ATTRS: Class-level set of core attribute names
                    that are handled specially
        name: Name of the language model
        return_full_response: Whether to return complete API response
        stream: Whether to enable response streaming
        json_output: Whether responses should be formatted as JSON
        temperature: Optional temperature value (0.0-2.0) controlling
                     response randomness

    Example:
        Create and use model configuration:
        ```python
        # Basic configuration
        config = ModelConfig(
            name="gpt-4",
            temperature=0.7,
            stream=True
        )

        # Get parameters for API call
        params = config.get_parameters()
        print(params)  # Output: {"temperature": 0.7, "stream": True}

        # Merge with new parameters
        new_config = config.merge(temperature=0.5, top_p=0.9)
        ```
    """

    CORE_ATTRS: ClassVar[set] = {
        "name",
        "return_full_response",
        "stream",
        "json_output",
        "temperature",
    }

    def __init__(
        self,
        name: str,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ):
        if (
            temperature is not None and not 0.0 <= temperature <= 2.0
        ):  # pragma: no cover
            raise ValueError("Temperature must be between 0.0 and 2.0")

        self.name = name
        self.return_full_response = return_full_response
        self.stream = stream
        self.json_output = json_output
        self.temperature = temperature
        self._extra_kwargs = kwargs

    def get_parameters(self) -> Dict[str, Any]:
        """Get all non-None parameters as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of parameter names to values.

        Example:
            Get parameters for API call:
            ```python
            config = ModelConfig(
                name="gpt-4",
                temperature=0.7,
                top_p=None
            )
            params = config.get_parameters()
            print(params)  # Output: {"temperature": 0.7}
            ```
        """
        params: Dict[str, Any] = {
            "return_full_response": self.return_full_response,
            "stream": self.stream,
            "json_output": self.json_output,
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature

        params.update(self._extra_kwargs)
        return {k: v for k, v in params.items() if v is not None}

    def merge(self, **kwargs: Any) -> "ModelConfig":
        """Create new configuration by merging current with new parameters.

        Args:
            **kwargs: New parameter values to merge with existing ones.

        Returns:
            ModelConfig: New configuration instance with merged parameters.

        Example:
            Merge configurations:
            ```python
            base_config = ModelConfig(name="gpt-4", temperature=0.7)
            new_config = base_config.merge(
                temperature=0.5,
                top_p=0.9,
                stream=True
            )
            ```
        """
        core_kwargs = {k: kwargs[k] for k in self.CORE_ATTRS if k in kwargs}
        extra_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.CORE_ATTRS
        }

        return ModelConfig(
            name=core_kwargs.get("name", self.name),
            return_full_response=core_kwargs.get(
                "return_full_response", self.return_full_response
            ),
            stream=core_kwargs.get("stream", self.stream),
            json_output=core_kwargs.get("json_output", self.json_output),
            temperature=core_kwargs.get("temperature", self.temperature),
            **{**self._extra_kwargs, **extra_kwargs},
        )

    @classmethod
    def from_dict(
        cls, config: Dict[str, Any]
    ) -> "ModelConfig":  # pragma: no cover
        """Create configuration instance from a dictionary.

        Args:
            config: Dictionary containing configuration parameters.
                Must include 'name' key.

        Returns:
            ModelConfig: New configuration instance.

        Raises:
            ValueError: If 'name' is missing from config dictionary.

        Example:
            Create from dictionary:
            ```python
            config = ModelConfig.from_dict({
                "name": "gpt-4",
                "temperature": 0.7,
                "stream": True
            })
            ```
        """
        if "name" not in config:
            raise ValueError(
                "Model name is required in configuration dictionary"
            )

        core_params = {k: config[k] for k in cls.CORE_ATTRS if k in config}
        extra_params = {
            k: v for k, v in config.items() if k not in cls.CORE_ATTRS
        }

        return cls(**core_params, **extra_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary containing all configuration parameters.

        Example:
            Convert to dictionary:
            ```python
            config = ModelConfig(name="gpt-4", temperature=0.7)
            data = config.to_dict()
            print(data)  # Output: {"name": "gpt-4", "temperature": 0.7, ...}
            ```
        """
        return {
            "name": self.name,
            "return_full_response": self.return_full_response,
            "stream": self.stream,
            "json_output": self.json_output,
            "temperature": self.temperature,
            **self._extra_kwargs,
        }
