from typing import Any, ClassVar, Dict


class ModelConfig:
    CORE_ATTRS: ClassVar[set] = {
        "name",
        "return_full_response",
        "stream",
        "json_output",
    }

    def __init__(
        self,
        name: str,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        **kwargs: Any,
    ):
        self.name = name
        self.return_full_response = return_full_response
        self.stream = stream
        self.json_output = json_output
        self._extra_kwargs = kwargs

    def get_parameters(self) -> Dict[str, Any]:
        params = {
            "return_full_response": self.return_full_response,
            "stream": self.stream,
            "json_output": self.json_output,
        }
        params.update(self._extra_kwargs)
        return {k: v for k, v in params.items() if v is not None}

    def merge(self, **kwargs: Any) -> "ModelConfig":
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
            **{**self._extra_kwargs, **extra_kwargs},
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
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
        return {
            "name": self.name,
            "return_full_response": self.return_full_response,
            "stream": self.stream,
            "json_output": self.json_output,
            **self._extra_kwargs,
        }
