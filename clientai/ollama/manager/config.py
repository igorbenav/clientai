import ipaddress
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class OllamaServerConfig:
    """
    Configuration settings for Ollama server.

    Attributes:
        host: Hostname to bind the server to
        port: Port number to listen on
        timeout: Maximum time in seconds to wait for server startup
        check_interval: Time in seconds between server readiness checks
        gpu_layers: Number of layers to run on GPU
        compute_unit: Compute device to use ('cpu', 'gpu', 'auto')
        cpu_threads: Number of CPU threads to use
        memory_limit: Memory limit for the server
                      (format: number + GiB/MiB, e.g., "8GiB")
        gpu_memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        gpu_devices: GPU device IDs to use
        env_vars: Additional environment variables
        extra_args: Additional command line arguments
    """

    host: str = "127.0.0.1"
    port: int = 11434
    timeout: int = 30
    check_interval: float = 1.0
    gpu_layers: Optional[int] = None
    compute_unit: Optional[str] = None
    cpu_threads: Optional[int] = None
    memory_limit: Optional[str] = None
    gpu_memory_fraction: Optional[float] = None
    gpu_devices: Optional[Union[List[int], int]] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)

    def _validate_host(self) -> None:
        """Validate the host address."""
        if not self.host:
            raise ValueError("Host cannot be empty")

        try:
            ipaddress.ip_address(self.host)
        except ValueError:
            if not re.match(r"^[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*$", self.host):
                raise ValueError(f"Invalid host: {self.host}")

    def _validate_port(self) -> None:
        """Validate the port number."""
        if not 1 <= self.port <= 65535:
            raise ValueError(
                f"Port must be between 1 and 65535, got {self.port}"
            )

    def _validate_timeout_and_interval(self) -> None:
        """Validate timeout and check interval."""
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.check_interval <= 0:
            raise ValueError("Check interval must be positive")
        if self.check_interval > self.timeout:
            raise ValueError("Check interval cannot be greater than timeout")

    def _validate_gpu_settings(self) -> None:
        """Validate GPU-related settings."""
        if self.gpu_layers is not None:
            if not isinstance(self.gpu_layers, int) or self.gpu_layers < 0:
                raise ValueError("gpu_layers must be a non-negative integer")

        if self.gpu_memory_fraction is not None:
            if not 0.0 <= self.gpu_memory_fraction <= 1.0:
                raise ValueError(
                    "gpu_memory_fraction must be between 0.0 and 1.0"
                )

        if self.gpu_devices is not None:
            if isinstance(self.gpu_devices, int):
                if self.gpu_devices < 0:
                    raise ValueError("GPU device ID must be non-negative")
            elif isinstance(self.gpu_devices, list):
                if not all(
                    isinstance(d, int) and d >= 0 for d in self.gpu_devices
                ):
                    raise ValueError(
                        "All GPU device IDs must be non-negative integers"
                    )
                if len(self.gpu_devices) != len(set(self.gpu_devices)):
                    raise ValueError(
                        "Duplicate GPU device IDs are not allowed"
                    )
            else:
                raise ValueError(
                    "gpu_devices must be an integer or list of integers"
                )

    def _validate_compute_unit(self) -> None:
        """Validate compute unit setting."""
        if self.compute_unit and self.compute_unit not in [
            "cpu",
            "gpu",
            "auto",
        ]:
            raise ValueError(
                "compute_unit must be one of: 'cpu', 'gpu', 'auto'"
            )

    def _validate_cpu_threads(self) -> None:
        """Validate CPU threads setting."""
        if self.cpu_threads is not None:
            if not isinstance(self.cpu_threads, int) or self.cpu_threads <= 0:
                raise ValueError("cpu_threads must be a positive integer")

    def _validate_memory_limit(self) -> None:
        """Validate memory limit format."""
        if self.memory_limit is not None:
            pattern = r"^\d+(\.\d+)?[MGT]iB$"
            if not re.match(pattern, self.memory_limit):
                raise ValueError(
                    "memory_limit must be in format: NUMBER + UNIT, "
                    "where UNIT is MiB, GiB, or TiB (e.g., '8GiB')"
                )

            match = re.match(r"^\d+(\.\d+)?", self.memory_limit)
            if match is None:
                raise ValueError("Invalid memory_limit format")

            value = float(match.group())
            unit = self.memory_limit[-3:]

            if unit == "MiB" and value < 100:
                raise ValueError("memory_limit in MiB must be at least 100")
            elif unit == "GiB" and value < 0.1:
                raise ValueError("memory_limit in GiB must be at least 0.1")
            elif unit == "TiB" and value < 0.001:
                raise ValueError("memory_limit in TiB must be at least 0.001")

    def _validate_env_vars(self) -> None:
        """Validate environment variables."""
        if not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in self.env_vars.items()
        ):
            raise ValueError("All environment variables must be strings")

    def _validate_extra_args(self) -> None:
        """Validate extra arguments."""
        if not all(isinstance(arg, str) for arg in self.extra_args):
            raise ValueError("All extra arguments must be strings")

    def __post_init__(self):
        """Validate all configuration after initialization."""
        self._validate_host()
        self._validate_port()
        self._validate_timeout_and_interval()
        self._validate_gpu_settings()
        self._validate_compute_unit()
        self._validate_cpu_threads()
        self._validate_memory_limit()
        self._validate_env_vars()
        self._validate_extra_args()

    @property
    def base_url(self) -> str:
        """Get the base URL for the Ollama server."""
        return f"http://{self.host}:{self.port}"
