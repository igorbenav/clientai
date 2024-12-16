import os
import platform as plt
import subprocess
from enum import Enum
from typing import Dict, List

from .config import OllamaServerConfig


class Platform(Enum):
    """
    Enumeration of supported operating systems.

    This enumeration defines the operating system platforms
    that are officially supported by the Ollama server.

    Attributes:
        LINUX: Linux-based operating systems
        MACOS: macOS operating systems
        WINDOWS: Windows operating systems
    """

    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"


class GPUVendor(Enum):
    """
    Enumeration of supported GPU vendors.

    This enumeration defines the GPU vendors and architectures
    that are supported for hardware acceleration.

    Attributes:
        NVIDIA: NVIDIA GPUs with CUDA support
        AMD: AMD GPUs with ROCm support
        APPLE: Apple Silicon GPUs
        NONE: No GPU or unsupported GPU
    """

    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    NONE = "none"


class PlatformInfo:
    """
    Detects and provides information about
    the system platform and capabilities.

    This class handles platform detection, GPU capabilities, and generates
    appropriate environment configurations for the Ollama server.

    Attributes:
        _platform: The detected operating system platform.
        _gpu_vendor: The detected GPU vendor and capabilities.
        _cpu_count: The number of available CPU cores/threads.

    Example:
        Basic usage:
        ```python
        platform_info = PlatformInfo()
        print(platform_info.platform)  # Current OS platform
        print(platform_info.gpu_vendor)  # Available GPU vendor
        print(platform_info.cpu_count)  # Available CPU cores
        ```

        Get environment configuration:
        ```python
        config = OllamaServerConfig(gpu_layers=35)
        env = platform_info.get_environment(config)
        ```

        Get server command:
        ```python
        config = OllamaServerConfig(host="0.0.0.0", port=8000)
        cmd = platform_info.get_server_command(config)
        ```
    """

    def __init__(self) -> None:
        """
        Initialize platform information by detecting system capabilities.
        """
        self._platform: Platform = self._detect_platform()
        self._gpu_vendor: GPUVendor = self._detect_gpu()
        self._cpu_count: int = self._detect_cpu_count()

    def _detect_platform(self) -> Platform:
        """
        Detect the current operating system platform.

        Returns:
            Platform: The detected platform enumeration value.

        Raises:
            RuntimeError: If the current platform is not supported.

        Note:
            This is an internal method used during initialization
            to determine the operating system.
        """
        system = plt.system().lower()
        if system == "linux":
            return Platform.LINUX
        elif system == "darwin":
            return Platform.MACOS
        elif system == "windows":
            return Platform.WINDOWS
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")

    def _detect_gpu(self) -> GPUVendor:
        """
        Detect available GPU vendor and capabilities.

        Returns:
            GPUVendor: The detected GPU vendor enumeration value.

        Note:
            This is an internal method that checks for:
            - Apple Silicon on macOS
            - NVIDIA GPUs using nvidia-smi
            - AMD GPUs using rocminfo on Linux
        """
        if self._platform == Platform.MACOS and plt.processor() == "arm":
            return GPUVendor.APPLE

        try:
            nvidia_smi = (
                "nvidia-smi.exe"
                if self._platform == Platform.WINDOWS
                else "nvidia-smi"
            )
            subprocess.run(
                [nvidia_smi],
                capture_output=True,
                check=True,
            )
            return GPUVendor.NVIDIA
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        if self._platform == Platform.LINUX:
            try:
                subprocess.run(
                    ["rocminfo"],
                    capture_output=True,
                    check=True,
                )
                return GPUVendor.AMD
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        return GPUVendor.NONE

    def _detect_cpu_count(self) -> int:
        """
        Detect the number of CPU cores/threads available.

        Returns:
            int: The number of available CPU cores/threads,
                 defaults to 1 if detection fails.

        Note:
            This is an internal method used during initialization
            to determine CPU resources.
        """
        return os.cpu_count() or 1

    @property
    def platform(self) -> Platform:
        """
        Get the current operating system platform.

        Returns:
            Platform: The detected platform enumeration value.

        Example:
            ```python
            platform_info = PlatformInfo()
            if platform_info.platform == Platform.LINUX:
                print("Running on Linux")
            ```
        """
        return self._platform

    @property
    def gpu_vendor(self) -> GPUVendor:
        """
        Get the detected GPU vendor.

        Returns:
            GPUVendor: The detected GPU vendor enumeration value.

        Example:
            ```python
            platform_info = PlatformInfo()
            if platform_info.gpu_vendor == GPUVendor.NVIDIA:
                print("NVIDIA GPU detected")
            ```
        """
        return self._gpu_vendor

    @property
    def cpu_count(self) -> int:
        """
        Get the number of CPU cores/threads available.

        Returns:
            int: The number of available CPU cores/threads.

        Example:
            ```python
            platform_info = PlatformInfo()
            print(f"Available CPU cores: {platform_info.cpu_count}")
            ```
        """
        return self._cpu_count

    def get_environment(self, config: OllamaServerConfig) -> Dict[str, str]:
        """
        Get platform-specific environment variables based on configuration.

        This method generates environment variables based on the detected
        platform capabilities and the provided configuration.

        Args:
            config: The server configuration to use.

        Returns:
            Dict[str, str]: Dictionary of environment variables.

        Example:
            Configure for NVIDIA GPU:
            ```python
            config = OllamaServerConfig(
                gpu_layers=35,
                gpu_memory_fraction=0.8
            )
            env = platform_info.get_environment(config)
            ```

            Configure CPU threads:
            ```python
            config = OllamaServerConfig(cpu_threads=8)
            env = platform_info.get_environment(config)
            ```
        """
        env = os.environ.copy()

        if self.gpu_vendor == GPUVendor.NVIDIA:
            if config.gpu_layers is not None:
                env["OLLAMA_GPU_LAYERS"] = str(config.gpu_layers)
            if config.gpu_devices is not None:
                devices = (
                    config.gpu_devices
                    if isinstance(config.gpu_devices, list)
                    else [config.gpu_devices]
                )
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
            if config.gpu_memory_fraction is not None:
                env["CUDA_MEM_FRACTION"] = str(config.gpu_memory_fraction)

        elif self.gpu_vendor == GPUVendor.AMD:
            if config.gpu_devices is not None:
                devices = (
                    config.gpu_devices
                    if isinstance(config.gpu_devices, list)
                    else [config.gpu_devices]
                )
                env["GPU_DEVICE_ORDINAL"] = ",".join(map(str, devices))
            if config.gpu_memory_fraction is not None:
                env["GPU_MAX_HEAP_SIZE"] = (
                    f"{int(config.gpu_memory_fraction * 100)}%"
                )

        elif self.gpu_vendor == GPUVendor.APPLE:
            if config.gpu_layers is not None:
                env["OLLAMA_GPU_LAYERS"] = str(config.gpu_layers)

        if config.cpu_threads is not None:
            if self.platform == Platform.WINDOWS:
                env["NUMBER_OF_PROCESSORS"] = str(config.cpu_threads)
            else:
                env["GOMAXPROCS"] = str(config.cpu_threads)

        if config.memory_limit and self.platform != Platform.WINDOWS:
            env["GOMEMLIMIT"] = config.memory_limit

        if config.env_vars:
            env.update(config.env_vars)

        return env

    def get_server_command(self, config: OllamaServerConfig) -> List[str]:
        """
        Get the platform-specific command to start the server.

        This method generates the appropriate command-line arguments
        for starting the Ollama server on the current platform.

        Args:
            config: The server configuration to use.

        Returns:
            List[str]: Command list suitable for subprocess execution.

        Example:
            Default configuration:
            ```python
            cmd = platform_info.get_server_command(OllamaServerConfig())
            >> ["ollama", "serve"]
            ```

            Custom host and port:
            ```python
            config = OllamaServerConfig(
                host="0.0.0.0",
                port=8000
            )
            cmd = platform_info.get_server_command(config)
            >> ["ollama", "serve", "--host", "0.0.0.0", "--port", "8000"]
            ```
        """
        base_cmd = (
            ["ollama.exe"] if self.platform == Platform.WINDOWS else ["ollama"]
        )
        cmd = base_cmd + ["serve"]

        if config.host != "127.0.0.1":
            cmd.extend(["--host", config.host])
        if config.port != 11434:
            cmd.extend(["--port", str(config.port)])
        if config.extra_args:
            cmd.extend(config.extra_args)

        return cmd
