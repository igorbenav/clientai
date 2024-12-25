import http.client
import subprocess
from typing import Any, Union, cast
from unittest.mock import MagicMock, patch

import pytest

from clientai.ollama.manager import OllamaManager, OllamaServerConfig
from clientai.ollama.manager.exceptions import (
    ExecutableNotFoundError,
    ServerStartupError,
    ServerTimeoutError,
)
from clientai.ollama.manager.platform_info import (
    GPUVendor,
    Platform,
    PlatformInfo,
)


class MockProcess:
    """Mock for subprocess.Popen that supports type hints."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._poll_value = None
        self.terminate = MagicMock()
        self.wait = MagicMock()
        self.communicate = MagicMock(return_value=("", ""))

    def poll(self) -> Union[int, None]:
        """Mock poll method."""
        return self._poll_value


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.Popen for server process management."""
    with patch(
        "clientai.ollama.manager.core.subprocess", autospec=True
    ) as mock:
        process = MockProcess()
        mock.Popen = MagicMock(return_value=process)
        mock.CREATE_NO_WINDOW = 0x08000000
        mock.PIPE = subprocess.PIPE
        yield mock


@pytest.fixture
def mock_http_client():
    """Mock http.client for server health checks."""
    with patch(
        "clientai.ollama.manager.core.http.client.HTTPConnection"
    ) as mock:
        mock_conn = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_conn.getresponse.return_value = mock_response
        mock.return_value = mock_conn
        yield mock


@pytest.fixture
def mock_platform_info():
    """Mock PlatformInfo for system information."""
    with patch(
        "clientai.ollama.manager.core.PlatformInfo", autospec=True
    ) as MockPlatformInfo:
        platform_info = MagicMock(spec=PlatformInfo)
        platform_info.platform = Platform.LINUX
        platform_info.gpu_vendor = GPUVendor.NVIDIA
        platform_info.cpu_count = 8

        def get_environment(config: Any) -> dict[str, str]:
            """Dynamic environment generation based on config."""
            env = {
                "PATH": "/usr/local/bin",
            }
            if config.gpu_layers is not None:
                env["OLLAMA_GPU_LAYERS"] = str(config.gpu_layers)

            if platform_info.gpu_vendor == GPUVendor.NVIDIA:
                if config.gpu_devices is not None:
                    devices = (
                        config.gpu_devices
                        if isinstance(config.gpu_devices, list)
                        else [config.gpu_devices]
                    )
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
                if config.gpu_memory_fraction is not None:
                    env["CUDA_MEM_FRACTION"] = str(config.gpu_memory_fraction)

            elif platform_info.gpu_vendor == GPUVendor.AMD:  # pragma: no cover
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

            return env

        platform_info.get_environment.side_effect = get_environment

        def get_server_command(config: Any) -> list[str]:
            """Get platform-specific server command."""
            base_cmd = ["ollama", "serve"]
            if config.host != "127.0.0.1":  # pragma: no cover
                base_cmd.extend(["--host", config.host])
            if config.port != 11434:  # pragma: no cover
                base_cmd.extend(["--port", str(config.port)])
            return base_cmd

        platform_info.get_server_command.side_effect = get_server_command

        MockPlatformInfo.return_value = platform_info
        yield platform_info


@pytest.fixture
def manager(mock_platform_info):
    """Create a manager instance with test configuration."""
    config = OllamaServerConfig(
        host="127.0.0.1",
        port=11434,
        gpu_layers=35,
        gpu_memory_fraction=0.8,
        gpu_devices=0,
    )
    return OllamaManager(config)


def test_init_default_config():
    """Test manager initialization with default config."""
    manager = OllamaManager()
    assert isinstance(manager.config, OllamaServerConfig)
    assert manager.config.host == "127.0.0.1"
    assert manager.config.port == 11434


def test_init_custom_config():
    """Test manager initialization with custom config."""
    config = OllamaServerConfig(host="localhost", port=11435, gpu_layers=35)
    manager = OllamaManager(config)
    assert manager.config == config


def test_start_server_success(
    manager, mock_subprocess, mock_http_client, mock_platform_info
):
    """Test successful server startup."""
    manager.start()

    mock_subprocess.Popen.assert_called_once()
    assert manager._process is not None
    mock_platform_info.get_environment.assert_called_once_with(manager.config)
    mock_platform_info.get_server_command.assert_called_once_with(
        manager.config
    )


def test_start_server_executable_not_found(manager, mock_subprocess):
    """Test error handling when Ollama executable is not found."""
    mock_subprocess.Popen.side_effect = FileNotFoundError()

    with pytest.raises(ExecutableNotFoundError) as exc_info:
        manager.start()

    assert "Ollama executable not found" in str(exc_info.value)
    assert manager._process is None


def test_start_server_already_running(manager):
    """Test error when attempting to start an already running server."""
    manager._process = cast(subprocess.Popen[str], MockProcess())

    with pytest.raises(ServerStartupError) as exc_info:
        manager.start()

    assert "already running" in str(exc_info.value)


def test_stop_server_success(manager):
    """Test successful server shutdown."""
    process = MockProcess()
    manager._process = cast(subprocess.Popen[str], process)

    manager.stop()

    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    assert manager._process is None


def test_stop_server_not_running(manager):
    """Test stopping server when it's not running."""
    manager._process = None
    manager.stop()


def test_context_manager(manager, mock_subprocess, mock_http_client):
    """Test using manager as a context manager."""
    with manager as m:
        assert m._process is not None
        assert isinstance(m, OllamaManager)
    assert m._process is None


@pytest.mark.parametrize(
    "platform_type,expected_cmd",
    [
        (Platform.WINDOWS, ["ollama.exe", "serve"]),
        (Platform.LINUX, ["ollama", "serve"]),
        (Platform.MACOS, ["ollama", "serve"]),
    ],
)
def test_platform_specific_commands(platform_type, expected_cmd):
    """Test platform-specific command generation."""
    with patch(
        "clientai.ollama.manager.core.PlatformInfo", autospec=True
    ) as MockPlatformInfo:
        platform_info = MagicMock(spec=PlatformInfo)
        platform_info.platform = platform_type

        def get_server_command(config):
            base_cmd = (
                ["ollama.exe"]
                if platform_type == Platform.WINDOWS
                else ["ollama"]
            )
            return base_cmd + ["serve"]

        platform_info.get_server_command.side_effect = get_server_command
        MockPlatformInfo.return_value = platform_info

        manager = OllamaManager()
        result = manager._platform_info.get_server_command(manager.config)
        assert result == expected_cmd


@pytest.mark.parametrize(
    "gpu_vendor,config,expected_env",
    [
        (
            GPUVendor.NVIDIA,
            OllamaServerConfig(
                gpu_layers=35, gpu_memory_fraction=0.8, gpu_devices=0
            ),
            {
                "PATH": "/usr/local/bin",
                "OLLAMA_GPU_LAYERS": "35",
                "CUDA_VISIBLE_DEVICES": "0",
                "CUDA_MEM_FRACTION": "0.8",
            },
        ),
        (
            GPUVendor.AMD,
            OllamaServerConfig(
                gpu_layers=35, gpu_memory_fraction=0.8, gpu_devices=0
            ),
            {
                "PATH": "/usr/local/bin",
                "OLLAMA_GPU_LAYERS": "35",
                "GPU_DEVICE_ORDINAL": "0",
                "GPU_MAX_HEAP_SIZE": "80%",
            },
        ),
        (
            GPUVendor.APPLE,
            OllamaServerConfig(gpu_layers=35),
            {"PATH": "/usr/local/bin", "OLLAMA_GPU_LAYERS": "35"},
        ),
        (
            GPUVendor.NONE,
            OllamaServerConfig(gpu_layers=35),
            {"PATH": "/usr/local/bin", "OLLAMA_GPU_LAYERS": "35"},
        ),
    ],
)
def test_gpu_specific_environment(gpu_vendor, config, expected_env):
    """Test GPU-specific environment variable generation."""
    with patch(
        "clientai.ollama.manager.core.PlatformInfo", autospec=True
    ) as MockPlatformInfo:
        platform_info = MagicMock(spec=PlatformInfo)
        platform_info.platform = Platform.LINUX
        platform_info.gpu_vendor = gpu_vendor

        def get_environment(cfg):
            env = {
                "PATH": "/usr/local/bin",
            }

            if cfg.gpu_layers is not None:
                env["OLLAMA_GPU_LAYERS"] = str(cfg.gpu_layers)

            if gpu_vendor == GPUVendor.NVIDIA:
                if cfg.gpu_devices is not None:
                    devices = (
                        cfg.gpu_devices
                        if isinstance(cfg.gpu_devices, list)
                        else [cfg.gpu_devices]
                    )
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
                if cfg.gpu_memory_fraction is not None:
                    env["CUDA_MEM_FRACTION"] = str(cfg.gpu_memory_fraction)

            elif gpu_vendor == GPUVendor.AMD:  # pragma: no cover
                if cfg.gpu_devices is not None:
                    devices = (
                        cfg.gpu_devices
                        if isinstance(cfg.gpu_devices, list)
                        else [cfg.gpu_devices]
                    )
                    env["GPU_DEVICE_ORDINAL"] = ",".join(map(str, devices))
                if cfg.gpu_memory_fraction is not None:
                    env["GPU_MAX_HEAP_SIZE"] = (
                        f"{int(cfg.gpu_memory_fraction * 100)}%"
                    )

            return env

        platform_info.get_environment.side_effect = get_environment
        MockPlatformInfo.return_value = platform_info

        manager = OllamaManager(config)
        env = manager._platform_info.get_environment(config)

        for key, value in expected_env.items():
            assert (
                env[key] == value
            ), f"Expected {key}={value}, got {env.get(key)}"

        for key in env:
            assert (
                key in expected_env
            ), f"Unexpected environment variable: {key}"


def test_health_check_error_handling(
    manager, mock_subprocess, mock_http_client
):
    """Test various HTTP health check error scenarios."""
    manager.config.timeout = 0.1

    process = MockProcess()
    process._poll_value = None
    mock_subprocess.Popen.return_value = process

    mock_response = MagicMock()
    mock_response.status = 500
    mock_http_client.return_value.getresponse.return_value = mock_response

    with pytest.raises(ServerTimeoutError):
        manager.start()

    mock_http_client.return_value.request.side_effect = OSError()

    with pytest.raises(ServerTimeoutError):
        manager.start()

    mock_http_client.return_value.request.side_effect = (
        http.client.HTTPException()
    )

    with pytest.raises(ServerTimeoutError):
        manager.start()


def test_cleanup_on_error(manager, mock_subprocess, mock_http_client):
    """Test proper cleanup when an error occurs during startup."""
    process = MockProcess()
    mock_subprocess.Popen.return_value = process

    mock_http_client.return_value.request.side_effect = (
        http.client.HTTPException("Connection failed")
    )
    manager.config.timeout = 0.1

    with pytest.raises(ServerTimeoutError) as exc_info:
        manager.start()

    assert manager._process is None
    assert "did not start within" in str(exc_info.value)
    mock_http_client.return_value.close.assert_called()
    process.terminate.assert_called()


def test_resource_error_handling(manager, mock_subprocess):
    """Test handling of resource allocation errors."""
    process = MockProcess(returncode=1)
    process._poll_value = 1
    process.communicate.return_value = ("", "cannot allocate memory")
    mock_subprocess.Popen.return_value = process

    with pytest.raises(ServerStartupError) as exc_info:
        manager.start()

    assert "cannot allocate memory" in str(exc_info.value)


def test_custom_host_port(mock_subprocess, mock_http_client):
    """Test server startup with custom host and port."""
    config = OllamaServerConfig(host="localhost", port=11435)
    manager = OllamaManager(config)

    process = MockProcess()
    mock_subprocess.Popen.return_value = process

    try:
        manager.start()
    except ServerTimeoutError:  # pragma: no cover
        pass

    mock_http_client.assert_called_with("localhost", 11435, timeout=5)
