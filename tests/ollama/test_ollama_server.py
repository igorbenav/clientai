import pytest

from clientai.ollama import OllamaServerConfig


def test_default_config():
    """Test default configuration initialization."""
    config = OllamaServerConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 11434
    assert config.timeout == 30
    assert config.check_interval == 1.0
    assert config.gpu_layers is None
    assert config.compute_unit is None
    assert config.cpu_threads is None
    assert config.memory_limit is None
    assert config.gpu_memory_fraction is None
    assert config.gpu_devices is None
    assert config.env_vars == {}
    assert config.extra_args == []


def test_host_validation():
    """Test host address validation."""
    OllamaServerConfig(host="127.0.0.1")
    OllamaServerConfig(host="0.0.0.0")
    OllamaServerConfig(host="localhost")
    OllamaServerConfig(host="example.com")

    with pytest.raises(ValueError, match="Invalid host"):
        OllamaServerConfig(host="invalid..host")
    with pytest.raises(ValueError, match="Host cannot be empty"):
        OllamaServerConfig(host="")


def test_port_validation():
    """Test port number validation."""
    OllamaServerConfig(port=1)
    OllamaServerConfig(port=8080)
    OllamaServerConfig(port=65535)

    with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
        OllamaServerConfig(port=0)
    with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
        OllamaServerConfig(port=65536)


def test_timeout_and_interval_validation():
    """Test timeout and check interval validation."""
    OllamaServerConfig(timeout=30, check_interval=1.0)
    OllamaServerConfig(timeout=60, check_interval=2.0)

    with pytest.raises(ValueError, match="Timeout must be positive"):
        OllamaServerConfig(timeout=0)
    with pytest.raises(ValueError, match="Check interval must be positive"):
        OllamaServerConfig(check_interval=0)
    with pytest.raises(
        ValueError, match="Check interval cannot be greater than timeout"
    ):
        OllamaServerConfig(timeout=10, check_interval=20)


def test_gpu_settings_validation():
    """Test GPU-related settings validation."""
    OllamaServerConfig(gpu_layers=35)
    OllamaServerConfig(gpu_memory_fraction=0.8)
    OllamaServerConfig(gpu_devices=0)
    OllamaServerConfig(gpu_devices=[0, 1])

    with pytest.raises(
        ValueError, match="gpu_layers must be a non-negative integer"
    ):
        OllamaServerConfig(gpu_layers=-1)

    with pytest.raises(
        ValueError, match="gpu_memory_fraction must be between 0.0 and 1.0"
    ):
        OllamaServerConfig(gpu_memory_fraction=1.5)

    with pytest.raises(ValueError, match="GPU device ID must be non-negative"):
        OllamaServerConfig(gpu_devices=-1)
    with pytest.raises(
        ValueError, match="All GPU device IDs must be non-negative integers"
    ):
        OllamaServerConfig(gpu_devices=[-1, 0])
    with pytest.raises(
        ValueError, match="Duplicate GPU device IDs are not allowed"
    ):
        OllamaServerConfig(gpu_devices=[0, 0])
    with pytest.raises(
        ValueError, match="gpu_devices must be an integer or list of integers"
    ):
        OllamaServerConfig(gpu_devices="0")


def test_compute_unit_validation():
    """Test compute unit validation."""
    OllamaServerConfig(compute_unit="cpu")
    OllamaServerConfig(compute_unit="gpu")
    OllamaServerConfig(compute_unit="auto")

    with pytest.raises(ValueError, match="compute_unit must be one of"):
        OllamaServerConfig(compute_unit="invalid")


def test_cpu_threads_validation():
    """Test CPU threads validation."""
    OllamaServerConfig(cpu_threads=1)
    OllamaServerConfig(cpu_threads=8)

    with pytest.raises(
        ValueError, match="cpu_threads must be a positive integer"
    ):
        OllamaServerConfig(cpu_threads=0)
    with pytest.raises(
        ValueError, match="cpu_threads must be a positive integer"
    ):
        OllamaServerConfig(cpu_threads=-1)


def test_memory_limit_validation():
    """Test memory limit validation."""
    OllamaServerConfig(memory_limit="8GiB")
    OllamaServerConfig(memory_limit="1024MiB")
    OllamaServerConfig(memory_limit="1TiB")
    OllamaServerConfig(memory_limit="0.5GiB")

    with pytest.raises(ValueError, match="memory_limit must be in format"):
        OllamaServerConfig(memory_limit="8GB")
    with pytest.raises(ValueError, match="memory_limit must be in format"):
        OllamaServerConfig(memory_limit="8G")
    with pytest.raises(ValueError, match="memory_limit must be in format"):
        OllamaServerConfig(memory_limit="8 GiB")

    with pytest.raises(
        ValueError, match="memory_limit in MiB must be at least 100"
    ):
        OllamaServerConfig(memory_limit="50MiB")
    with pytest.raises(
        ValueError, match="memory_limit in GiB must be at least 0.1"
    ):
        OllamaServerConfig(memory_limit="0.05GiB")
    with pytest.raises(
        ValueError, match="memory_limit in TiB must be at least 0.001"
    ):
        OllamaServerConfig(memory_limit="0.0005TiB")


def test_env_vars_validation():
    """Test environment variables validation."""
    OllamaServerConfig(env_vars={"KEY": "value"})
    OllamaServerConfig(env_vars={"MULTIPLE": "vars", "ARE": "valid"})

    with pytest.raises(
        ValueError, match="All environment variables must be strings"
    ):
        OllamaServerConfig(env_vars={"KEY": 123})
    with pytest.raises(
        ValueError, match="All environment variables must be strings"
    ):
        OllamaServerConfig(env_vars={123: "value"})


def test_extra_args_validation():
    """Test extra arguments validation."""
    OllamaServerConfig(extra_args=["--verbose"])
    OllamaServerConfig(extra_args=["--arg1", "--arg2"])

    with pytest.raises(
        ValueError, match="All extra arguments must be strings"
    ):
        OllamaServerConfig(extra_args=[123])
    with pytest.raises(
        ValueError, match="All extra arguments must be strings"
    ):
        OllamaServerConfig(extra_args=["--valid", 123])


def test_base_url_property():
    """Test base_url property."""
    config = OllamaServerConfig(host="localhost", port=8080)
    assert config.base_url == "http://localhost:8080"

    config = OllamaServerConfig()
    assert config.base_url == "http://127.0.0.1:11434"
