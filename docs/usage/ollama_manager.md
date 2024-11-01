# Ollama Manager Guide

## Introduction

Ollama Manager provides a streamlined way to prototype and develop applications using Ollama's AI models. Instead of manually managing the Ollama server process, installing it as a service, or running it in a separate terminal, Ollama Manager handles the entire lifecycle programmatically.

**Key Benefits for Prototyping:**
- Start/stop Ollama server automatically within your Python code
- Configure resources dynamically based on your needs
- Handle multiple server instances for testing
- Automatic cleanup of resources
- Platform-independent operation

## Quick Start

```python
from clientai import ClientAI
from clientai.ollama import OllamaManager

# Basic usage - server starts automatically and stops when done
with OllamaManager() as manager:
    # Create a client that connects to the managed server
    client = ClientAI('ollama', host="http://localhost:11434")
    
    # Use the client normally
    response = client.generate_text(
        "Explain quantum computing",
        model="llama2"
    )
    print(response)
# Server automatically stops when exiting the context

```

## Installation

```bash
# Install with Ollama support
pip install "clientai[ollama]"

# Install with all providers
pip install "clientai[all]"
```

## Core Concepts

### Server Lifecycle Management

1. **Context Manager (Recommended)**
   ```python
   with OllamaManager() as manager:
       # Server starts automatically
       client = ClientAI('ollama')
       # Use client...
   # Server stops automatically
   ```

2. **Manual Management**
   ```python
   manager = OllamaManager()
   try:
       manager.start()
       client = ClientAI('ollama')
       # Use client...
   finally:
       manager.stop()
   ```

### Configuration Management

```python
from clientai.ollama import OllamaServerConfig

# Create custom configuration
config = OllamaServerConfig(
    host="127.0.0.1",
    port=11434,
    gpu_layers=35,
    memory_limit="8GiB"
)

# Use configuration with manager
with OllamaManager(config) as manager:
    client = ClientAI('ollama')
    # Use client...
```

## Resource Configuration Guide

### GPU Configuration

1. **Basic GPU Setup**
   ```python
   config = OllamaServerConfig(
       gpu_layers=35,  # Number of layers to run on GPU
       gpu_memory_fraction=0.8  # 80% of GPU memory
   )
   ```

2. **Multi-GPU Setup**
   ```python
   config = OllamaServerConfig(
       gpu_devices=[0, 1],  # Use first two GPUs
       gpu_memory_fraction=0.7
   )
   ```

### Memory Management

The `memory_limit` parameter requires specific formatting following Go's memory limit syntax:

```python
# Correct memory limit formats
config = OllamaServerConfig(
    memory_limit="8GiB"    # 8 gibibytes
    # OR
    memory_limit="8192MiB" # Same as 8GiB
)

# Invalid formats that will cause errors
config = OllamaServerConfig(
    memory_limit="8GiB"     # Wrong unit
    memory_limit="8 GiB"   # No spaces allowed
    memory_limit="8g"      # Must specify full unit
)
```

**Valid Memory Units:**
- `B` - Bytes
- `KiB` - Kibibytes (1024 bytes)
- `MiB` - Mebibytes (1024² bytes)
- `GiB` - Gibibytes (1024³ bytes)
- `TiB` - Tebibytes (1024⁴ bytes)

**Common Configurations:**

1. **High-Performance Setup**
   ```python
   config = OllamaServerConfig(
       memory_limit="24GiB",
       cpu_threads=16,
       gpu_layers=40
   )
   ```

2. **Balanced Setup**
   ```python
   config = OllamaServerConfig(
       memory_limit="16GiB",
       cpu_threads=8,
       gpu_layers=32
   )
   ```

3. **Resource-Constrained Setup**
   ```python
   config = OllamaServerConfig(
       memory_limit="8GiB",
       cpu_threads=4,
       gpu_layers=24
   )
   ```

4. **Dynamic Memory Allocation**
   ```python
   import psutil
   
   # Calculate available memory and use 70%
   available_gib = (psutil.virtual_memory().available / (1024**3))
   memory_limit = f"{int(available_gib * 0.7)}GiB"
   
   config = OllamaServerConfig(
       memory_limit=memory_limit,
       cpu_threads=8
   )
   ```

### Model-Specific Memory Guidelines

Different models require different amounts of memory:

1. **Large Language Models (>30B parameters)**
   ```python
   config = OllamaServerConfig(
       memory_limit="24GiB",
       gpu_memory_fraction=0.9
   )
   ```

2. **Medium Models (7B-30B parameters)**
   ```python
   config = OllamaServerConfig(
       memory_limit="16GiB",
       gpu_memory_fraction=0.8
   )
   ```

3. **Small Models (<7B parameters)**
   ```python
   config = OllamaServerConfig(
       memory_limit="8GiB",
       gpu_memory_fraction=0.7
   )
   ```

### Advanced Configuration Reference

```python
config = OllamaServerConfig(
    # Server settings
    host="127.0.0.1",      # Server bind address
    port=11434,            # Server port number
    timeout=30,            # Maximum startup wait time in seconds
    check_interval=1.0,    # Health check interval in seconds
    
    # GPU settings
    gpu_layers=35,          # More layers = more GPU utilization
    gpu_memory_fraction=0.8, # 0.0 to 1.0 (80% GPU memory)
    gpu_devices=[0],        # Specific GPU devices to use
    
    # CPU settings
    cpu_threads=8,          # Number of CPU threads
    memory_limit="16GiB",   # Maximum RAM usage (must use GiB/MiB units)
    
    # Compute settings
    compute_unit="auto",    # "auto", "cpu", or "gpu"
    
    # Additional settings
    env_vars={"CUSTOM_VAR": "value"},  # Additional environment variables
    extra_args=["--verbose"]           # Additional command line arguments
)
```

Each setting explained:

**Server Settings:**
- `host`: IP address to bind the server to
  - Default: "127.0.0.1" (localhost)
  - Use "0.0.0.0" to allow external connections
- `port`: Port number for the server
  - Default: 11434
  - Change if default port is in use
- `timeout`: Maximum time to wait for server startup
  - Unit: seconds
  - Increase for slower systems
- `check_interval`: Time between server health checks
  - Unit: seconds
  - Adjust based on system responsiveness

**GPU Settings:**
- `gpu_layers`: Number of model layers to offload to GPU
  - Higher = more GPU utilization
  - Lower = more CPU utilization
  - Model-dependent (typically 24-40)
- `gpu_memory_fraction`: Portion of GPU memory to use
  - Range: 0.0 to 1.0
  - Higher values may improve performance
  - Lower values leave room for other applications
- `gpu_devices`: Specific GPU devices to use
  - Single GPU: `gpu_devices=0`
  - Multiple GPUs: `gpu_devices=[0, 1]`
  - None: `gpu_devices=None`

**CPU Settings:**
- `cpu_threads`: Number of CPU threads to use
  - Default: System dependent
  - Recommended: Leave some threads for system
  - Example: `os.cpu_count() - 2`
- `memory_limit`: Maximum RAM allocation
  - Must use `GiB` or `MiB` units
  - Examples: "8GiB", "16384MiB"
  - Should not exceed available system RAM

**Compute Settings:**
- `compute_unit`: Preferred compute device
  - "auto": Let Ollama decide (recommended)
  - "cpu": Force CPU-only operation
  - "gpu": Force GPU operation if available

**Additional Settings:**
- `env_vars`: Additional environment variables
  - Used for platform-specific settings
  - Example: `{"CUDA_VISIBLE_DEVICES": "0,1"}`
- `extra_args`: Additional CLI arguments
  - Passed directly to Ollama server
  - Example: `["--verbose", "--debug"]`
```

## Common Use Cases

### 1. Development and Prototyping

```python
# Quick setup for development
with OllamaManager() as manager:
    client = ClientAI('ollama')
    
    # Test different prompts
    prompts = [
        "Write a poem about AI",
        "Explain quantum physics",
        "Create a Python function"
    ]
    
    for prompt in prompts:
        response = client.generate_text(prompt, model="llama2")
        print(f"Prompt: {prompt}\nResponse: {response}\n")
```

### 2. Multiple Model Testing

```python
# Test different models with different configurations
models = ["llama2", "codellama", "mistral"]

for model in models:
    # Adjust configuration based on model
    if model == "llama2":
        config = OllamaServerConfig(gpu_layers=35)
    else:
        config = OllamaServerConfig(gpu_layers=28)
    
    with OllamaManager(config) as manager:
        client = ClientAI('ollama')
        response = client.generate_text(
            "Explain how you work",
            model=model
        )
        print(f"{model}: {response}\n")
```

### 3. A/B Testing Configurations

```python
def test_configuration(config, prompt, model):
    start_time = time.time()
    
    with OllamaManager(config) as manager:
        client = ClientAI('ollama')
        response = client.generate_text(prompt, model=model)
        
    duration = time.time() - start_time
    return response, duration

# Test different configurations
configs = {
    "high_gpu": OllamaServerConfig(gpu_layers=40, gpu_memory_fraction=0.9),
    "balanced": OllamaServerConfig(gpu_layers=32, gpu_memory_fraction=0.7),
    "low_resource": OllamaServerConfig(gpu_layers=24, gpu_memory_fraction=0.5)
}

for name, config in configs.items():
    response, duration = test_configuration(
        config,
        "Write a long story about space",
        "llama2"
    )
    print(f"Configuration {name}: {duration:.2f} seconds")
```

### 4. Production Setup

```python
import logging

logging.basicConfig(level=logging.INFO)

def create_production_manager():
    config = OllamaServerConfig(
        # Stable production settings
        gpu_layers=32,
        gpu_memory_fraction=0.7,
        memory_limit="16GiB",
        timeout=60,  # Longer timeout for stability
        check_interval=2.0
    )
    
    try:
        manager = OllamaManager(config)
        manager.start()
        return manager
    except Exception as e:
        logging.error(f"Failed to start Ollama: {e}")
        raise

# Use in production
try:
    manager = create_production_manager()
    client = ClientAI('ollama')
    # Use client...
finally:
    manager.stop()
```

## Error Handling

Ollama Manager provides several specific exception types to help you handle different error scenarios effectively:

### ExecutableNotFoundError
Occurs when the Ollama executable cannot be found on the system.

**Common causes:**
- Ollama not installed
- Ollama not in system PATH
- Incorrect installation

**How to handle:**
```python
try:
    manager = OllamaManager()
    manager.start()
except ExecutableNotFoundError:
    # Guide user through installation
    if platform.system() == "Darwin":
        print("Install Ollama using: brew install ollama")
    elif platform.system() == "Linux":
        print("Install Ollama using: curl -fsSL https://ollama.com/install.sh | sh")
    else:
        print("Download Ollama from: https://ollama.com/download")
```

### ServerStartupError
Occurs when the server fails to start properly.

**Common causes:**
- Port already in use
- Insufficient permissions
- Corrupt installation
- Resource conflicts

**How to handle:**
```python
try:
    manager = OllamaManager()
    manager.start()
except ServerStartupError as e:
    if "address already in use" in str(e).lower():
        # Try alternative port
        config = OllamaServerConfig(port=11435)
        manager = OllamaManager(config)
        manager.start()
    elif "permission denied" in str(e).lower():
        print("Please run with appropriate permissions")
    else:
        print(f"Server startup failed: {e}")
```

### ServerTimeoutError
Occurs when the server doesn't respond within the configured timeout period.

**Common causes:**
- System under heavy load
- Insufficient resources
- Network issues
- Too short timeout period

**How to handle:**
```python
config = OllamaServerConfig(
    timeout=60,  # Increase timeout
    check_interval=2.0  # Reduce check frequency
)
try:
    manager = OllamaManager(config)
    manager.start()
except ServerTimeoutError:
    # Either retry with longer timeout or fail gracefully
    config.timeout = 120
    try:
        manager = OllamaManager(config)
        manager.start()
    except ServerTimeoutError:
        print("Server unresponsive after extended timeout")
```

### ResourceError
Occurs when there are insufficient system resources to run Ollama.

**Common causes:**
- Insufficient memory
- GPU memory allocation failures
- Too many CPU threads requested
- Disk space constraints

**How to handle:**
```python
try:
    manager = OllamaManager()
    manager.start()
except ResourceError as e:
    if "memory" in str(e).lower():
        # Try with reduced memory settings
        config = OllamaServerConfig(
            memory_limit="4GiB",
            gpu_memory_fraction=0.5
        )
    elif "gpu" in str(e).lower():
        # Fallback to CPU
        config = OllamaServerConfig(compute_unit="cpu")
    
    try:
        manager = OllamaManager(config)
        manager.start()
    except ResourceError:
        print("Unable to allocate required resources")
```

### ConfigurationError
Occurs when provided configuration values are invalid.

**Common causes:**
- Invalid memory format
- Invalid GPU configuration
- Incompatible settings
- Out-of-range values

**How to handle:**
```python
try:
    config = OllamaServerConfig(
        gpu_memory_fraction=1.5  # Invalid value
    )
except ConfigurationError as e:
    # Use safe default values
    config = OllamaServerConfig(
        gpu_memory_fraction=0.7,
        gpu_layers=32
    )
```

### UnsupportedPlatformError
Occurs when running on an unsupported platform or configuration.

**Common causes:**
- Unsupported operating system
- Missing system features
- Incompatible hardware

**How to handle:**
```python
try:
    manager = OllamaManager()
    manager.start()
except UnsupportedPlatformError as e:
    # Fall back to alternative configuration or inform user
    print(f"Platform not supported: {e}")
    print("Supported platforms: Windows, macOS, Linux")
```

### Memory-Related Error Handling

```python
def start_with_memory_fallback():
    try:
        # Try optimal memory configuration
        config = OllamaServerConfig(memory_limit="16GiB")
        return OllamaManager(config)
    except ServerStartupError as e:
        if "GOMEMLIMIT" in str(e):
            # Fall back to lower memory configuration
            config = OllamaServerConfig(memory_limit="8GiB")
            return OllamaManager(config)
        raise  # Re-raise if error is not memory-related
```

### Best Practices for Error Handling

1. **Graceful Degradation**
```python
def start_with_fallback():
    # Try optimal configuration first
    config = OllamaServerConfig(
        gpu_layers=35,
        gpu_memory_fraction=0.8
    )
    
    try:
        return OllamaManager(config)
    except ResourceError:
        # Fall back to minimal configuration
        config = OllamaServerConfig(
            gpu_layers=24,
            gpu_memory_fraction=0.5
        )
        return OllamaManager(config)
```

2. **Logging and Monitoring**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    manager = OllamaManager()
    manager.start()
except ServerStartupError as e:
    logger.error(f"Server startup failed: {e}")
    logger.debug(f"Full error details: {e.original_exception}")
```

3. **Recovery Strategies**
```python
def start_with_retry(max_attempts=3, delay=5):
    for attempt in range(max_attempts):
        try:
            manager = OllamaManager()
            manager.start()
            return manager
        except (ServerStartupError, ServerTimeoutError) as e:
            if attempt == max_attempts - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(delay)
```

Remember to always clean up resources properly, even when handling errors:
```python
manager = None
try:
    manager = OllamaManager()
    manager.start()
    # Use manager...
except Exception as e:
    logger.error(f"Error occurred: {e}")
finally:
    if manager is not None:
        manager.stop()
```

## Performance Monitoring

```python
import psutil
from contextlib import contextmanager
import time

@contextmanager
def monitor_performance():
    start_time = time.time()
    start_cpu = psutil.cpu_percent()
    start_mem = psutil.virtual_memory().percent
    
    yield
    
    duration = time.time() - start_time
    cpu_diff = psutil.cpu_percent() - start_cpu
    mem_diff = psutil.virtual_memory().percent - start_mem
    
    print(f"Duration: {duration:.2f}s")
    print(f"CPU impact: {cpu_diff:+.1f}%")
    print(f"Memory impact: {mem_diff:+.1f}%")

# Use with Ollama Manager
with monitor_performance():
    with OllamaManager() as manager:
        client = ClientAI('ollama')
        response = client.generate_text(
            "Write a story",
            model="llama2"
        )
```

## Best Practices

1. **Always use context managers** when possible
2. **Start with conservative resource settings** and adjust up
3. **Monitor system resources** during development
4. **Implement proper error handling**
5. **Use appropriate configurations** for development vs. production
6. **Clean up resources** properly in all code paths
7. **Log important events** for troubleshooting
8. **Test different configurations** to find optimal settings
9. **Consider platform-specific settings** for cross-platform applications
10. **Implement graceful degradation** when resources are constrained

## Troubleshooting

- **Server won't start**: Check if Ollama is installed and port is available
- **Performance issues**: Monitor and adjust resource configuration
- **Memory errors**: Reduce memory_limit and gpu_memory_fraction
- **GPU errors**: Try reducing gpu_layers or switch to CPU
- **Timeout errors**: Increase timeout value in configuration
- **Platform-specific issues**: Check platform support and requirements

Remember that Ollama Manager is a powerful tool for prototyping and development, but always monitor system resources and adjust configurations based on your specific needs and hardware capabilities.