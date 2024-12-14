# OllamaManager Class API Reference

The `OllamaManager` class is a utility class that manages the lifecycle of a local Ollama server instance. It handles server process startup, monitoring, and shutdown while respecting platform-specific requirements and custom configurations. The manager supports configurable GPU acceleration, CPU thread allocation, and memory limits through `OllamaServerConfig`. It provides both context manager and manual management interfaces for controlling the server process.

## Class Definition

::: clientai.ollama.OllamaManager
    rendering:
      show_if_no_docstring: true