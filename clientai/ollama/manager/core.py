import http.client
import logging
import subprocess
import sys
import time
from typing import Any, Dict, Optional, cast
from urllib.parse import urlparse

from .config import OllamaServerConfig
from .exceptions import (
    ExecutableNotFoundError,
    ResourceError,
    ServerStartupError,
    ServerTimeoutError,
    raise_ollama_error,
)
from .platform_info import Platform, PlatformInfo


class OllamaManager:
    """
    Manages the Ollama server process and configuration.

    This class provides methods to start, stop, and manage the lifecycle
    of an Ollama server instance with configurable parameters.

    Attributes:
        config: The server configuration used by the manager.
        _process: The underlying server process.
        _platform_info: Information about the current platform.

    Args:
        config: Optional server configuration. If None, uses defaults.

    Raises:
        ImportError: If required system dependencies are not installed.

    Example:
        Basic usage with defaults:
        ```python
        with OllamaManager() as manager:
            # Server is running with default configuration
            pass  # Server automatically stops when exiting context
        ```

        Custom configuration:
        ```python
        config = OllamaServerConfig(
            gpu_layers=35,
            gpu_memory_fraction=0.8,
            cpu_threads=8
        )
        manager = OllamaManager(config)
        manager.start()
        # ... use the server ...
        manager.stop()
        ```
    """

    def __init__(self, config: Optional[OllamaServerConfig] = None) -> None:
        """
        Initialize the Ollama manager.

        Args:
            config: Optional server configuration. If None, uses defaults.
        """
        self.config = config or OllamaServerConfig()
        self._process: Optional[subprocess.Popen[str]] = None
        self._platform_info = PlatformInfo()

    def start(self) -> None:
        """
        Start the Ollama server using the configured parameters.

        This method initializes and starts the Ollama server process,
        waiting for it to become healthy before returning.

        Raises:
            ServerStartupError: If the server fails to start
            ServerTimeoutError: If the server doesn't respond within timeout
            ExecutableNotFoundError: If the Ollama executable is not found
            ResourceError: If there are insufficient system resources

        Example:
            Start with default configuration:
            ```python
            manager = OllamaManager()
            manager.start()
            ```

            Start with custom configuration:
            ```python
            config = OllamaServerConfig(gpu_layers=35)
            manager = OllamaManager(config)
            manager.start()
            ```
        """
        if self._process is not None:
            raise ServerStartupError("Ollama server is already running")

        logging.info(
            f"Starting Ollama server on {self.config.host}:{self.config.port}"
        )

        try:
            popen_kwargs: Dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "env": self._platform_info.get_environment(self.config),
            }

            if self._platform_info.platform == Platform.WINDOWS:
                if sys.platform == "win32":
                    popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            self._process = subprocess.Popen(
                self._platform_info.get_server_command(self.config),
                **popen_kwargs,
            )

        except FileNotFoundError as e:
            raise_ollama_error(
                ExecutableNotFoundError,
                "Ollama executable not found. Ensure Ollama is installed.",
                e,
            )
        except MemoryError as e:
            raise_ollama_error(
                ResourceError, "Insufficient memory to start Ollama server", e
            )
        except Exception as e:
            raise_ollama_error(
                ServerStartupError,
                f"Failed to start Ollama server: {str(e)}",
                e,
            )

        try:
            self._wait_for_server()
        except Exception as e:
            self.stop()
            raise e

    def stop(self) -> None:
        """
        Stop the running Ollama server instance.

        This method terminates the Ollama server process if it's running.
        It will wait for the process to complete before returning.

        Example:
            Stop a running server:
            ```python
            manager = OllamaManager()
            manager.start()
            # ... use the server ...
            manager.stop()
            ```

            Using context manager (automatic stop):
            ```python
            with OllamaManager() as manager:
                # ... use the server ...
                pass  # Server stops automatically
            ```
        """
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait()
            finally:
                self._process = None
                logging.info("Ollama server stopped")

    def _check_server_health(self) -> bool:
        """
        Check if the server is responding to health checks.

        This method attempts to connect to the server and verify
        its health status.

        Returns:
            bool: True if server is healthy, False otherwise

        Note:
            This is an internal method used by the manager to verify
            server status during startup.
        """
        try:
            url = urlparse(self.config.base_url)
            conn = http.client.HTTPConnection(
                url.hostname or self.config.host,
                url.port or self.config.port,
                timeout=5,
            )
            try:
                conn.request("GET", "/")
                response = conn.getresponse()
                return response.status == 200
            finally:
                conn.close()
        except (http.client.HTTPException, ConnectionRefusedError, OSError):
            return False

    def _wait_for_server(self) -> None:
        """
        Wait for the server to become ready and responsive.

        This method polls the server until it responds successfully or
        times out. It checks both process health and server responsiveness.

        Raises:
            ServerStartupError: If the server process terminates unexpectedly
            ServerTimeoutError: If the server doesn't respond within timeout

        Note:
            This is an internal method used during the server startup process.
        """
        start_time = time.time()

        while time.time() - start_time < self.config.timeout:
            process = cast(subprocess.Popen[str], self._process)

            if process.poll() is not None:
                stdout, stderr = process.communicate()
                error_msg = (
                    f"Ollama process terminated unexpectedly.\n"
                    f"Exit code: {process.returncode}\n"
                    f"stdout: {stdout}\n"
                    f"stderr: {stderr}"
                )
                self._process = None
                raise ServerStartupError(error_msg)

            if self._check_server_health():
                logging.info("Ollama server is ready")
                return

            time.sleep(self.config.check_interval)

        raise ServerTimeoutError(
            f"Ollama server did not start within {self.config.timeout} seconds"
        )

    def __enter__(self) -> "OllamaManager":
        """Context manager entry point that starts the server."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point that ensures the server is stopped."""
        self.stop()
