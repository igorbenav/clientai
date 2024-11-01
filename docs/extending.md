# Extending ClientAI: Adding a New Provider

This guide will walk you through the process of adding support for a new AI provider to the ClientAI package. By following these steps, you'll be able to integrate a new provider seamlessly into the existing structure.

## Overview

To add a new provider, you'll need to:

1. Create a new directory for the provider
2. Implement the provider-specific types
3. Implement the provider class
4. Update the main ClientAI class
5. Update the package constants
6. Add tests for the new provider

Let's go through each step in detail.

## Step 1: Create a New Directory

First, create a new directory for your provider in the `clientai` folder. For example, if you're adding support for a provider called "NewAI", create a directory named `newai`:

```
clientai/
    newai/
        __init__.py
        _typing.py
        provider.py
```

## Step 2: Implement Provider-Specific Types

In the `_typing.py` file, define the types specific to your provider. This should include response types, client types, and any other necessary types. Here's an example structure:

```python
# clientai/newai/_typing.py

from typing import Any, Dict, Iterator, Protocol, TypedDict, Union
from .._common_types import GenericResponse

class NewAIResponse(TypedDict):
    # Define the structure of a full response from NewAI
    pass

class NewAIStreamResponse(TypedDict):
    # Define the structure of a streaming response chunk from NewAI
    pass

class NewAIClientProtocol(Protocol):
    # Define the expected interface for the NewAI client
    pass

NewAIGenericResponse = GenericResponse[
    str, NewAIResponse, NewAIStreamResponse
]

# Add any other necessary types
```

## Step 3: Implement the Provider Class

In the `provider.py` file, implement the `Provider` class that inherits from `AIProvider`. This class should implement the `generate_text` and `chat` methods:

```python
# clientai/newai/provider.py

from ..ai_provider import AIProvider
from ._typing import NewAIClientProtocol, NewAIGenericResponse
from typing import List
from .._common_types import Message

class Provider(AIProvider):
    def __init__(self, api_key: str):
        # Initialize the NewAI client
        pass

    def generate_text(
        self,
        prompt: str,
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs
    ) -> NewAIGenericResponse:
        # Implement text generation logic
        pass

    def chat(
        self,
        messages: List[Message],
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs
    ) -> NewAIGenericResponse:
        # Implement chat logic
        pass

    # Implement any helper methods as needed
```

Make sure to handle both streaming and non-streaming responses, as well as the `return_full_response` option.

## Step 4: Implement Unified Error Handling

Before implementing the provider class, set up error handling for your provider. This ensures consistent error reporting across all providers.

1. First, import the necessary error types:

```python
# clientai/newai/provider.py

from ..exceptions import (
    APIError,
    AuthenticationError,
    ClientAIError,
    InvalidRequestError,
    ModelError,
    RateLimitError,
    TimeoutError,
)
```

2. Implement the error mapping method in your provider class:

```python
class Provider(AIProvider):
    ...
    def _map_exception_to_clientai_error(
        self,
        e: Exception,
        status_code: Optional[int] = None
    ) -> ClientAIError:
        """
        Maps NewAI-specific exceptions to ClientAI exceptions.

        Args:
            e: The caught exception
            status_code: Optional HTTP status code

        Returns:
            ClientAIError: The appropriate ClientAI exception
        """
        error_message = str(e)
        status_code = status_code or getattr(e, "status_code", None)

        # Map NewAI-specific exceptions to ClientAI exceptions
        if isinstance(e, NewAIAuthError):
            return AuthenticationError(
                error_message,
                status_code=401,
                original_error=e
            )
        elif isinstance(e, NewAIRateLimitError):
            return RateLimitError(
                error_message,
                status_code=429,
                original_error=e
            )
        elif "model not found" in error_message.lower():
            return ModelError(
                error_message,
                status_code=404,
                original_error=e
            )
        elif isinstance(e, NewAIInvalidRequestError):
            return InvalidRequestError(
                error_message,
                status_code=400,
                original_error=e
            )
        elif isinstance(e, NewAITimeoutError):
            return TimeoutError(
                error_message,
                status_code=408,
                original_error=e
            )
        
        # Default to APIError for unknown errors
        return APIError(
            error_message,
            status_code,
            original_error=e
        )
```

## Step 5: Update the Main ClientAI Class

Update the `clientai/client_ai.py` file to include support for your new provider:

1. Add an import for your new provider:
   ```python
   from .newai import NEWAI_INSTALLED
   ```

2. Update the `__init__` method of the `ClientAI` class to handle the new provider:
   ```python
    
    def __init__(self, provider_name: str, **kwargs):
        prov_name = provider_name
        # ----- add "newai" here -----
        if prov_name not in ["openai", "replicate", "ollama", "newai"]:
            raise ValueError(f"Unsupported provider: {prov_name}")

        if (
            prov_name == "openai"
            and not OPENAI_INSTALLED
            ...
            # ----- also add "newai" here -----
            or prov_name == "newai" 
            and not NEWAI_INSTALLED
        ):
            raise ImportError(
                f"The {prov_name} provider is not installed. "
                f"Please install it with 'pip install clientai[{prov_name}]'."
            )
   ```

3. Add the new provider to the provider initialization logic:
    ```python
    ...
    try:
        provider_module = import_module(
            f".{prov_name}.provider", package="clientai"
        )
        provider_class = getattr(provider_module, "Provider")
        
        # ----- add "newai" here -----
        if prov_name in ["openai", "replicate", "newai"]:
            self.provider = cast(
                P, provider_class(api_key=kwargs.get("api_key"))
            )
        ...
    ```

## Step 6: Update Package Constants and Dependencies

1. In the `clientai/_constants.py` file, add a constant for your new provider:

```python
NEWAI_INSTALLED = find_spec("newai") is not None
```

2. Update the `clientai/__init__.py` file to export the new constant:

```python
from ._constants import NEWAI_INSTALLED
__all__ = [
    # ... existing exports ...
    "NEWAI_INSTALLED",
]
```

3. Update the `pyproject.toml` file to include the new provider as an optional dependency:

```toml
[tool.poetry.dependencies]
python = "^3.9"
pydantic = "^2.9.2"
openai = {version = "^1.50.2", optional = true}
replicate = {version = "^0.34.1", optional = true}
ollama = {version = "^0.3.3", optional = true}
newai-package = {version = "^1.0.0", optional = true}  # Add this line
```

4. Define an optional group for the new provider:

```toml
[tool.poetry.group.newai]
optional = true

[tool.poetry.group.newai.dependencies]
newai-package = "^1.0.0"
```

5. Include the new provider in the development dependencies:

```toml
[tool.poetry.group.dev.dependencies]
ruff = "^0.6.8"
pytest = "^8.3.3"
mypy = "1.9.0"
openai = "^1.50.2"
replicate = "^0.34.1"
ollama = "^0.3.3"
newai-package = "^1.0.0"  # Add this line
```

6. Run `poetry update` to update the `poetry.lock` file with the new dependencies.

These changes allow users to install the new provider's dependencies using Poetry:

```
poetry install --with newai
```

If users are not using Poetry and are installing the package via pip, they can still use the extras syntax:

```
pip install clientai[newai]
```

## Step 7: Add Tests

Create a new test file for your provider in the `tests` directory:

```
tests/
    newai/
        __init__.py
        test_provider.py
```

Implement tests for your new provider, ensuring that you cover both the `generate_text` and `chat` methods, as well as streaming and non-streaming scenarios.

## Step 8: Test Error Handling

Also create a new test file to test your provider's exceptions in the `tests` directory:

```
tests/
    newai/
        __init__.py
        test_exceptions.py
```

Add tests to ensure unified tests are being handled with a reference to the original error.

## Step 9: Update Documentation

Don't forget to update the documentation to include information about the new provider:

1. Add a new file `docs/api/newai_provider.md` with the following template for the NewAI provider.
```md
# NewAI Provider API Reference

The `NewAI` class implements the `AIProvider` interface for the NewAI service. It provides methods for text generation and chat functionality using NewAI's models.

## Class Definition

::: clientai.newai.Provider
    rendering:
      show_if_no_docstring: true
```

2. Update `docs/index.md` to add a reference to the new provider.
3. Update `docs/quick-start.md` to include an example of how to use the new provider.
4. Update `docs/api/overview.md` to include a link to the new provider's documentation.

## Contribution Guidelines

After implementing your new provider, it's important to ensure that your code meets the project's quality standards and follows the contribution guidelines. Please refer to our [Contributing Guide](./community/CONTRIBUTING.md) for detailed information on how to contribute to ClientAI.

### Quick Summary of Development Tools

ClientAI uses the following tools for maintaining code quality:

1. **pytest**: For running tests
   ```
   poetry run pytest
   ```

2. **mypy**: For type checking
   ```
   mypy clientai
   ```

3. **ruff**: For code linting and formatting
   ```
   ruff check --fix
   ruff format
   ```

Make sure to run these tools and address any issues before submitting your contribution.

## Conclusion

By following these steps, you can successfully add support for a new AI provider to the ClientAI package. Remember to maintain consistency with the existing code style and structure, and to thoroughly test your implementation.

If you're contributing this new provider to the main ClientAI repository, make sure to follow the contribution guidelines and submit a pull request with your changes. Your contribution will help expand the capabilities of ClientAI and benefit the entire community.

Thank you for your interest in extending ClientAI!