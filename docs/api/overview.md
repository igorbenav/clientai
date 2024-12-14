# API Reference Overview

Welcome to the API Reference section of ClientAI documentation. This section provides detailed information about the various classes, functions, and modules that make up ClientAI. Whether you're looking to integrate ClientAI into your project, extend its functionality, or simply explore its capabilities, this section will guide you through the intricacies of our codebase.

## Key Components

ClientAI's API is comprised of several key components, each serving a specific purpose:

1. **ClientAI Class**: This is the main class of our library. It provides a unified interface for interacting with different AI providers and is the primary entry point for using ClientAI.

    - [ClientAI Class Reference](client/clientai.md)

2. **AIProvider Class**: An abstract base class that defines the interface for all AI provider implementations. It ensures consistency across different providers.

    - [AIProvider Class Reference](client/ai_provider.md)

3. **Provider-Specific Classes**: These classes implement the AIProvider interface for each supported AI service (Ollama, OpenAI, Replicate, Groq).

    - [Ollama Provider Reference](client/specific_providers/ollama_provider.md)
    - [OpenAI Provider Reference](client/specific_providers/openai_provider.md)
    - [Replicate Provider Reference](client/specific_providers/replicate_provider.md)
    - [Groq Provider Reference](client/specific_providers/groq_provider.md)

4. **Ollama Manager**: These classes handle the local Ollama server configuration and lifecycle management.

    - [OllamaManager Class Reference](client/ollama_manager/ollama_manager.md)
    - [OllamaServerConfig Class Reference](client/ollama_manager/ollama_server_config.md)

## Usage

Each component is documented with its own dedicated page, where you can find detailed information about its methods, parameters, return types, and usage examples. These pages are designed to provide you with all the information you need to understand and work with ClientAI effectively.

### Basic Usage Example

Here's a quick example of how to use the main ClientAI class:

```python
from clientai import ClientAI

# Initialize the client
client = ClientAI('openai', api_key="your-openai-api-key")

# Generate text
response = client.generate_text(
    "Explain quantum computing",
    model="gpt-3.5-turbo"
)

print(response)
```

For more detailed usage instructions and examples, please refer to the [Usage Guide](../usage/overview.md).

## Contribution

We welcome contributions to ClientAI! If you're interested in contributing, please refer to our [Contributing Guidelines](../community/CONTRIBUTING.md). Contributions can range from bug fixes and documentation improvements to adding support for new AI providers.

## Feedback

Your feedback is crucial in helping us improve ClientAI and its documentation. If you have any suggestions, corrections, or queries, please don't hesitate to reach out to us via GitHub issues or our community channels.