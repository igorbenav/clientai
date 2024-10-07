<style>
    .md-typeset h1,
    .md-content__button {
        display: none;
    }
</style>

<p align="center">
  <a href="https://github.com/igorbenav/clientai">
    <img src="assets/clientai.png?raw=true" alt="ClientAI logo" width="45%" height="auto">
  </a>
</p>
<p align="center" markdown=1>
  <i>A unified client for seamless interaction with multiple AI providers.</i>
</p>
<p align="center" markdown=1>
<a href="https://github.com/igorbenav/clientai/actions/workflows/tests.yml">
  <img src="https://github.com/igorbenav/clientai/actions/workflows/tests.yml/badge.svg" alt="Tests"/>
</a>
<a href="https://pypi.org/project/clientai/">
  <img src="https://img.shields.io/pypi/v/clientai?color=%2334D058&label=pypi%20package" alt="PyPi Version"/>
</a>
<a href="https://pypi.org/project/clientai/">
  <img src="https://img.shields.io/pypi/pyversions/clientai.svg?color=%2334D058" alt="Supported Python Versions"/>
</a>
</p>
<hr>
<p align="justify">
<b>ClientAI</b> is a Python package that provides a unified interface for interacting with multiple AI providers, including OpenAI, Replicate, and Ollama. It offers seamless integration and consistent methods for text generation and chat functionality across different AI platforms.
</p>
<hr>

## Features

- **Unified Interface**: Consistent methods for text generation and chat across multiple AI providers.
- **Multiple Providers**: Support for OpenAI, Replicate, and Ollama, with easy extensibility for future providers.
- **Streaming Support**: Efficient streaming of responses for real-time applications.
- **Flexible Configuration**: Easy setup with provider-specific configurations.
- **Customizable**: Extensible design for adding new providers or customizing existing ones.
- **Type Hinting**: Comprehensive type annotations for better development experience.
- **Provider Isolation**: Optional installation of provider-specific dependencies to keep your environment lean.

## Minimal Example

Here's a quick example to get you started with ClientAI:

```python
from clientai import ClientAI

# Initialize with OpenAI
client = ClientAI('openai', api_key="your-openai-key")

# Generate text
response = client.generate_text(
    "Tell me a joke",
    model="gpt-3.5-turbo",
)

print(response)

# Chat functionality
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "What is its population?"}
]

response = client.chat(
    messages,
    model="gpt-3.5-turbo",
)

print(response)
```

## Requirements

Before installing ClientAI, ensure you have the following prerequisites:

* **Python:** Version 3.9 or newer.
* **Dependencies:** The core ClientAI package has minimal dependencies. Provider-specific packages (e.g., `openai`, `replicate`, `ollama`) are optional and can be installed separately.

## Installing

To install ClientAI with all providers, run:

```sh
pip install clientai[all]
```

Or, if you prefer to install only specific providers:

```sh
pip install clientai[openai]  # For OpenAI support
pip install clientai[replicate]  # For Replicate support
pip install clientai[ollama]  # For Ollama support
```

## Usage

ClientAI offers a consistent way to interact with different AI providers:

1. Initialize the client with your chosen provider and credentials.
2. Use the `generate_text` method for text generation tasks.
3. Use the `chat` method for conversational interactions.

Both methods support streaming responses and returning full response objects.

For more detailed usage examples and advanced features, please refer to the Usage section of this documentation.

## License

[`MIT`](community/LICENSE.md)