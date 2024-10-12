# ClientAI

<p align="center">
  <a href="https://igorbenav.github.io/clientai/">
    <img src="assets/ClientAI.png" alt="ClientAI logo" width="45%" height="auto">
  </a>
</p>

<p align="center">
  <i>A unified client for seamless interaction with multiple AI providers.</i>
</p>

<p align="center">
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

---

**ClientAI** is a Python package that provides a unified interface for interacting with multiple AI providers, including OpenAI, Replicate, and Ollama. It offers seamless integration and consistent methods for text generation and chat functionality across different AI platforms.

**Documentation**: [igorbenav.github.io/clientai/](https://igorbenav.github.io/clientai/)

---

## Features

- 🔄 **Unified Interface**: Consistent methods for text generation and chat across multiple AI providers.
- 🔌 **Multiple Providers**: Support for OpenAI, Replicate, and Ollama, with easy extensibility for future providers.
- 🌊 **Streaming Support**: Efficient streaming of responses for real-time applications.
- 🎛️ **Flexible Configuration**: Easy setup with provider-specific configurations.
- 🔧 **Customizable**: Extensible design for adding new providers or customizing existing ones.
- 🧠 **Type Hinting**: Comprehensive type annotations for better development experience.
- 🔒 **Provider Isolation**: Optional installation of provider-specific dependencies to keep your environment lean.

## Requirements

Before installing ClientAI, ensure you have the following:

- **Python**: Version 3.9 or newer.
- **Dependencies**: The core ClientAI package has minimal dependencies. Provider-specific packages (e.g., `openai`, `replicate`, `ollama`) are optional and can be installed separately.

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

ClientAI provides a simple and consistent way to interact with different AI providers. Here are some examples:

### Initializing the Client

```python
from clientai import ClientAI

# Initialize with OpenAI
openai_client = ClientAI('openai', api_key="your-openai-key")

# Initialize with Replicate
replicate_client = ClientAI('replicate', api_key="your-replicate-key")

# Initialize with Ollama
ollama_client = ClientAI('ollama', host="your-ollama-host")
```

### Generating Text

```python
# Using OpenAI
response = openai_client.generate_text(
    "Tell me a joke",
    model="gpt-3.5-turbo",
)

# Using Replicate
response = replicate_client.generate_text(
    "Explain quantum computing",
    model="meta/llama-2-70b-chat:latest",
)

# Using Ollama
response = ollama_client.generate_text(
    "What is the capital of France?",
    model="llama2",
)
```

### Chat Functionality

```python
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "What is its population?"}
]

# Using OpenAI
response = openai_client.chat(
    messages,
    model="gpt-3.5-turbo",
)

# Using Replicate
response = replicate_client.chat(
    messages,
    model="meta/llama-2-70b-chat:latest",
)

# Using Ollama
response = ollama_client.chat(
    messages,
    model="llama2",
)
```

### Streaming Responses

```python
for chunk in client.generate_text(
    "Tell me a long story",
    model="gpt-3.5-turbo",
    stream=True
):
    print(chunk, end="", flush=True)
```

## Contributing

Contributions to ClientAI are welcome! Please refer to our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Igor Magalhaes – [@igormagalhaesr](https://twitter.com/igormagalhaesr) – igormagalhaesr@gmail.com
[github.com/igorbenav](https://github.com/igorbenav/)