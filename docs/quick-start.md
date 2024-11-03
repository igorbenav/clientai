# Quickstart

This guide will help you get started with ClientAI quickly. We'll cover the basic setup and usage for each supported AI provider.

## Minimal Example

Here's a minimal example to get you started with ClientAI:

```python title="quickstart.py"
from clientai import ClientAI

# Initialize the client (example with OpenAI)
client = ClientAI('openai', api_key="your-openai-api-key")

# Generate text
response = client.generate_text(
    "Tell me a joke",
    model="gpt-3.5-turbo",
)
print(response)

# Use chat functionality
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What's its population?"}
]
response = client.chat(messages, model="gpt-3.5-turbo")
print(response)
```

## Setup for Different Providers

### OpenAI

```python title="openai_setup.py" hl_lines="4"
from clientai import ClientAI

# Initialize the OpenAI client
client = ClientAI('openai', api_key="your-openai-api-key")

# Now you can use the client for text generation or chat
```

### Replicate

```python title="replicate_setup.py" hl_lines="4"
from clientai import ClientAI

# Initialize the Replicate client
client = ClientAI('replicate', api_key="your-replicate-api-key")

# Now you can use the client for text generation or chat
```

### Ollama

```python title="ollama_setup.py" hl_lines="4"
from clientai import ClientAI

# Initialize the Ollama client
client = ClientAI('ollama', host="your-ollama-host")

# Now you can use the client for text generation or chat
```

### Groq

```python title="groq_setup.py" hl_lines="4"
from clientai import ClientAI

# Initialize the Groq client
client = ClientAI('groq', host="your-ollama-host")

# Now you can use the client for text generation or chat
```

## Basic Usage

Once you have initialized the client, you can use it for text generation and chat functionality:

### Text Generation

```python title="text_generation.py" hl_lines="6-10"
from clientai import ClientAI

client = ClientAI('openai', api_key="your-openai-api-key")

# Generate text
response = client.generate_text(
    "Explain the concept of quantum computing",
    model="gpt-3.5-turbo",
    max_tokens=100
)
print(response)
```

### Chat

```python title="chat.py" hl_lines="6-15"
from clientai import ClientAI

client = ClientAI('openai', api_key="your-openai-api-key")

# Use chat functionality
messages = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a branch of artificial intelligence..."},
    {"role": "user", "content": "Can you give an example of its application?"}
]
response = client.chat(
    messages,
    model="gpt-3.5-turbo",
    max_tokens=150
)
print(response)
```

### Ollama Server Management

If you're running Ollama locally, ClientAI provides a convenient way to manage the Ollama server:

```python title="ollama_manager.py"
from clientai.ollama import OllamaManager

# Start and automatically stop the server using a context manager
with OllamaManager() as manager:
    # Server is now running
    client = ClientAI('ollama')
    response = client.generate_text("Hello, world!", model="llama2")
    print(response)
```

You can also configure basic server settings:

```python
from clientai.ollama import OllamaManager, OllamaServerConfig

config = OllamaServerConfig(
    host="127.0.0.1",
    port=11434,
    gpu_layers=35  # Optional: Number of layers to run on GPU
)

with OllamaManager(config) as manager:
    # Your code here
    pass
```

## Next Steps

Now that you've seen the basics of ClientAI, you can:

1. Explore more advanced features like streaming responses and handling full response objects.
2. Check out the [Usage Guide](usage/overview.md) for detailed information on all available methods and options.
3. See the [API Reference](api/overview.md) for a complete list of ClientAI's classes and methods.

Remember to handle API keys securely and never expose them in your code or version control systems.