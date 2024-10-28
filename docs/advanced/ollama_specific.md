# Ollama-Specific Parameters in ClientAI

This guide covers the Ollama-specific parameters that can be passed to ClientAI's `generate_text` and `chat` methods. These parameters are passed as additional keyword arguments to customize Ollama's behavior.

## generate_text Method

### Basic Structure
```python
from clientai import ClientAI

client = ClientAI('ollama')
response = client.generate_text(
    prompt="Your prompt here",    # Required
    model="llama2",              # Required
    suffix="Optional suffix",     # Ollama-specific
    system="System message",      # Ollama-specific
    template="Custom template",   # Ollama-specific
    context=[1, 2, 3],           # Ollama-specific
    format="json",               # Ollama-specific
    options={"temperature": 0.7}, # Ollama-specific
    keep_alive="5m"              # Ollama-specific
)
```

### Ollama-Specific Parameters

#### `suffix: str`
- Text to append to the generated output
```python
response = client.generate_text(
    prompt="Write a story about a robot",
    model="llama2",
    suffix="\n\nThe End."
)
```

#### `system: str`
- System message to guide the model's behavior
```python
response = client.generate_text(
    prompt="Explain quantum computing",
    model="llama2",
    system="You are a quantum physics professor explaining concepts to beginners"
)
```

#### `template: str`
- Custom prompt template
```python
response = client.generate_text(
    prompt="What is Python?",
    model="llama2",
    template="Question: {{.Prompt}}\n\nDetailed answer:"
)
```

#### `context: List[int]`
- Context from previous interactions
```python
# First request
first_response = client.generate_text(
    prompt="Tell me a story about space",
    model="llama2"
)

# Continue the story using context
continued_response = client.generate_text(
    prompt="What happened next?",
    model="llama2",
    context=first_response.context  # Context from previous response
)
```

#### `format: Literal['', 'json']`
- Controls response format
```python
response = client.generate_text(
    prompt="List three fruits with their colors",
    model="llama2",
    format="json"
)
```

#### `options: Optional[Options]`
- Model-specific parameters
```python
response = client.generate_text(
    prompt="Write a creative story",
    model="llama2",
    options={
        "temperature": 0.9,
        "top_p": 0.8,
        "top_k": 40
    }
)
```

#### `keep_alive: Optional[Union[float, str]]`
- Model memory retention duration
```python
response = client.generate_text(
    prompt="Quick calculation",
    model="llama2",
    keep_alive="10m"  # Keep model loaded for 10 minutes
)
```

## chat Method

### Basic Structure
```python
response = client.chat(
    model="llama2",              # Required
    messages=[...],              # Required
    tools=[...],                 # Ollama-specific
    format="json",               # Ollama-specific
    options={"temperature": 0.7}, # Ollama-specific
    keep_alive="5m"              # Ollama-specific
)
```

### Ollama-Specific Parameters

#### `tools: Optional[List[Dict]]`
- Tools available for the model (requires stream=False)
```python
response = client.chat(
    model="llama2",
    messages=[{"role": "user", "content": "What's 2+2?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic math",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }
        }
    }],
    stream=False
)
```

#### `format: Literal['', 'json']`
- Controls response format
```python
response = client.chat(
    model="llama2",
    messages=[
        {"role": "user", "content": "List three countries with their capitals"}
    ],
    format="json"
)
```

#### `options: Optional[Options]`
- Model-specific parameters
```python
response = client.chat(
    model="llama2",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    options={
        "temperature": 0.8,
        "top_p": 0.9,
        "presence_penalty": 0.5
    }
)
```

#### `keep_alive: Optional[Union[float, str]]`
- Model memory retention duration
```python
response = client.chat(
    model="llama2",
    messages=[{"role": "user", "content": "Hello"}],
    keep_alive=300.0  # 5 minutes in seconds
)
```

## Complete Examples

### Example 1: Creative Writing with generate_text
```python
response = client.generate_text(
    prompt="Write a short story about AI",
    model="llama2",
    system="You are a creative writer specializing in science fiction",
    template="Story prompt: {{.Prompt}}\n\nCreative story:",
    options={
        "temperature": 0.9,
        "top_p": 0.95
    },
    suffix="\n\nThe End.",
    keep_alive="10m"
)
```

### Example 2: JSON Response with chat
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant that provides structured data"},
    {"role": "user", "content": "List 3 programming languages with their key features"}
]

response = client.chat(
    model="llama2",
    messages=messages,
    format="json",
    options={
        "temperature": 0.3,  # Lower temperature for more structured output
        "top_p": 0.9
    }
)
```

### Example 3: Multimodal Chat with Image
```python
messages = [
    {
        "role": "user",
        "content": "What's in this image?",
        "images": ["encoded_image_data_or_path"]
    }
]

response = client.chat(
    model="llava",
    messages=messages,
    format="json",
    keep_alive="5m"
)
```

### Example 4: Contextual Generation
```python
# First generation
first_response = client.generate_text(
    prompt="Write the beginning of a mystery story",
    model="llama2",
    options={"temperature": 0.8}
)

# Continue the story using context
continued_response = client.generate_text(
    prompt="Continue the story with a plot twist",
    model="llama2",
    context=first_response.context,
    options={"temperature": 0.8}
)
```

## Parameter Validation Notes

1. Both `model` and `prompt`/`messages` are required
2. When using `tools`, `stream` must be `False`
3. `format` only accepts `''` or `'json'`
4. Image support requires multimodal models (e.g., llava)
5. Context preservation works only with `generate_text`
6. Keep alive duration can be string (e.g., "5m") or float (seconds)

These parameters allow you to fully customize Ollama's behavior while working with ClientAI's abstraction layer.