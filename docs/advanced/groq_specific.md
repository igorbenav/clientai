# Groq-Specific Parameters in ClientAI

This guide covers the Groq-specific parameters that can be passed to ClientAI's `generate_text` and `chat` methods. These parameters are passed as additional keyword arguments to customize Groq's behavior.

## generate_text Method

### Basic Structure
```python
from clientai import ClientAI

client = ClientAI('groq', api_key="your-groq-api-key")
response = client.generate_text(
    prompt="Your prompt here",          # Required
    model="llama3-8b-8192",            # Required
    frequency_penalty=0.5,              # Groq-specific
    presence_penalty=0.2,               # Groq-specific
    max_tokens=100,                     # Groq-specific
    response_format={"type": "json"},   # Groq-specific
    seed=12345,                        # Groq-specific
    temperature=0.7,                    # Groq-specific
    top_p=0.9,                         # Groq-specific
    n=1,                               # Groq-specific
    stop=["END"],                      # Groq-specific
    stream=False,                      # Groq-specific
    stream_options=None,               # Groq-specific
    functions=None,                    # Groq-specific (Deprecated)
    function_call=None,                # Groq-specific (Deprecated)
    tools=None,                        # Groq-specific
    tool_choice=None,                  # Groq-specific
    parallel_tool_calls=True,          # Groq-specific
    user="user_123"                    # Groq-specific
)
```

### Groq-Specific Parameters

#### `frequency_penalty: Optional[float]`
- Range: -2.0 to 2.0
- Default: 0
- Penalizes tokens based on their frequency in the text
```python
response = client.generate_text(
    prompt="Write a creative story",
    model="llama3-8b-8192",
    frequency_penalty=0.7  # Reduces repetition
)
```

#### `presence_penalty: Optional[float]`
- Range: -2.0 to 2.0
- Default: 0
- Penalizes tokens based on their presence in prior text
```python
response = client.generate_text(
    prompt="Write a varied story",
    model="llama3-8b-8192",
    presence_penalty=0.6  # Encourages topic diversity
)
```

#### `max_tokens: Optional[int]`
- Maximum tokens for completion
- Limited by model's context length
```python
response = client.generate_text(
    prompt="Write a summary",
    model="llama3-8b-8192",
    max_tokens=100
)
```

#### `response_format: Optional[Dict]`
- Controls output structure
- Requires explicit JSON instruction in prompt
```python
response = client.generate_text(
    prompt="List three colors in JSON",
    model="llama3-8b-8192",
    response_format={"type": "json_object"}
)
```

#### `seed: Optional[int]`
- For deterministic generation
```python
response = client.generate_text(
    prompt="Generate a random number",
    model="llama3-8b-8192",
    seed=12345
)
```

#### `temperature: Optional[float]`
- Range: 0 to 2
- Default: 1
- Controls randomness in output
```python
response = client.generate_text(
    prompt="Write creatively",
    model="llama3-8b-8192",
    temperature=0.7  # More creative output
)
```

#### `top_p: Optional[float]`
- Range: 0 to 1
- Default: 1
- Alternative to temperature, called nucleus sampling
```python
response = client.generate_text(
    prompt="Generate text",
    model="llama3-8b-8192",
    top_p=0.1  # Only consider top 10% probability tokens
)
```

#### `n: Optional[int]`
- Default: 1
- Number of completions to generate
- Note: Currently only n=1 is supported
```python
response = client.generate_text(
    prompt="Generate a story",
    model="llama3-8b-8192",
    n=1
)
```

#### `stop: Optional[Union[str, List[str]]]`
- Up to 4 sequences where generation stops
```python
response = client.generate_text(
    prompt="Write until you see END",
    model="llama3-8b-8192",
    stop=["END", "STOP"]  # Stops at either sequence
)
```

#### `stream: Optional[bool]`
- Default: False
- Enable token streaming
```python
for chunk in client.generate_text(
    prompt="Tell a story",
    model="llama3-8b-8192",
    stream=True
):
    print(chunk, end="", flush=True)
```

#### `stream_options: Optional[Dict]`
- Options for streaming responses
- Only used when stream=True
```python
response = client.generate_text(
    prompt="Long story",
    model="llama3-8b-8192",
    stream=True,
    stream_options={"chunk_size": 1024}
)
```

#### `user: Optional[str]`
- Unique identifier for end-user tracking
```python
response = client.generate_text(
    prompt="Hello",
    model="llama3-8b-8192",
    user="user_123"
)
```

## chat Method

### Basic Structure
```python
response = client.chat(
    model="llama3-8b-8192",            # Required
    messages=[...],                    # Required
    tools=[...],                      # Groq-specific
    tool_choice="auto",               # Groq-specific
    parallel_tool_calls=True,         # Groq-specific
    response_format={"type": "json"}, # Groq-specific
    temperature=0.7,                  # Groq-specific
    frequency_penalty=0.5,            # Groq-specific
    presence_penalty=0.2,             # Groq-specific
    max_tokens=100,                   # Groq-specific
    seed=12345,                      # Groq-specific
    stop=["END"],                    # Groq-specific
    stream=False,                    # Groq-specific
    stream_options=None,             # Groq-specific
    top_p=0.9,                       # Groq-specific
    n=1,                             # Groq-specific
    user="user_123"                  # Groq-specific
)
```

### Groq-Specific Parameters

#### `tools: Optional[List[Dict]]`
- List of available tools (max 128)
```python
response = client.chat(
    model="llama3-70b-8192",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather data",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }]
)
```

#### `tool_choice: Optional[Union[str, Dict]]`
- Controls tool selection behavior
- Values: "none", "auto", "required"
```python
response = client.chat(
    model="llama3-70b-8192",
    messages=[{"role": "user", "content": "Calculate something"}],
    tool_choice="auto"  # or "none" or "required"
)
```

#### `parallel_tool_calls: Optional[bool]`
- Default: True
- Enable parallel function calling
```python
response = client.chat(
    model="llama3-70b-8192",
    messages=[{"role": "user", "content": "Multiple tasks"}],
    parallel_tool_calls=True
)
```

## Complete Examples

### Example 1: Structured Output with Tools
```python
response = client.chat(
    model="llama3-70b-8192",
    messages=[
        {"role": "system", "content": "You are a data assistant"},
        {"role": "user", "content": "Get weather for Paris"}
    ],
    response_format={"type": "json_object"},
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather data",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }],
    tool_choice="auto",
    temperature=0.7,
    max_tokens=200,
    seed=42
)
```

### Example 2: Advanced Text Generation
```python
response = client.generate_text(
    prompt="Write a technical analysis",
    model="mixtral-8x7b-32768",
    max_tokens=500,
    frequency_penalty=0.7,
    presence_penalty=0.6,
    temperature=0.4,
    top_p=0.9,
    stop=["END", "CONCLUSION"],
    user="analyst_1",
    seed=42
)
```

### Example 3: Streaming Generation
```python
for chunk in client.chat(
    model="llama3-8b-8192",
    messages=[{"role": "user", "content": "Explain quantum physics"}],
    stream=True,
    temperature=0.7,
    max_tokens=1000,
    stream_options={"chunk_size": 1024}
):
    print(chunk, end="", flush=True)
```

## Parameter Validation Notes

1. Both `model` and `prompt`/`messages` are required
2. Model must be one of: "gemma-7b-it", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"
3. `n` parameter only supports value of 1
4. `stop` sequences limited to 4 maximum
5. Tool usage limited to 128 functions
6. `response_format` requires explicit JSON instruction in prompt
7. Parameters like `logprobs`, `logit_bias`, and `top_logprobs` are not yet supported
8. Deterministic generation with `seed` is best-effort
9. `functions` and `function_call` are deprecated in favor of `tools` and `tool_choice`

These parameters allow you to fully customize Groq's behavior while working with ClientAI's abstraction layer.