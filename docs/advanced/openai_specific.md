# OpenAI-Specific Parameters in ClientAI

This guide covers the OpenAI-specific parameters that can be passed to ClientAI's `generate_text` and `chat` methods. These parameters are passed as additional keyword arguments to customize OpenAI's behavior.

## generate_text Method

### Basic Structure
```python
from clientai import ClientAI

client = ClientAI('openai', api_key="your-openai-api-key")
response = client.generate_text(
    prompt="Your prompt here",          # Required
    model="gpt-3.5-turbo",             # Required
    frequency_penalty=0.5,              # OpenAI-specific
    presence_penalty=0.2,               # OpenAI-specific
    logit_bias={123: 100},             # OpenAI-specific
    max_completion_tokens=100,          # OpenAI-specific
    response_format={"type": "json"},   # OpenAI-specific
    seed=12345                         # OpenAI-specific
)
```

### OpenAI-Specific Parameters

#### `frequency_penalty: Optional[float]`
- Range: -2.0 to 2.0
- Penalizes tokens based on their frequency in the text
```python
response = client.generate_text(
    prompt="Write a creative story",
    model="gpt-3.5-turbo",
    frequency_penalty=0.7  # Reduces repetition
)
```

#### `presence_penalty: Optional[float]`
- Range: -2.0 to 2.0
- Penalizes tokens based on their presence in prior text
```python
response = client.generate_text(
    prompt="Write a varied story",
    model="gpt-3.5-turbo",
    presence_penalty=0.6  # Encourages topic diversity
)
```

#### `logit_bias: Optional[Dict[str, int]]`
- Maps token IDs to bias values (-100 to 100)
```python
response = client.generate_text(
    prompt="Write about technology",
    model="gpt-3.5-turbo",
    logit_bias={
        123: 100,  # Increases likelihood of token 123
        456: -100  # Decreases likelihood of token 456
    }
)
```

#### `max_completion_tokens: Optional[int]`
- Maximum tokens for completion
```python
response = client.generate_text(
    prompt="Write a summary",
    model="gpt-3.5-turbo",
    max_completion_tokens=100
)
```

#### `response_format: ResponseFormat`
- Controls output structure
```python
response = client.generate_text(
    prompt="List three colors",
    model="gpt-4",
    response_format={"type": "json_object"}
)
```

#### `seed: Optional[int]`
- For deterministic generation (Beta)
```python
response = client.generate_text(
    prompt="Generate a random number",
    model="gpt-3.5-turbo",
    seed=12345
)
```

#### `user: str`
- Unique identifier for end-user tracking
```python
response = client.generate_text(
    prompt="Hello",
    model="gpt-3.5-turbo",
    user="user_123"
)
```

## chat Method

### Basic Structure
```python
response = client.chat(
    model="gpt-3.5-turbo",            # Required
    messages=[...],                    # Required
    tools=[...],                      # OpenAI-specific
    tool_choice="auto",               # OpenAI-specific
    response_format={"type": "json"}, # OpenAI-specific
    logprobs=True,                    # OpenAI-specific
    top_logprobs=5                    # OpenAI-specific
)
```

### OpenAI-Specific Parameters

#### `tools: Iterable[ChatCompletionToolParam]`
- List of available tools (max 128)
```python
response = client.chat(
    model="gpt-4",
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

#### `tool_choice: ChatCompletionToolChoiceOptionParam`
- Controls tool selection behavior
```python
response = client.chat(
    model="gpt-4",
    messages=[{"role": "user", "content": "Calculate something"}],
    tool_choice="auto"  # or "none" or "required"
)
```

#### `modalities: Optional[List[ChatCompletionModality]]`
- Output types for generation
```python
response = client.chat(
    model="gpt-4o-audio-preview",
    messages=[{"role": "user", "content": "Generate audio"}],
    modalities=["text", "audio"]
)
```

#### `audio: Optional[ChatCompletionAudioParam]`
- Audio output parameters
```python
response = client.chat(
    model="gpt-4o-audio-preview",
    messages=[{"role": "user", "content": "Speak this"}],
    modalities=["audio"],
    audio={"model": "tts-1", "voice": "alloy"}
)
```

#### `metadata: Optional[Dict[str, str]]`
- Custom tags for filtering
```python
response = client.chat(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={"purpose": "greeting", "user_type": "new"}
)
```

## Complete Examples

### Example 1: Structured Output with Tools
```python
response = client.chat(
    model="gpt-4",
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
    tool_choice="auto"
)
```

### Example 2: Advanced Text Generation
```python
response = client.generate_text(
    prompt="Write a technical analysis",
    model="gpt-4",
    max_completion_tokens=500,
    frequency_penalty=0.7,
    presence_penalty=0.6,
    logit_bias={123: 50},
    user="analyst_1",
    seed=42
)
```

### Example 3: Audio Generation
```python
response = client.chat(
    model="gpt-4o-audio-preview",
    messages=[{"role": "user", "content": "Explain quantum physics"}],
    modalities=["text", "audio"],
    audio={
        "model": "tts-1",
        "voice": "nova",
        "speed": 1.0
    },
    metadata={"type": "educational"}
)
```

## Parameter Validation Notes

1. Both `model` and `prompt`/`messages` are required
2. `response_format` requires compatible models
3. Tool usage limited to 128 functions
4. Audio generation requires specific models
5. `logprobs` must be True when using `top_logprobs`
6. `seed` feature is in Beta and not guaranteed

These parameters allow you to fully customize OpenAI's behavior while working with ClientAI's abstraction layer.