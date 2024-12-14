# Replicate-Specific Parameters in ClientAI

This guide covers the Replicate-specific parameters that can be passed to ClientAI's `generate_text` and `chat` methods. These parameters are passed as additional keyword arguments to customize Replicate's behavior.

## generate_text Method

### Basic Structure
```python
from clientai import ClientAI

client = ClientAI('replicate', api_key="your-replicate-api-key")
response = client.generate_text(
    prompt="Your prompt here",     # Required
    model="owner/name:version",    # Required
    webhook="https://...",         # Replicate-specific
    webhook_completed="https://...",# Replicate-specific
    webhook_events_filter=[...],   # Replicate-specific
    stream=False,                  # Optional
    wait=True                      # Replicate-specific
)
```

### Replicate-Specific Parameters

#### `webhook: Optional[str]`
- URL to receive POST requests with prediction updates
```python
response = client.generate_text(
    prompt="Write a story",
    model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
    webhook="https://your-server.com/webhook"
)
```

#### `webhook_completed: Optional[str]`
- URL for receiving completion notifications
```python
response = client.generate_text(
    prompt="Generate text",
    model="meta/llama-2-70b:latest",
    webhook_completed="https://your-server.com/completed"
)
```

#### `webhook_events_filter: Optional[List[str]]`
- List of events that trigger webhooks
- Common events: `"completed"`, `"output"`
```python
response = client.generate_text(
    prompt="Analyze text",
    model="meta/llama-2-70b:latest",
    webhook_events_filter=["completed", "output"]
)
```

#### `wait: Optional[Union[int, bool]]`
- Controls request blocking behavior
- True: keeps request open up to 60 seconds
- int: specifies seconds to hold request (1-60)
- False: doesn't wait (default)
```python
response = client.generate_text(
    prompt="Complex analysis",
    model="meta/llama-2-70b:latest",
    wait=30  # Wait for 30 seconds
)
```

#### `stream: bool`
- Enables token streaming for supported models
```python
for chunk in client.generate_text(
    prompt="Write a story",
    model="meta/llama-2-70b:latest",
    stream=True
):
    print(chunk, end="")
```

## chat Method

### Basic Structure
```python
response = client.chat(
    model="meta/llama-2-70b:latest",  # Required
    messages=[...],                    # Required
    webhook="https://...",             # Replicate-specific
    webhook_completed="https://...",   # Replicate-specific
    webhook_events_filter=[...],       # Replicate-specific
    wait=True                          # Replicate-specific
)
```

### Message Formatting
Replicate formats chat messages into a single prompt:
```python
prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
prompt += "\nassistant: "
```

## Training Parameters

When using Replicate's training capabilities:

```python
response = client.train(
    model="stability-ai/sdxl",
    version="39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    input={
        "input_images": "https://domain/images.zip",
        "token_string": "TOK",
        "caption_prefix": "a photo of TOK",
        "max_train_steps": 1000,
        "use_face_detection_instead": False
    },
    destination="username/model-name"
)
```

## Complete Examples

### Example 1: Generation with Webhooks
```python
response = client.generate_text(
    prompt="Write a scientific paper summary",
    model="meta/llama-2-70b:latest",
    webhook="https://your-server.com/updates",
    webhook_completed="https://your-server.com/completed",
    webhook_events_filter=["completed"],
    wait=True
)
```

### Example 2: Chat with Streaming
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Write a haiku about coding"}
]

for chunk in client.chat(
    messages=messages,
    model="meta/llama-2-70b:latest",
    stream=True
):
    print(chunk, end="")
```

### Example 3: Image Generation
```python
response = client.generate_text(
    prompt="A portrait of a wombat gentleman",
    model="stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
    wait=60
)
```

## Error Handling

ClientAI maps Replicate's exceptions to its own error types:
```python
try:
    response = client.generate_text(
        prompt="Test prompt",
        model="meta/llama-2-70b:latest",
        wait=True
    )
except ClientAIError as e:
    print(f"Error: {e}")
```

Error mappings:
- `AuthenticationError`: API key issues
- `RateLimitError`: Rate limit exceeded
- `ModelError`: Model not found or failed
- `InvalidRequestError`: Invalid parameters
- `TimeoutError`: Request timeout (default 300s)
- `APIError`: Other server errors

## Parameter Validation Notes

1. Both `model` and `prompt`/`messages` are required
2. Model string format: `"owner/name:version"` or `"owner/name"` for latest version
3. `wait` must be boolean or integer 1-60
4. Webhook URLs must be valid HTTP/HTTPS URLs
5. `webhook_events_filter` must contain valid event types
6. Some models may not support streaming
7. File inputs can be URLs or local file paths

These parameters allow you to leverage Replicate's features through ClientAI, including model management, webhook notifications, and streaming capabilities.