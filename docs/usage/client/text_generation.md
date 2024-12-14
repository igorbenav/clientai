# Text Generation with ClientAI

This guide explores how to use ClientAI for text generation tasks across different AI providers. You'll learn about the various options and parameters available for generating text.

## Table of Contents

1. [Basic Text Generation](#basic-text-generation)
2. [Advanced Parameters](#advanced-parameters)
3. [Streaming Responses](#streaming-responses)
4. [Provider-Specific Features](#provider-specific-features)
5. [Best Practices](#best-practices)

## Basic Text Generation

To generate text using ClientAI, use the `generate_text` method:

```python
from clientai import ClientAI

client = ClientAI('openai', api_key="your-openai-api-key")

response = client.generate_text(
    "Write a short story about a robot learning to paint.",
    model="gpt-3.5-turbo"
)

print(response)
```

This will generate a short story based on the given prompt.

## Advanced Parameters

ClientAI supports various parameters to fine-tune text generation:

```python
response = client.generate_text(
    "Explain the theory of relativity",
    model="gpt-4",
    max_tokens=150,
    temperature=0.7,
    top_p=0.9,
    presence_penalty=0.1,
    frequency_penalty=0.1
)
```

- `max_tokens`: Maximum number of tokens to generate
- `temperature`: Controls randomness (0.0 to 1.0)
- `top_p`: Nucleus sampling parameter
- `presence_penalty`: Penalizes new tokens based on their presence in the text so far
- `frequency_penalty`: Penalizes new tokens based on their frequency in the text so far

Note: Available parameters may vary depending on the provider.

## Streaming Responses

For long-form content, you can use streaming to get partial responses as they're generated:

```python
for chunk in client.generate_text(
    "Write a comprehensive essay on climate change",
    model="gpt-3.5-turbo",
    stream=True
):
    print(chunk, end="", flush=True)
```

This allows for real-time display of generated text, which can be useful for user interfaces or long-running generations.

## Provider-Specific Features

Different providers may offer unique features. Here are some examples:

### OpenAI

```python
response = openai_client.generate_text(
    "Translate the following to French: 'Hello, how are you?'",
    model="gpt-3.5-turbo"
)
```

### Replicate

```python
response = replicate_client.generate_text(
    "Generate a haiku about mountains",
    model="meta/llama-2-70b-chat:latest"
)
```

### Ollama

```python
response = ollama_client.generate_text(
    "Explain the concept of neural networks",
    model="llama2"
)
```

## Best Practices

1. **Prompt Engineering**: Craft clear and specific prompts for better results.

   ```python
   good_prompt = "Write a detailed description of a futuristic city, focusing on transportation and architecture."
   ```

2. **Model Selection**: Choose appropriate models based on your task complexity and requirements.

3. **Error Handling**: Always handle potential errors in text generation:

   ```python
   try:
       response = client.generate_text("Your prompt here", model="gpt-3.5-turbo")
   except Exception as e:
       print(f"An error occurred: {e}")
   ```

4. **Rate Limiting**: Be mindful of rate limits imposed by providers. Implement appropriate delays or queuing mechanisms for high-volume applications.

5. **Content Filtering**: Implement content filtering or moderation for user-facing applications to ensure appropriate outputs.

6. **Consistency**: For applications requiring consistent outputs, consider using lower temperature values or implementing your own post-processing.

By following these guidelines and exploring the various parameters and features available, you can effectively leverage ClientAI for a wide range of text generation tasks across different AI providers.