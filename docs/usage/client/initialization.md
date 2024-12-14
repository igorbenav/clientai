# Initializing ClientAI

This guide covers the process of initializing ClientAI with different AI providers. You'll learn how to set up ClientAI for use with OpenAI, Replicate, and Ollama.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [OpenAI Initialization](#openai-initialization)
3. [Replicate Initialization](#replicate-initialization)
4. [Ollama Initialization](#ollama-initialization)
5. [Multiple Provider Initialization](#multiple-provider-initialization)
6. [Best Practices](#best-practices)

## Prerequisites

Before initializing ClientAI, ensure you have:

1. Installed ClientAI: `pip install clientai[all]`
2. Obtained necessary API keys for the providers you plan to use
3. Basic understanding of Python and asynchronous programming

## OpenAI Initialization

To initialize ClientAI with OpenAI:

```python
from clientai import ClientAI

openai_client = ClientAI('openai', api_key="your-openai-api-key")
```

Replace `"your-openai-api-key"` with your actual OpenAI API key.

## Replicate Initialization

To initialize ClientAI with Replicate:

```python
from clientai import ClientAI

replicate_client = ClientAI('replicate', api_key="your-replicate-api-key")
```

Replace `"your-replicate-api-key"` with your actual Replicate API key.

## Ollama Initialization

To initialize ClientAI with Ollama:

```python
from clientai import ClientAI

ollama_client = ClientAI('ollama', host="http://localhost:11434")
```

Ensure that you have Ollama running locally on the specified host.

## Groq Initialization

To initialize ClientAI with Groq:

```python
from clientai import ClientAI

replicate_client = ClientAI('groq', api_key="your-groq-api-key")
```

## Multiple Provider Initialization

You can initialize multiple providers in the same script:

```python
from clientai import ClientAI

openai_client = ClientAI('openai', api_key="your-openai-api-key")
replicate_client = ClientAI('replicate', api_key="your-replicate-api-key")
groq_client = ClientAI('groq', api_key="your-groq-api-key")
ollama_client = ClientAI('ollama', host="http://localhost:11434")
```

## Best Practices

1. **Environment Variables**: Store API keys in environment variables instead of hardcoding them in your script:

   ```python
   import os
   from clientai import ClientAI

   openai_client = ClientAI('openai', api_key=os.getenv('OPENAI_API_KEY'))
   ```

2. **Error Handling**: Wrap initialization in a try-except block to handle potential errors:

   ```python
   try:
       client = ClientAI('openai', api_key="your-openai-api-key")
   except ValueError as e:
       print(f"Error initializing ClientAI: {e}")
   ```

3. **Configuration Files**: For projects with multiple providers, consider using a configuration file:

   ```python
   import json
   from clientai import ClientAI

   with open('config.json') as f:
       config = json.load(f)

   openai_client = ClientAI('openai', **config['openai'])
   replicate_client = ClientAI('replicate', **config['replicate'])
   ```

4. **Lazy Initialization**: If you're not sure which provider you'll use, initialize clients only when needed:

   ```python
   def get_client(provider):
       if provider == 'openai':
           return ClientAI('openai', api_key="your-openai-api-key")
       elif provider == 'replicate':
           return ClientAI('replicate', api_key="your-replicate-api-key")
       # ... other providers ...
   
   # Use the client when needed
   client = get_client('openai')
   ```

By following these initialization guidelines, you'll be well-prepared to start using ClientAI with various AI providers in your projects.