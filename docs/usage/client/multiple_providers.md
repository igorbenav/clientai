# Working with Multiple Providers in ClientAI

This guide explores techniques for effectively using multiple AI providers within a single project using ClientAI. You'll learn how to switch between providers and leverage their unique strengths.

## Table of Contents

1. [Setting Up Multiple Providers](#setting-up-multiple-providers)
2. [Switching Between Providers](#switching-between-providers)
3. [Leveraging Provider Strengths](#leveraging-provider-strengths)
4. [Load Balancing and Fallback Strategies](#load-balancing-and-fallback-strategies)
5. [Best Practices](#best-practices)

## Setting Up Multiple Providers

First, initialize ClientAI with multiple providers:

```python
from clientai import ClientAI

openai_client = ClientAI('openai', api_key="your-openai-api-key")
replicate_client = ClientAI('replicate', api_key="your-replicate-api-key")
ollama_client = ClientAI('ollama', host="http://localhost:11434")
```

## Switching Between Providers

Create a function to switch between providers based on your requirements:

```python
def get_provider(task):
    if task == "translation":
        return openai_client
    elif task == "code_generation":
        return replicate_client
    elif task == "local_inference":
        return ollama_client
    else:
        return openai_client  # Default to OpenAI

# Usage
task = "translation"
provider = get_provider(task)
response = provider.generate_text("Translate 'Hello' to French", model="gpt-3.5-turbo")
```

This approach allows you to dynamically select the most appropriate provider for each task.

## Leveraging Provider Strengths

Different providers excel in different areas. Here's how you can leverage their strengths:

```python
def translate_text(text, target_language):
    return openai_client.generate_text(
        f"Translate '{text}' to {target_language}",
        model="gpt-3.5-turbo"
    )

def generate_code(prompt):
    return replicate_client.generate_text(
        prompt,
        model="meta/llama-2-70b-chat:latest"
    )

def local_inference(prompt):
    return ollama_client.generate_text(
        prompt,
        model="llama2"
    )

# Usage
french_text = translate_text("Hello, world!", "French")
python_code = generate_code("Write a Python function to calculate the Fibonacci sequence")
quick_response = local_inference("What's the capital of France?")
```

## Load Balancing and Fallback Strategies

Implement load balancing and fallback strategies to ensure reliability:

```python
import random

providers = [openai_client, replicate_client, ollama_client]

def load_balanced_generate(prompt, max_retries=3):
    for _ in range(max_retries):
        try:
            provider = random.choice(providers)
            return provider.generate_text(prompt, model=provider.default_model)
        except Exception as e:
            print(f"Error with provider {provider.__class__.__name__}: {e}")
    raise Exception("All providers failed after max retries")

# Usage
try:
    response = load_balanced_generate("Explain the concept of machine learning")
    print(response)
except Exception as e:
    print(f"Failed to generate text: {e}")
```

This function randomly selects a provider and falls back to others if there's an error.

## Best Practices

1. **Provider Selection Logic**: Develop clear criteria for selecting providers based on task requirements, cost, and performance.

   ```python
   def select_provider(task, complexity, budget):
       if complexity == "high" and budget == "high":
           return openai_client  # Assuming OpenAI has more advanced models
       elif task == "code" and budget == "medium":
           return replicate_client
       else:
           return ollama_client  # Assuming Ollama is the most cost-effective
   ```

2. **Consistent Interface**: Create wrapper functions to provide a consistent interface across providers:

   ```python
   def unified_generate(prompt, provider=None):
       if provider is None:
           provider = get_default_provider()
       return provider.generate_text(prompt, model=provider.default_model)

   # Usage
   response = unified_generate("Explain quantum computing")
   ```

3. **Error Handling and Logging**: Implement comprehensive error handling and logging when working with multiple providers:

   ```python
   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   def safe_generate(prompt, provider):
       try:
           return provider.generate_text(prompt, model=provider.default_model)
       except Exception as e:
           logger.error(f"Error with {provider.__class__.__name__}: {e}")
           return None
   ```

4. **Performance Monitoring**: Track the performance of different providers to optimize selection:

   ```python
   import time

   def timed_generate(prompt, provider):
       start_time = time.time()
       result = provider.generate_text(prompt, model=provider.default_model)
       elapsed_time = time.time() - start_time
       logger.info(f"{provider.__class__.__name__} took {elapsed_time:.2f} seconds")
       return result
   ```

5. **Configuration Management**: Use configuration files or environment variables to manage provider settings:

   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()

   openai_client = ClientAI('openai', api_key=os.getenv('OPENAI_API_KEY'))
   replicate_client = ClientAI('replicate', api_key=os.getenv('REPLICATE_API_KEY'))
   ollama_client = ClientAI('ollama', host=os.getenv('OLLAMA_HOST'))
   ```

6. **Caching**: Implement caching to reduce redundant API calls and improve response times:

   ```python
   from functools import lru_cache

   @lru_cache(maxsize=100)
   def cached_generate(prompt, provider_name):
       provider = get_provider(provider_name)
       return provider.generate_text(prompt, model=provider.default_model)

   # Usage
   response = cached_generate("What is the speed of light?", "openai")
   ```

By following these practices and leveraging the strengths of multiple providers, you can create more robust, efficient, and versatile applications with ClientAI.