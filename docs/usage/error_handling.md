# Error Handling in ClientAI

ClientAI provides a robust error handling system that unifies exceptions across different AI providers. This guide covers how to handle potential errors when using ClientAI.

## Table of Contents

1. [Exception Hierarchy](#exception-hierarchy)
2. [Handling Errors](#handling-errors)
3. [Provider-Specific Error Mapping](#provider-specific-error-mapping)
4. [Best Practices](#best-practices)

## Exception Hierarchy

ClientAI uses a custom exception hierarchy to provide consistent error handling across different AI providers:

```python
from clientai.exceptions import (
    ClientAIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelError,
    TimeoutError,
    APIError
)
```

- `ClientAIError`: Base exception class for all ClientAI errors.
- `AuthenticationError`: Raised when there's an authentication problem with the AI provider.
- `RateLimitError`: Raised when the AI provider's rate limit is exceeded.
- `InvalidRequestError`: Raised when the request to the AI provider is invalid.
- `ModelError`: Raised when there's an issue with the specified model.
- `TimeoutError`: Raised when a request to the AI provider times out.
- `APIError`: Raised when there's an API-related error from the AI provider.

## Handling Errors

Here's how to handle potential errors when using ClientAI:

```python
from clientai import ClientAI
from clientai.exceptions import (
    ClientAIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelError,
    TimeoutError,
    APIError
)

client = ClientAI('openai', api_key="your-openai-api-key")

try:
    response = client.generate_text("Tell me a joke", model="gpt-3.5-turbo")
    print(f"Generated text: {response}")
except AuthenticationError as e:
    print(f"Authentication error: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except InvalidRequestError as e:
    print(f"Invalid request: {e}")
except ModelError as e:
    print(f"Model error: {e}")
except TimeoutError as e:
    print(f"Request timed out: {e}")
except APIError as e:
    print(f"API error: {e}")
except ClientAIError as e:
    print(f"An unexpected ClientAI error occurred: {e}")
```

## Provider-Specific Error Mapping

ClientAI maps provider-specific errors to its custom exception hierarchy. For example:

### OpenAI

```python
def _map_exception_to_clientai_error(self, e: Exception) -> None:
    error_message = str(e)
    status_code = getattr(e, 'status_code', None)

    if isinstance(e, OpenAIAuthenticationError) or "incorrect api key" in error_message.lower():
        raise AuthenticationError(error_message, status_code, original_error=e)
    elif status_code == 429 or "rate limit" in error_message.lower():
        raise RateLimitError(error_message, status_code, original_error=e)
    elif status_code == 404 or "not found" in error_message.lower():
        raise ModelError(error_message, status_code, original_error=e)
    elif status_code == 400 or "invalid" in error_message.lower():
        raise InvalidRequestError(error_message, status_code, original_error=e)
    elif status_code == 408 or "timeout" in error_message.lower():
        raise TimeoutError(error_message, status_code, original_error=e)
    elif status_code and status_code >= 500:
        raise APIError(error_message, status_code, original_error=e)
    
    raise ClientAIError(error_message, status_code, original_error=e)
```

### Replicate

```python
def _map_exception_to_clientai_error(self, e: Exception, status_code: int = None) -> ClientAIError:
    error_message = str(e)
    status_code = status_code or getattr(e, 'status_code', None)

    if "authentication" in error_message.lower() or "unauthorized" in error_message.lower():
        return AuthenticationError(error_message, status_code, original_error=e)
    elif "rate limit" in error_message.lower():
        return RateLimitError(error_message, status_code, original_error=e)
    elif "not found" in error_message.lower():
        return ModelError(error_message, status_code, original_error=e)
    elif "invalid" in error_message.lower():
        return InvalidRequestError(error_message, status_code, original_error=e)
    elif "timeout" in error_message.lower() or status_code == 408:
        return TimeoutError(error_message, status_code, original_error=e)
    elif status_code == 400:
        return InvalidRequestError(error_message, status_code, original_error=e)
    else:
        return APIError(error_message, status_code, original_error=e)
```

### Groq

```python
def _map_exception_to_clientai_error(self, e: Exception) -> ClientAIError:
    error_message = str(e)

    if isinstance(e, (GroqAuthenticationError | PermissionDeniedError)):
        return AuthenticationError(
            error_message,
            status_code=getattr(e, "status_code", 401),
            original_error=e,
        )
    elif isinstance(e, GroqRateLimitError):
        return RateLimitError(error_message, status_code=429, original_error=e)
    elif isinstance(e, NotFoundError):
        return ModelError(error_message, status_code=404, original_error=e)
    elif isinstance(e, (BadRequestError | UnprocessableEntityError | ConflictError)):
        return InvalidRequestError(
            error_message,
            status_code=getattr(e, "status_code", 400),
            original_error=e,
        )
    elif isinstance(e, APITimeoutError):
        return TimeoutError(error_message, status_code=408, original_error=e)
    elif isinstance(e, InternalServerError):
        return APIError(
            error_message,
            status_code=getattr(e, "status_code", 500),
            original_error=e,
        )
    elif isinstance(e, APIStatusError):
        status = getattr(e, "status_code", 500)
        if status >= 500:
            return APIError(error_message, status_code=status, original_error=e)
        return InvalidRequestError(error_message, status_code=status, original_error=e)

    return ClientAIError(error_message, status_code=500, original_error=e)
```

## Best Practices

1. **Specific Exception Handling**: Catch specific exceptions when you need to handle them differently.

2. **Logging**: Log errors for debugging and monitoring purposes.

   ```python
   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   try:
       response = client.generate_text("Tell me a joke", model="gpt-3.5-turbo")
   except ClientAIError as e:
       logger.error(f"An error occurred: {e}", exc_info=True)
   ```

3. **Retry Logic**: Implement retry logic for transient errors like rate limiting.

   ```python
   import time
   from clientai.exceptions import RateLimitError

   def retry_generate(prompt, model, max_retries=3, delay=1):
       for attempt in range(max_retries):
           try:
               return client.generate_text(prompt, model=model)
           except RateLimitError as e:
               if attempt == max_retries - 1:
                   raise
               wait_time = e.retry_after if hasattr(e, 'retry_after') else delay * (2 ** attempt)
               logger.warning(f"Rate limit reached. Waiting for {wait_time} seconds...")
               time.sleep(wait_time)
   ```

4. **Graceful Degradation**: Implement fallback options when errors occur.

   ```python
   def generate_with_fallback(prompt, primary_client, fallback_client):
       try:
           return primary_client.generate_text(prompt, model="gpt-3.5-turbo")
       except ClientAIError as e:
           logger.warning(f"Primary client failed: {e}. Falling back to secondary client.")
           return fallback_client.generate_text(prompt, model="llama-2-70b-chat")
   ```

By following these practices and utilizing ClientAI's unified error handling system, you can create more robust and maintainable applications that gracefully handle errors across different AI providers.