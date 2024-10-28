# Error Handling and Retry Strategies

This guide covers best practices for handling errors and implementing retry strategies when working with ClientAI. Learn how to gracefully handle API errors, implement effective retry mechanisms, and build robust AI applications.

## Table of Contents
1. [Common Error Types](#common-error-types)
2. [Basic Error Handling](#basic-error-handling)
3. [Retry Strategies](#retry-strategies)
4. [Advanced Error Handling Patterns](#advanced-error-handling-patterns)
5. [Provider-Specific Considerations](#provider-specific-considerations)

## Common Error Types

ClientAI provides a unified error hierarchy for all providers:

```python
from clientai.exceptions import (
    ClientAIError,          # Base exception for all errors
    AuthenticationError,    # API key or auth issues
    RateLimitError,        # Rate limits exceeded
    InvalidRequestError,    # Malformed requests
    ModelError,            # Model-related issues
    TimeoutError,          # Request timeouts
    APIError              # General API errors
)
```

## Basic Error Handling

### Simple Try-Except Pattern
```python
from clientai import ClientAI
from clientai.exceptions import ClientAIError, RateLimitError

client = ClientAI("openai", api_key="your-api-key")

try:
    response = client.generate_text(
        prompt="Write a story",
        model="gpt-3.5-turbo"
    )
except RateLimitError as e:
    print(f"Rate limit hit. Status code: {e.status_code}")
    print(f"Original error: {e.original_error}")
except ClientAIError as e:
    print(f"Generation failed: {e}")
```

## Retry Strategies

### Simple Retry with Exponential Backoff
```python
import time
from typing import TypeVar, Callable
from clientai.exceptions import RateLimitError, TimeoutError

T = TypeVar('T')

def with_retry(
    operation: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0
) -> T:
    """
    Execute an operation with exponential backoff retry logic.
    
    Args:
        operation: Function to retry
        max_retries: Maximum retry attempts
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base for exponential backoff
        max_delay: Maximum delay between retries
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return operation()
        except (RateLimitError, TimeoutError) as e:
            last_exception = e
            if attempt == max_retries - 1:
                raise
                
            delay = min(
                initial_delay * (exponential_base ** attempt),
                max_delay
            )
            time.sleep(delay)
            
    raise last_exception or ClientAIError("Retry failed")

# Usage Example
def generate_text():
    return client.generate_text(
        prompt="Write a story",
        model="gpt-3.5-turbo"
    )

result = with_retry(generate_text)
```

### Provider-Aware Retry Strategy
```python
from typing import Dict, Optional

class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        retry_on: Optional[tuple] = None
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.retry_on = retry_on or (RateLimitError, TimeoutError)

PROVIDER_RETRY_CONFIGS = {
    "openai": RetryConfig(max_retries=3, initial_delay=1.0),
    "anthropic": RetryConfig(max_retries=5, initial_delay=2.0),
    "ollama": RetryConfig(max_retries=2, initial_delay=0.5)
}

def get_retry_config(provider: str) -> RetryConfig:
    """Get provider-specific retry configuration."""
    return PROVIDER_RETRY_CONFIGS.get(
        provider,
        RetryConfig()  # Default config
    )
```

## Advanced Error Handling Patterns

### Circuit Breaker Pattern
```python
from typing import Optional
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.is_open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = datetime.now()
        if self.failures >= self.failure_threshold:
            self.is_open = True

    def can_proceed(self) -> bool:
        if not self.is_open:
            return True

        if self.last_failure_time and \
           datetime.now() - self.last_failure_time > timedelta(seconds=self.reset_timeout):
            self.reset()
            return True
        
        return False

    def reset(self) -> None:
        self.failures = 0
        self.is_open = False
        self.last_failure_time = None

# Usage
circuit_breaker = CircuitBreaker()

def generate_with_circuit_breaker(prompt: str, model: str) -> str:
    if not circuit_breaker.can_proceed():
        raise ClientAIError("Circuit breaker is open")
        
    try:
        return client.generate_text(prompt, model=model)
    except ClientAIError as e:
        circuit_breaker.record_failure()
        raise
```

### Fallback Chain Pattern
```python
class FallbackChain:
    def __init__(self, default_response: Optional[str] = None):
        self.default_response = default_response
        self.handlers: list = []

    def add_handler(
        self,
        client: ClientAI,
        model: str,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.handlers.append((client, model, circuit_breaker))
        return self

    def execute(self, prompt: str) -> str:
        last_error = None
        
        for client, model, circuit_breaker in self.handlers:
            if circuit_breaker and not circuit_breaker.can_proceed():
                continue
                
            try:
                return client.generate_text(prompt, model=model)
            except ClientAIError as e:
                if circuit_breaker:
                    circuit_breaker.record_failure()
                last_error = e
                
        if self.default_response:
            return self.default_response
            
        raise last_error or ClientAIError("All handlers failed")

# Usage
fallback_chain = FallbackChain("Sorry, service unavailable")
fallback_chain.add_handler(
    ClientAI("openai"), "gpt-4", CircuitBreaker()
).add_handler(
    ClientAI("anthropic"), "claude-2", CircuitBreaker()
)

response = fallback_chain.execute("Write a story")
```

## Provider-Specific Considerations

### OpenAI
- Implements rate limiting with retry-after headers
- Supports automatic retries for intermittent errors
- Provides detailed error messages with status codes

### Anthropic
- Uses HTTP 429 for rate limits
- May require longer backoff periods
- Provides structured error responses

### Ollama
- Local deployment may have different error patterns
- Network errors more common than rate limits
- May require custom timeout configurations

## Best Practices

1. **Always Use Specific Exception Types**
   ```python
   try:
       response = client.generate_text(prompt, model)
   except RateLimitError:
       # Handle rate limits
   except ModelError:
       # Handle model issues
   except ClientAIError:
       # Handle other errors
   ```

2. **Implement Graceful Degradation**
   ```python
   def generate_with_fallback(prompt: str) -> str:
       try:
           return client.generate_text(
               prompt, model="gpt-4"
           )
       except (RateLimitError, ModelError):
           return client.generate_text(
               prompt, model="gpt-3.5-turbo"
           )
       except ClientAIError:
           return "Service temporarily unavailable"
   ```

3. **Use Appropriate Retry Strategies**
    - Implement exponential backoff
    - Respect rate limits and retry-after headers
    - Set reasonable timeout values
    - Use circuit breakers for system protection

4. **Log Errors Appropriately**
   ```python
   import logging
   
   logger = logging.getLogger(__name__)
   
   try:
       response = client.generate_text(prompt, model)
   except ClientAIError as e:
       logger.error(
           "Generation failed",
           extra={
               "status_code": e.status_code,
               "error_type": type(e).__name__,
               "original_error": str(e.original_error)
           }
       )
   ```

By following these error handling and retry strategies, you can build robust applications that gracefully handle failures and provide reliable service to your users.