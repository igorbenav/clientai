# Error Handling and Retry Strategies

This guide covers best practices for handling errors and implementing retry strategies when working with ClientAI. Learn how to gracefully handle API errors, implement effective retry mechanisms, and build robust AI applications.

## Table of Contents
1. [Common Error Types](#common-error-types)
2. [Client Error Handling](#client-error-handling)
3. [Agent Error Handling](#agent-error-handling)
4. [Retry Strategies](#retry-strategies)
5. [Advanced Error Handling Patterns](#advanced-error-handling-patterns)

## Common Error Types

ClientAI provides a unified error hierarchy that normalizes errors across all providers. This means you can handle errors consistently regardless of which AI provider you're using. 

### Client-Level Exceptions

```python
from clientai.exceptions import (
    ClientAIError,          # Base exception for all client errors
    AuthenticationError,    # API key or auth issues
    APIError,              # General API errors
    RateLimitError,        # Rate limits exceeded
    InvalidRequestError,    # Malformed requests
    ModelError,            # Model-related issues
    ProviderNotInstalledError,  # Missing provider package
    TimeoutError           # Request timeouts
)
```

Each exception includes:
- A descriptive message
- Optional HTTP status code
- Optional reference to the original error

### Agent-Level Exceptions

```python
from clientai.agent.exceptions import (
    AgentError,    # Base exception for all agent errors
    StepError,     # Step execution/validation errors
    ToolError      # Tool execution/validation errors
)
```

## Client Error Handling

At the client level, error handling focuses on direct interactions with AI providers. The most common pattern is to handle specific exceptions first, followed by more general ones:

```python
from clientai import ClientAI
from clientai.exceptions import (
    ClientAIError, 
    RateLimitError,
    AuthenticationError,
    ModelError
)

client = ClientAI("openai", api_key="your-api-key")

try:
    response = client.generate_text(
        prompt="Write a story",
        model="gpt-3.5-turbo"
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    print(f"Status code: {e.status_code}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
    print(f"Original error: {e.original_error}")
except ModelError as e:
    print(f"Model error: {e}")
except ClientAIError as e:
    print(f"Generation failed: {e}")
```

## Agent Error Handling

Agent error handling builds on client error handling but adds considerations for workflow steps and tools. Here's how to handle agent-specific errors:

```python
from clientai.agent import Agent, think
from clientai.agent.exceptions import StepError, ToolError, AgentError

class ErrorAwareAgent(Agent):
    @think("analyze")
    def analyze_data(self, data: str) -> str:
        try:
            # Attempt to use tool
            result = self.use_tool("analyzer", data=data)
            return f"Analysis result: {result}"
        except ToolError:
            # Fallback to direct prompt if tool fails
            return f"Please analyze this data: {data}"

# Using the agent
agent = ErrorAwareAgent(client=client, default_model="gpt-4")

try:
    result = agent.run("Analyze this data")
except StepError as e:
    print(f"Step execution failed: {e}")
except ToolError as e:
    print(f"Tool execution failed: {e}")
except AgentError as e:
    print(f"Agent error: {e}")
```

## Retry Strategies

ClientAI supports several approaches to implementing retries:

### 1. Internal Agent Retries

The agent system provides built-in retry capabilities through step configuration:

```python
from clientai.agent.config import StepConfig

class RetryAgent(Agent):
    @think(
        "analyze",
        step_config=StepConfig(
            retry_count=3,
            timeout=30.0
            # use_internal_retry is True by default
        )
    )
    def analyze_data(self, data: str) -> str:
        return f"Please analyze this data: {data}"
```

### 2. External Retry Libraries

For more complex retry patterns, you can use external retry libraries like Stamina. When using external retry mechanisms, disable internal retries:

```python
import stamina

class StaminaAgent(Agent):
    @think(
        "analyze",
        step_config=StepConfig(
            use_internal_retry=False  # Disable internal retry
        )
    )
    @stamina.retry(
        on=(RateLimitError, TimeoutError),
        attempts=3
    )
    def analyze_data(self, data: str) -> str:
        return f"Please analyze this data: {data}"
```

## Agent Error Handling and Step Configuration

The agent system provides fine-grained control over error handling through step configuration. Each step can be configured to handle failures differently, enabling robust workflows that gracefully handle errors.

### Step Configuration and Error Flow

Steps can be marked as required or optional, affecting how errors propagate through the workflow:

```python
from clientai.agent import Agent, think, act
from clientai.agent.config import StepConfig
from clientai.exceptions import ModelError, ToolError

class AnalysisAgent(Agent):
    # Required step - failure stops workflow
    @think(
        "analyze",
        step_config=StepConfig(
            required=True,         # Workflow fails if this step fails
            retry_count=3,         # Retry up to 3 times
            timeout=30.0          # 30 second timeout
        )
    )
    def analyze_data(self, data: str) -> str:
        return f"Please perform critical analysis of: {data}"

    # Optional step - workflow continues if it fails
    @think(
        "enhance",
        step_config=StepConfig(
            required=False,        # Workflow continues if this fails
            retry_count=1,         # Try once more on failure
            pass_result=False      # Don't update context on failure
        )
    )
    def enhance_analysis(self, analysis: str) -> str:
        return f"Please enhance this analysis: {analysis}"

    # Final required step
    @act(
        "summarize",
        step_config=StepConfig(
            required=True,
            pass_result=True      # Pass result to next step
        )
    )
    def summarize_results(self, enhanced: str) -> str:
        return f"Please summarize these results: {enhanced}"

# Usage
agent = AnalysisAgent(client=client, default_model="gpt-4")
try:
    result = agent.run("Sample data")
except StepError as e:
    if "analyze" in str(e):
        print("Critical analysis failed")
    elif "summarize" in str(e):
        print("Summary generation failed")
    else:
        print(f"Step failed: {e}")
```

### Result Passing and Error State

The `pass_result` parameter controls how results flow between steps, especially important during error handling:

```python
class DataProcessingAgent(Agent):
    @think(
        "analyze",
        step_config=StepConfig(
            required=True,
            pass_result=True      # Success: result passed to next step
        )
    )
    def analyze_data(self, data: str) -> str:
        return f"Analyze: {data}"

    @think(
        "validate",
        step_config=StepConfig(
            required=False,        # Optional validation
            pass_result=False,     # Don't pass failed validation results
            retry_count=2         # Retry validation twice
        )
    )
    def validate_analysis(self, analysis: str) -> str:
        try:
            result = self.use_tool("validator", data=analysis)
            return f"Validation result: {result}"
        except ToolError:
            # Let it fail but don't pass result
            raise

    @act(
        "process",
        step_config=StepConfig(
            required=True,
            pass_result=True
        )
    )
    def process_results(self, validated_or_original: str) -> str:
        # Gets original analysis if validation failed
        return f"Process: {validated_or_original}"
```

### Retry Configuration Patterns

Here's how to combine retry settings with required and optional steps:

```python
class ResilientAgent(Agent):
    # Critical step with extensive retries
    @think(
        "critical_analysis",
        step_config=StepConfig(
            required=True,
            retry_count=5,
            timeout=45.0,
            use_internal_retry=True
        )
    )
    def analyze_critical(self, data: str) -> str:
        return f"Critical analysis: {data}"

    # Optional enhancement with limited retries
    @think(
        "enhance",
        step_config=StepConfig(
            required=False,
            retry_count=2,
            timeout=15.0,
            use_internal_retry=True
        )
    )
    def enhance_results(self, analysis: str) -> str:
        return f"Enhance: {analysis}"

    # Final step with external retry (e.g., Stamina)
    @think(
        "summarize",
        step_config=StepConfig(
            required=True,
            use_internal_retry=False  # Using external retry
        )
    )
    @stamina.retry(
        on=ModelError,
        attempts=3
    )
    def summarize(self, results: str) -> str:
        return f"Summarize: {results}"
```

### Graceful Degradation with Optional Steps

Here's a complete example of implementing graceful degradation using optional steps:

```python
class DegradingAnalysisAgent(Agent):
    @think(
        "detailed_analysis",
        step_config=StepConfig(
            required=False,        # Optional - can fail
            retry_count=2,
            pass_result=False      # Don't pass failed results
        )
    )
    def analyze_detailed(self, data: str) -> str:
        try:
            # Try complex analysis first
            result = self.use_tool("complex_analyzer", data=data)
            return f"Detailed analysis: {result}"
        except ToolError:
            raise

    @think(
        "basic_analysis",
        step_config=StepConfig(
            required=False,        # Optional fallback
            retry_count=1,
            pass_result=True      # Pass results if successful
        )
    )
    def analyze_basic(self, data: str) -> str:
        try:
            # Simpler analysis as fallback
            result = self.use_tool("basic_analyzer", data=data)
            return f"Basic analysis: {result}"
        except ToolError:
            raise

    @think(
        "minimal_analysis",
        step_config=StepConfig(
            required=True,         # Must succeed
            pass_result=True,
            retry_count=3
        )
    )
    def analyze_minimal(self, data: str) -> str:
        # Minimal analysis - just use LLM
        return f"Please provide a minimal analysis of: {data}"

# Usage showing graceful degradation
agent = DegradingAnalysisAgent(client=client, default_model="gpt-4")

try:
    result = agent.run("Complex dataset")
    # Will try detailed -> basic -> minimal,
    # using the best successful analysis
except StepError as e:
    if "minimal_analysis" in str(e):
        print("Even minimal analysis failed")
    else:
        print(f"Unexpected failure: {e}")
```

## Advanced Error Handling Patterns

### Circuit Breaker Pattern

The circuit breaker pattern prevents system overload by temporarily stopping operations after a series of failures:

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

# Usage with agent
class CircuitBreakerAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit_breaker = CircuitBreaker()

    @think("analyze")
    def analyze_data(self, data: str) -> str:
        if not self.circuit_breaker.can_proceed():
            return "Service temporarily unavailable"
        
        try:
            result = self.use_tool("analyzer", data=data)
            return f"Analysis result: {result}"
        except ToolError as e:
            self.circuit_breaker.record_failure()
            raise
```

### Fallback Chain Pattern

```python
from typing import Optional, List, Tuple

class FallbackChain:
    def __init__(self, default_response: Optional[str] = None):
        self.default_response = default_response
        self.handlers: List[Tuple[ClientAI, str, Optional[CircuitBreaker]]] = []

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

# Usage with agent
class FallbackAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_chain = FallbackChain(
            default_response="Unable to process request"
        )
        self.fallback_chain.add_handler(
            ClientAI("openai", api_key=OPENAI_API_KEY), "gpt-4", CircuitBreaker()
        ).add_handler(
            ClientAI("groq", api_key=GROQ_API_KEY), "llama-3.1-70b-versatile", CircuitBreaker()
        )

    @think("analyze")
    def analyze_data(self, data: str) -> str:
        return self.fallback_chain.execute(
            f"Please analyze this data: {data}"
        )
```

## Best Practices

1. **Use Specific Exception Types**
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

3. **Log Errors Appropriately**
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

4. **Choose the Right Retry Strategy**
    - Use internal retries for simple cases
    - Use external retry libraries for complex patterns
    - Never mix internal and external retries
    - Consider provider-specific characteristics

By following these error handling and retry strategies, you can build robust applications that handle failures gracefully and provide reliable service to your users.