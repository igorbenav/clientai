# Complete Guide to Output Validation in ClientAI

When working with AI models, getting structured, reliable outputs can be challenging. ClientAI Agents offer two ways to help you tame those outputs: JSON formatting and Pydantic validation. Let's explore how to use them effectively.

## Understanding Your Options

You have three ways to handle agent outputs:

1. Regular text output (default)
2. JSON-formatted output (`json_output=True`)
3. Validated output with Pydantic models (`json_output=True` with `return_type`)

Let's look at when to use each approach.

## Simple Text Output

When you just need text responses, use the default configuration:

```python
class SimpleAgent(Agent):
    @think("analyze")
    def analyze_text(self, input_text: str) -> str:
        return f"Please analyze this text: {input_text}"

# Usage
agent = SimpleAgent(client=client, default_model="gpt-4")
result = agent.run("Hello world")  # Returns plain text
```

This is perfect for general text generation, summaries, or when you don't need structured data.

## JSON-Formatted Output

When you need structured data but don't want strict validation, use `json_output=True`:

```python
class StructuredAgent(Agent):
    @think(
        name="analyze",
        json_output=True  # Ensures JSON output
    )
    def analyze_data(self, input_data: str) -> str:
        return """
        Analyze this data and return as JSON with these fields:
        - summary: brief overview
        - key_points: list of main points
        - sentiment: positive, negative, or neutral
        
        Data: {input_data}
        """

# Usage
agent = StructuredAgent(client=client, default_model="gpt-4")
result = agent.run("Great product, highly recommend!")
# Returns parsed JSON like:
# {
#     "summary": "Positive product review",
#     "key_points": ["Strong recommendation", "General satisfaction"],
#     "sentiment": "positive"
# }
```

This approach gives you structured data while maintaining flexibility in the output format.

## Validated Output with Pydantic

When you need guaranteed output structure and type safety, combine `json_output` with Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ProductAnalysis(BaseModel):
    summary: str = Field(
        min_length=10,
        description="Brief overview of the analysis"
    )
    key_points: List[str] = Field(
        min_items=1,
        description="Main points from the analysis"
    )
    sentiment: str = Field(
        pattern="^(positive|negative|neutral)$",
        description="Overall sentiment"
    )
    confidence: float = Field(
        ge=0, le=1,
        description="Confidence score between 0 and 1"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Product categories if mentioned"
    )

class ValidatedAgent(Agent):
    @think(
        name="analyze",
        json_output=True,  # Required for validation
        return_type=ProductAnalysis  # Enables Pydantic validation
    )
    def analyze_review(self, review: str) -> ProductAnalysis:
        return """
        Analyze this product review and return a JSON object with:
        - summary: at least 10 characters
        - key_points: non-empty list of strings
        - sentiment: exactly "positive", "negative", or "neutral"
        - confidence: number between 0 and 1
        - categories: optional list of product categories
        
        Review: {review}
        """

# Usage
agent = ValidatedAgent(client=client, default_model="gpt-4")
try:
    result = agent.run("This laptop is amazing! Great battery life and performance.")
    print(f"Summary: {result.summary}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence}")
    for point in result.key_points:
        print(f"- {point}")
except ValidationError as e:
    print("Output validation failed:", e)
```

### Working with Complex Validation

For advanced validation needs, you can create sophisticated validation schemas that handle complex data structures and relationships:

```python
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class AnalysisMetrics(BaseModel):
    """Validated structure for analysis metrics"""
    accuracy: float = Field(ge=0, le=1, description="Analysis accuracy score")
    confidence: float = Field(ge=0, le=1, description="Confidence in results")
    relevance: float = Field(ge=0, le=1, description="Relevance score")

class DetailedAnalysis(BaseModel):
    """Complex validated analysis output"""
    timestamp: datetime = Field(description="When analysis was performed")
    metrics: AnalysisMetrics = Field(description="Analysis quality metrics")
    categories: List[str] = Field(min_items=1, description="Found categories")
    keywords: List[str] = Field(min_items=3, max_items=10)
    summary: str = Field(min_length=50, max_length=500)
    metadata: Optional[Dict[str, str]] = None

class AdvancedAgent(Agent):
    @think(
        name="analyze",
        json_output=True,
        return_type=DetailedAnalysis
    )
    def deep_analysis(self, content: str) -> DetailedAnalysis:
        """Perform detailed content analysis with validated output."""
        return f"""
        Analyze this content and return results matching the DetailedAnalysis schema:
        - timestamp: current UTC time
        - metrics: accuracy, confidence, and relevance scores (0-1)
        - categories: at least one relevant category
        - keywords: 3-10 key terms found
        - summary: 50-500 character summary
        - metadata: optional additional information
        
        Content: {content}
        """

# Usage with complex validation
agent = AdvancedAgent(client=client, default_model="gpt-4")

try:
    result = agent.run("Climate change impacts on global agriculture...")
    
    # Access validated fields with type safety
    print(f"Analysis timestamp: {result.timestamp}")
    print(f"Confidence score: {result.metrics.confidence}")
    print(f"Found categories: {', '.join(result.categories)}")
    
    if result.metadata:  # Handle optional field
        for key, value in result.metadata.items():
            print(f"{key}: {value}")
            
except ValidationError as e:
    print("Validation failed:", e)
```

### Cross-Field Validation

Sometimes you need to validate relationships between fields. Here's how to do that effectively:

```python
from typing import List
from pydantic import BaseModel, Field

class DataRange(BaseModel):
    min_value: float
    max_value: float
    values: List[float]
    
    def model_post_init(self, __context) -> None:
        """Validate relationships between fields"""
        # Ensure range is valid
        if self.max_value <= self.min_value:
            raise ValueError("max_value must be greater than min_value")
        
        # Validate all values fall within range
        out_of_range = [
            v for v in self.values 
            if v < self.min_value or v > self.max_value
        ]
        if out_of_range:
            raise ValueError(f"Values {out_of_range} outside of range")

class RangeAnalysisAgent(Agent):
    @think(
        name="analyze",
        json_output=True,
        return_type=DataRange
    )
    def analyze_range(self, data: str) -> DataRange:
        return f"""
        Analyze this data and provide:
        - min_value: minimum acceptable value
        - max_value: maximum acceptable value
        - values: list of values to validate
        
        Ensure max_value > min_value and all values within range.
        
        Data: {data}
        """

# Usage
agent = RangeAnalysisAgent(client=client, default_model="gpt-4")

# This will pass validation
result = agent.run("Range: 0-100, Values: 45, 67, 82")

# This will fail validation (values out of range)
try:
    result = agent.run("Range: 0-50, Values: 45, 67, 82")
except ValidationError as e:
    print(e)  # Values [67, 82] outside of range
```

## Retry Handling with Validation

When working with validated outputs, you might want to retry failed attempts. ClientAI provides built-in retry capabilities that work seamlessly with validation:

```python
from clientai.agent.config import StepConfig

class RetryAgent(Agent):
    @think(
        name="analyze",
        json_output=True,
        return_type=ProductAnalysis,
        step_config=StepConfig(
            retry_count=3,  # Number of retry attempts
            use_internal_retry=True  # Use ClientAI's retry mechanism
        )
    )
    def analyze_with_retry(self, data: str) -> ProductAnalysis:
        return """
        Analyze this data and return as JSON matching ProductAnalysis schema.
        If validation fails, the step will be retried up to 3 times.
        
        Data: {data}
        """

# Usage with retry handling
agent = RetryAgent(client=client, default_model="gpt-4")
try:
    result = agent.run("Product review: Good value but slow delivery")
    print(f"Analysis successful after retries: {result}")
except ValidationError as e:
    print("All retry attempts failed validation:", e)
```

### Configuring Retry Behavior

You can configure retry behavior at different levels:

1. **Step Level** - Using StepConfig:
```python
@think(
    "analyze",
    json_output=True,
    return_type=AnalysisResult,
    step_config=StepConfig(
        retry_count=2,  # Retry twice
        required=True,  # Step must succeed
        timeout=30.0,  # Timeout per attempt
        use_internal_retry=True  # Use built-in retry
    )
)
def analyze_data(self, data: str) -> AnalysisResult:
    return "Analyze with retries: {data}"
```

2. **Optional Steps** - Allow continuing on failure:
```python
@think(
    "enrich",
    json_output=True,
    return_type=EnrichmentData,
    step_config=StepConfig(
        retry_count=1,
        required=False  # Continue workflow if step fails
    )
)
def enrich_data(self, data: str) -> EnrichmentData:
    return "Attempt to enrich: {data}"
```

### Best Practices for Retries

1. **Selective Retry Usage:**
    - Enable retries for critical steps that must succeed
    - Consider making non-critical steps optional with `required=False`
    - Set appropriate retry counts based on operation importance

2. **Timeout Configuration:**
    - Set reasonable timeouts to prevent long-running retries
    - Consider your model's typical response time
    - Balance between giving enough time and failing fast

3. **Error Handling with Retries:**
```python
try:
    result = agent.run("Process this data")
except ValidationError as e:
    print("Validation failed after all retries:", e)
except WorkflowError as e:
    print("Workflow failed:", e)
    if hasattr(e, "__cause__"):
        print("Caused by:", e.__cause__)
```

4. **Monitoring Retry Behavior:**
    - Log retry attempts and failures for debugging
    - Track retry patterns to optimize settings
    - Consider alerting on high retry rates

## Important Considerations

1. **Compatibility Notes:**
    - You can't use `json_output=True` or validation with streaming responses (`stream=True`)
    - Validation isn't compatible with `return_full_response=True`
    - When using a Pydantic model as `return_type`, `json_output=True` is automatically enabled

2. **Best Practices:**
    - Always include clear field descriptions in your Pydantic models
    - Use type hints and Field validators to enforce constraints
    - Provide explicit output format instructions in your prompts
    - Handle validation errors appropriately in your application
    - Consider using optional fields for non-critical data

3. **Error Handling:**
    - Always wrap validated calls in try/except blocks
    - ValidationError will provide detailed information about what went wrong
    - Consider logging validation failures for debugging

## Making the Choice

- Use **plain text** when you need simple text responses
- Use **JSON output** when you need basic structure without strict validation
- Use **Pydantic validation** when you need guaranteed structure and type safety

Remember that validation choices affect both reliability and flexibility:

- More validation = More reliability but less flexibility
- Less validation = More flexibility but less reliability

Choose the appropriate level of validation based on your specific needs and reliability requirements.

## Next Steps

Now that you've mastered validation, check out our [Examples](../../examples/overview.md) section to see these concepts in action:

- How to build a simple Q&A bot that introduces core agent features
- Techniques for creating a task planner using `create_agent` and basic tools
- Methods for implementing a writing assistant with multi-step workflows
- Patterns for developing a sophisticated code analyzer with custom workflows

Each example demonstrates different aspects of ClientAI, from basic agent creation to complex systems combining steps, tools, and context management. Start with the [Simple Q&A Bot](../../examples/agent/simple_qa.md) to see ClientAI's fundamentals in practice, or jump straight to the [Code Analyzer](../../examples/agent/code_analyzer.md) for a more advanced implementation.