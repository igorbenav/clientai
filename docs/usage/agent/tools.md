# Tools and Tool Selection

Tools in ClientAI are functions that extend an agent's capabilities beyond language model interactions. The tool system provides:

- Function registration with automatic signature analysis
- Automated tool selection based on context
- Scoped tool availability for different workflow steps
- Comprehensive error handling and validation

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Understanding Tools](#understanding-tools)
3. [Creating Tools](#creating-tools)
4. [Tool Registration](#tool-registration)
5. [Tool Selection Configuration](#tool-selection-configuration)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)

## Prerequisites

Before working with tools, ensure you have:

1. A basic understanding of:
    - Python type hints and decorators
    - Function documentation practices
    - Basic ClientAI agent concepts

2. Proper imports:
    ```python
    from clientai.agent import Agent, tool
    from clientai.agent.tools import ToolSelectionConfig
    ```

## Understanding Tools

Tools in ClientAI are functions with:

- Clear type hints for parameters and return values
- Descriptive docstrings explaining functionality
- Optional configuration for usage scopes and selection criteria

### Tool Characteristics

Good tools should be:

- Focused on a single task
- Well-documented with clear inputs/outputs
- Error-handled appropriately
- Stateless when possible

## Creating Tools

There are two main ways to create tools:

### 1. Using the @tool Decorator
```python
@tool(name="Calculator", description="Performs basic arithmetic")
def add_numbers(x: int, y: int) -> int:
    """Add two numbers together.
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        Sum of the two numbers
    """
    return x + y
```

### 2. Direct Tool Creation
```python
from clientai.agent.tools import Tool

def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y

multiply_tool = Tool.create(
    func=multiply,
    name="Multiplier",
    description="Multiplies two numbers together"
)
```

## Tool Registration

Tools can be registered with agents in several ways:

### 1. During Agent Creation
```python
agent = MyAgent(
    client=client,
    default_model="gpt-4",
    tools=[
        calculator_tool,
        formatter_tool,
        ToolConfig(
            tool=multiply,
            scopes=["think", "act"],
            name="Multiplier"
        )
    ]
)
```

### 2. Using register_tool Method
```python
# Register with specific scopes
agent.register_tool(
    tool=process_text,
    name="TextProcessor",
    description="Processes text input",
    scopes=["think", "synthesize"]
)

# Register for all scopes (default)
agent.register_tool(
    tool=calculate,
    name="Calculator",
    description="Performs calculations"
)
```

### 3. Using register_tool as a Decorator
```python
# Register with specific scopes
@agent.register_tool(
    name="Calculator",
    description="Performs calculations",
    scopes=["think", "act"]
)
def calculate(x: int, y: int) -> int:
    return x + y

# Register for all scopes
@agent.register_tool(
    name="TextFormatter",
    description="Formats text"
)
def format_text(text: str) -> str:
    return text.upper()
```

## Tool Selection Configuration

Tool selection is a key feature that allows agents to automatically choose and use appropriate tools based on the task at hand. The selection process is powered by LLMs and can be customized to meet specific needs.

### Basic Configuration

Tool selection behavior can be customized using ToolSelectionConfig:

```python
config = ToolSelectionConfig(
    confidence_threshold=0.8,    # Minimum confidence for tool selection
    max_tools_per_step=3,        # Maximum tools per step
    prompt_template="Custom selection prompt: {task}\nTools: {tool_descriptions}"
)

agent = MyAgent(
    client=client,
    default_model="gpt-4",
    tool_selection_config=config,
    tool_model="gpt-4"          # Model for tool selection decisions
)
```

### Understanding Tool Selection

When a step with `use_tools=True` is executed, the tool selection process:

1. Builds a task description from the step's output
2. Gathers available tools based on the step's scope
3. Sends a structured prompt to the LLM
4. Processes the LLM's decision
5. Executes selected tools
6. Incorporates results back into the workflow

#### Selection Prompt Structure

The default selection prompt looks like this:

```
You are a helpful AI that uses tools to solve problems.

Task: [Step's output text]

Current Context:
- context_key1: value1
- context_key2: value2

Available Tools:
- Calculator
  Signature: add(x: int, y: int) -> int
  Description: Adds two numbers together
- TextProcessor
  Signature: process(text: str, uppercase: bool = False) -> str
  Description: Processes text with optional case conversion

Respond ONLY with a JSON object in this format:
{
    "tool_calls": [
        {
            "tool_name": "<name of tool>",
            "arguments": {
                "param_name": "param_value"
            },
            "confidence": <0.0-1.0>,
            "reasoning": "<why you chose this tool>"
        }
    ]
}
```

#### Tool Selection Results

The LLM responds with structured decisions:

```json
{
    "tool_calls": [
        {
            "tool_name": "Calculator",
            "arguments": {
                "x": 5,
                "y": 3
            },
            "confidence": 0.95,
            "reasoning": "The task requires adding two numbers together"
        }
    ]
}
```

These decisions are then:

1. Validated against tool signatures
2. Filtered by confidence threshold
3. Limited to max_tools_per_step
4. Executed in order
5. Results added to the prompt:

```
[Original prompt...]

Tool Execution Results:

Calculator:
Result: 8
Confidence: 0.95
Reasoning: The task requires adding two numbers together
```

### Customizing Selection

The selection process can be customized in several ways:

#### 1. Custom Prompt Template

```python
custom_template = """
Given the current task and tools, decide which tools would help.

Task: {task}
Context: {context}
Tools Available:
{tool_descriptions}

Return decision in JSON format:
{
    "tool_calls": [
        {
            "tool_name": "name",
            "arguments": {"param": "value"},
            "confidence": 0.0-1.0,
            "reasoning": "explanation"
        }
    ]
}
"""

config = ToolSelectionConfig(
    prompt_template=custom_template
)
```

#### 2. Selection Model Configuration

```python
agent = MyAgent(
    client=client,
    default_model="gpt-4",
    tool_model=ModelConfig(
        name="gpt-3.5-turbo",
        temperature=0.2,  # Lower temperature for more consistent selection
        top_p=0.1        # More focused sampling for decisions
    )
)
```

#### 3. Confidence Thresholds

```python
# Global configuration
config = ToolSelectionConfig(confidence_threshold=0.8)

# Step-specific configuration
@think(tool_confidence=0.9)  # Higher threshold for this step
def analyze(self, input_data: str) -> str:
    return f"Analyze this: {input_data}"
```

### Monitoring Tool Selection

The tool selection process can be monitored through logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Will show:
# - Full selection prompts
# - LLM responses
# - Tool execution attempts
# - Validation results
# - Error messages
```

Tool decisions are also stored in the agent's context:

```python
# After execution
decisions = agent.context.state["last_tool_decisions"]
for decision in decisions:
    print(f"Tool: {decision['tool_name']}")
    print(f"Confidence: {decision['confidence']}")
    print(f"Result: {decision['result']}")
```

## Advanced Usage

### Tool Scopes

Tools can be restricted to specific workflow steps:

```python
agent.register_tool(
    tool=analyze_data,
    name="DataAnalyzer",
    scopes=["think", "observe"]  # Only available in these steps
)
```

Valid scopes are:

- "think": For analysis and reasoning steps
- "act": For action and decision steps
- "observe": For data collection steps
- "synthesize": For summarization steps
- "all": Available in all steps (default)

### Custom Tool Models

Use different models for tool selection:

```python
agent = MyAgent(
    client=client,
    default_model="gpt-4",
    tool_model=ModelConfig(
        name="gpt-3.5-turbo",
        temperature=0.2
    )
)
```

### Direct Tool Usage

Tools can be used directly when needed:

```python
result = agent.use_tool(
    "Calculator",
    x=5,
    y=3
)
```

## Best Practices

### Tool Design
1. Keep tools focused and simple
   ```python
   # Good
   @tool
   def add(x: int, y: int) -> int:
       """Add two numbers."""
       return x + y
   
   # Avoid
   @tool
   def math_operations(x: int, y: int, operation: str) -> int:
       """Perform various math operations."""
       if operation == "add":
           return x + y
       elif operation == "multiply":
           return x * y
       # etc...
   ```

2. Use clear type hints and docstrings
   ```python
   @tool
   def process_text(
       text: str,
       uppercase: bool = False,
       max_length: Optional[int] = None
   ) -> str:
       """Process input text with formatting options.
       
       Args:
           text: The input text to process
           uppercase: Whether to convert to uppercase
           max_length: Optional maximum length
           
       Returns:
           Processed text string
           
       Raises:
           ValueError: If text is empty
       """
       if not text:
           raise ValueError("Text cannot be empty")
       
       result = text.upper() if uppercase else text
       return result[:max_length] if max_length else result
   ```

3. Handle errors gracefully
   ```python
   @tool
   def divide(x: float, y: float) -> float:
       """Divide two numbers.
       
       Args:
           x: Numerator
           y: Denominator
           
       Returns:
           Result of division
           
       Raises:
           ValueError: If attempting to divide by zero
       """
       if y == 0:
           raise ValueError("Cannot divide by zero")
       return x / y
   ```

### Tool Configuration

1. Set appropriate confidence thresholds based on task criticality
2. Group related tools with consistent scopes
3. Use specific tool models for complex selection decisions
4. Monitor and log tool usage for optimization

### Error Handling

- Always validate tool inputs
- Provide clear error messages
- Use appropriate exception types
- Log errors for debugging
```python
import logging

logger = logging.getLogger(__name__)

@tool
def process_data(data: List[float]) -> float:
    """Process numerical data.
    
    Args:
        data: List of numbers to process
        
    Returns:
        Processed result
        
    Raises:
        ValueError: If data is empty or contains invalid values
    """
    try:
        if not data:
            raise ValueError("Data cannot be empty")
        return sum(data) / len(data)
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise
```

For context handling, see next: [Context](../agent/context.md).