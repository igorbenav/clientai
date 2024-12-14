# Examples Overview

Welcome to the Examples section of the ClientAI documentation. We provide both complete example applications and core usage patterns to help you get started.

## Example Applications

### Client-Based Examples
1. [**Simple Q&A Bot**](client/simple_qa.md): Basic question-answering bot showing provider initialization, prompt handling, and core text generation/chat methods.

2. [**Multi-Provider Translator**](client/translator.md): Translation comparator demonstrating simultaneous usage of multiple providers, configurations, and response handling.

3. [**AI Dungeon Master**](client/ai_dungeon_master.md): Text-based RPG orchestrating multiple providers for game state management and dynamic narrative generation.

### Agent-Based Examples
1. [**Simple Q&A Bot**](agent/simple_qa.md): Q&A Bot implementation with Agent, introducing basic agent features.

2. [**Task Planner**](agent/task_planner.md): Basic agent that breaks down goals into steps, introducing create_agent and simple tool creation.

3. [**Writing Assistant**](agent/writing_assistant.md): Multi-step writing improvement agent showcasing workflow steps with think/act/synthesize, decorator configurations, and tool integration.

4. [**Code Analyzer**](agent/code_analyzer.md): Code analysis assistant showcasing custom workflows.

## Core Usage Patterns

### Working with Providers

```python
from clientai import ClientAI

# Initialize with your preferred provider
client = ClientAI('openai', api_key="your-openai-key")
# Or: ClientAI('groq', api_key="your-groq-key")
# Or: ClientAI('replicate', api_key="your-replicate-key")
# Or: ClientAI('ollama', host="your-ollama-host")

# Basic text generation
response = client.generate_text(
    "Tell me a joke",
    model="gpt-3.5-turbo",
)
print(response)

# Chat functionality
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "What is its population?"}
]

response = client.chat(
    messages,
    model="gpt-3.5-turbo",
)
print(response)
```

### Working with Agents

#### Quick-Start Agent
```python
from clientai import client
from clientai.agent import create_agent, tool

@tool(name="add", description="Add two numbers together")
def add(x: int, y: int) -> int:
    return x + y

@tool(name="multiply")
def multiply(x: int, y: int) -> int:
    """Multiply two numbers and return their product."""
    return x * y

# Create a simple calculator agent
calculator = create_agent(
    client=client("groq", api_key="your-groq-key"),
    role="calculator", 
    system_prompt="You are a helpful calculator assistant.",
    model="llama-3.2-3b-preview",
    tools=[add, multiply]
)

result = calculator.run("What is 5 plus 3, then multiplied by 2?")
print(result)
```

#### Custom Workflow Agent
```python
from clientai import Agent, think, act, tool

@tool(name="calculator")
def calculate_average(numbers: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    return sum(numbers) / len(numbers)

class DataAnalyzer(Agent):
    @think("analyze")
    def analyze_data(self, input_data: str) -> str:
        """Analyze sales data by calculating key metrics."""
        return f"""
            Please analyze these sales figures:
        
            {input_data}

            Calculate the average using the calculator tool
            and identify the trend.
            """

    @act
    def summarize(self, analysis: str) -> str:
        """Create a brief summary of the analysis."""
        return """
            Create a brief summary that includes:
            1. The average sales figure
            2. Whether sales are trending up or down
            3. One key recommendation
            """

# Initialize with the tool
analyzer = DataAnalyzer(
    client=client("replicate", api_key="your-replicate-key"),
    default_model="meta/meta-llama-3-70b-instruct",
    tool_confidence=0.8,
    tools=[calculate_average]
)

result = analyzer.run("Monthly sales: [1000, 1200, 950, 1100]")
print(result)
```

## Best Practices

1. **Handle API Keys Securely**: Never hardcode API keys in your source code
2. **Use Type Hints**: Take advantage of ClientAI's type system for better IDE support
3. **Implement Error Handling**: Add appropriate try/catch blocks for API calls
4. **Monitor Usage**: Keep track of API calls and token usage across providers

## Contributing

Have you built something interesting with ClientAI? We'd love to feature it! Check our [Contributing Guidelines](../community/CONTRIBUTING.md) for information on how to submit your examples.

## Next Steps

- Explore the [Usage Guide](../usage/overview.md) for detailed documentation
- Review the [API Reference](../api/overview.md) for complete API details
- Join our community to share your experiences and get help