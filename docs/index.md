<style>
    .md-typeset h1,
    .md-content__button {
        display: none;
    }
</style>

<p align="center">
  <a href="https://github.com/igorbenav/clientai">
    <img src="assets/ClientAI.png?raw=true" alt="ClientAI logo" width="45%" height="auto">
  </a>
</p>
<p align="center" markdown=1>
  <i>A unified client for AI providers with built-in agent support.</i>
</p>
<p align="center" markdown=1>
<a href="https://github.com/igorbenav/clientai/actions/workflows/tests.yml">
  <img src="https://github.com/igorbenav/clientai/actions/workflows/tests.yml/badge.svg" alt="Tests"/>
</a>
<a href="https://pypi.org/project/clientai/">
  <img src="https://img.shields.io/pypi/v/clientai?color=%2334D058&label=pypi%20package" alt="PyPi Version"/>
</a>
<a href="https://pypi.org/project/clientai/">
  <img src="https://img.shields.io/pypi/pyversions/clientai.svg?color=%2334D058" alt="Supported Python Versions"/>
</a>
</p>
<hr>
<p align="justify">
<b>ClientAI</b> is a Python package that provides a unified framework for building AI applications, from direct provider interactions to transparent LLM-powered agents, with seamless support for OpenAI, Replicate, Groq and Ollama.
</p>
<hr>

## Features

- **Unified Interface**: Consistent methods across multiple AI providers (OpenAI, Replicate, Groq, Ollama).
- **Streaming Support**: Real-time response streaming and chat capabilities.
- **Intelligent Agents**: Framework for building transparent, multi-step LLM workflows with tool integration.
- **Output Validation**: Built-in validation system for ensuring structured, reliable outputs from each step.
- **Modular Design**: Use components independently, from simple provider wrappers to complete agent systems.
- **Type Safety**: Comprehensive type hints for better development experience.

## Installing

To install ClientAI with all providers, run:

```sh
pip install clientai[all]
```

Or, if you prefer to install only specific providers:

```sh
pip install clientai[openai]  # For OpenAI support
pip install clientai[replicate]  # For Replicate support
pip install clientai[ollama]  # For Ollama support
pip install clientai[groq]  # For Groq support
```

## Design Philosophy

The ClientAI Agent module is built on four core principles:

1. **Prompt-Centric Design**: Prompts are the key interface between you and the LLM. They should be explicit, debuggable, and easy to understand. No hidden or obscured prompts - what you see is what is sent to the model.

2. **Customization First**: Every component is designed to be extended or overridden. Create custom steps, tool selectors, or entirely new workflow patterns. The architecture embraces your modifications.

3. **Zero Lock-In**: Start with high-level components and drop down to lower levels as needed. You can:
    - Extend `Agent` for custom behavior
    - Use individual components directly
    - Gradually replace parts with your own implementation
    - Or gradually migrate away entirely - no lock-in

## Examples

### 1. Basic Client Usage

```python
from clientai import ClientAI

# Initialize with OpenAI
client = ClientAI('openai', api_key="your-openai-key")

# Generate text
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

### 2. Quick-Start Agent

The fastest way to create a simple agent:

```python
from clientai import client
from clientai.agent import create_agent, tool

# creating a tool with an explicit description
@tool(name="add", description="Add two numbers together")
def add(x: int, y: int) -> int:
    return x + y

# creating a tool that uses the docstring as description
@tool(name="multiply")
def multiply(x: int, y: int) -> int:
    """Multiply two numbers and return their product."""
    return x * y

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

### 3. Custom Agent with Workflow

For more control, create a custom agent with defined steps:

```python
from clientai import Agent, think, act, Tool

# Create a standalone tool
@tool(name="calculator")
def calculate_average(numbers: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    return sum(numbers) / len(numbers)

class DataAnalyzer(Agent):
    # add an analyze step
    @think("analyze")
    def analyze_data(self, input_data: str) -> str:
        """Analyze sales data by calculating key metrics."""
        return f"""
            Please analyze these sales figures:
        
            {input_data}

            Calculate the average using the calculator tool
            and identify the trend.
            """

    # and also an act step
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

### 4. Custom Agent with Validation

For guaranteed output structure and type safety:

```python
from clientai.agent import Agent, think
from pydantic import BaseModel, Field
from typing import List

class Analysis(BaseModel):
    summary: str = Field(min_length=10)
    key_points: List[str] = Field(min_items=1)
    sentiment: str = Field(pattern="^(positive|negative|neutral)$")

class DataAnalyzer(Agent):
    @think(
        name="analyze",
        json_output=True,  # Enable JSON formatting
    )

    def analyze_data(self, data: str) -> Analysis: # Enable validation
        """Analyze data with validated output structure."""
        return """
        Analyze this data and return a JSON with:
        - summary: at least 10 characters
        - key_points: non-empty list
        - sentiment: positive, negative, or neutral

        Data: {data}
        """

# Initialize and use

analyzer = DataAnalyzer(client=client, default_model="gpt-4")
result = analyzer.run("Sales increased by 25% this quarter")
print(f"Sentiment: {result.sentiment}")
print(f"Key Points: {result.key_points}")
```

## Requirements

Before installing ClientAI, ensure you have the following prerequisites:

* **Python:** Version 3.9 or newer.
* **Dependencies:** The core ClientAI package has minimal dependencies. Provider-specific packages (e.g., `openai`, `replicate`, `ollama`, `groq`) are optional and can be installed separately.

## Usage

ClientAI offers three main ways to interact with AI providers:

1. Text Generation: Use the `generate_text` method for text generation tasks.
2. Chat: Use the `chat` method for conversational interactions.
3. Agents: Create intelligent agents with automated tool selection and workflow management.

All methods support streaming responses and returning full response objects.

## Next Steps

1. Check out the [Usage Guide](usage/overview.md) for detailed functionality and advanced features
2. See the [API Reference](api/overview.md) for complete API documentation
3. For agent development, see the [Agent Guide](usage/agent/creating_agents.md)
4. Explore our [Examples](examples/overview.md) for practical applications and real-world usage patterns

Remember to handle API keys securely and never expose them in your code or version control systems.

## License

[`MIT`](community/LICENSE.md)