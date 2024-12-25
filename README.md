# ClientAI

<p align="center">
  <a href="https://igorbenav.github.io/clientai/">
    <img src="assets/ClientAI.png" alt="ClientAI logo" width="45%" height="auto">
  </a>
</p>

<p align="center">
  <i>A unified client for AI providers with built-in agent support.</i>
</p>

<p align="center">
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

---

<b>ClientAI</b> is a Python package that provides a unified framework for building AI applications, from direct provider interactions to transparent LLM-powered agents, with seamless support for OpenAI, Replicate, Groq and Ollama.

**Documentation**: [igorbenav.github.io/clientai/](https://igorbenav.github.io/clientai/)

---

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
pip install "clientai[all]"
```

Or, if you prefer to install only specific providers:

```sh
pip install "clientai[openai]"  # For OpenAI support
pip install "clientai[replicate]"  # For Replicate support
pip install "clientai[ollama]"  # For Ollama support
pip install "clientai[groq]"  # For Groq support
```

## Quick Start Examples

### Basic Provider Usage

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

### Quick-Start Agent

```python
from clientai import client
from clientai.agent import create_agent, tool

@tool(name="calculator")
def calculate_average(numbers: list[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    return sum(numbers) / len(numbers)

analyzer = create_agent(
    client=client("groq", api_key="your-groq-key"),
    role="analyzer", 
    system_prompt="You are a helpful data analysis assistant.",
    model="llama-3.2-3b-preview",
    tools=[calculate_average]
)

result = analyzer.run("Calculate the average of these numbers: [1000, 1200, 950, 1100]")
print(result)
```

### 3. Custom Agent with Validation

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

See our [documentation](https://igorbenav.github.io/clientai/) for more examples, including:

- Custom workflow agents with multiple steps
- Complex tool integration and selection
- Advanced usage patterns and best practices

## Design Philosophy

The ClientAI Agent module is built on four core principles:

1. **Prompt-Centric Design**: Prompts are explicit, debuggable, and transparent. What you see is what is sent to the model.

2. **Customization First**: Every component is designed to be extended or overridden. Create custom steps, tool selectors, or entirely new workflow patterns.

3. **Zero Lock-In**: Start with high-level components and drop down to lower levels as needed. You can:
    - Extend `Agent` for custom behavior
    - Use individual components directly
    - Gradually replace parts with your own implementation
    - Or migrate away entirely - no lock-in

## Requirements

- **Python:** Version 3.9 or newer
- **Dependencies:** Core package has minimal dependencies. Provider-specific packages are optional.

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Igor Magalhaes – [@igormagalhaesr](https://twitter.com/igormagalhaesr) – igormagalhaesr@gmail.com
[github.com/igorbenav](https://github.com/igorbenav/)