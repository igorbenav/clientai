# Usage Overview

This Usage section provides comprehensive guides on how to effectively use ClientAI's two main components: the Client for direct AI provider interactions and the Agent for building autonomous AI workflows. Each topic focuses on specific aspects, ensuring you have all the information needed to leverage the full potential of ClientAI in your projects.

## Client Features

Learn how to initialize and use ClientAI with different AI providers. These guides cover the fundamentals of direct AI interaction:

- [Initialization Guide](client/initialization.md)
- [Text Generation Guide](client/text_generation.md)
- [Chat Functionality Guide](client/chat_functionality.md)
- [Multiple Providers Guide](client/multiple_providers.md)
- [Ollama Manager Guide](ollama_manager.md)

## Agent Features

Discover how to create and customize AI agents for autonomous workflows:

- [Creating Agents Guide](agent/creating_agents.md)
- [Workflow Steps Guide](agent/workflow_steps.md)
- [Tools and Tool Selection](agent/tools.md)
- [Context Management](agent/context.md)
- [Validation Guide](agent/validation.md)

## Getting Started

### Quick Start with Client

Here's a simple example using the basic client for direct AI interaction:

```python
from clientai import ClientAI

# Initialize the client
client = ClientAI('openai', api_key="your-openai-api-key")

# Generate text
response = client.generate_text(
    "Explain the concept of machine learning in simple terms.",
    model="gpt-3.5-turbo"
)

print(response)
```

### Quick Start with Agent

Here's how to create a simple agent with tools:

```python
from clientai import ClientAI, create_agent

client = ClientAI('openai', api_key="your-openai-api-key")

# Create a calculator tool
@tool(name="Calculator", description="Performs basic math operations")
def calculate(x: int, y: int) -> int:
    return x + y

# Create an agent with the calculator tool
agent = create_agent(
    client=client,
    role="math_helper",
    system_prompt="You are a helpful math assistant.",
    model="gpt-4",
    tools=[calculate]
)

# Run the agent
result = agent.run("What is 5 plus 3?")
print(result)
```

### Quick Start with Validation

Here's how to create an agent with validated outputs:

```python
from pydantic import BaseModel, Field
from typing import List

class Analysis(BaseModel):
    summary: str = Field(min_length=10)
    sentiment: str = Field(pattern="^(positive|negative|neutral)$")
    key_points: List[str] = Field(min_items=1)

class AnalysisAgent(Agent):
    @think(
        name="analyze",
        json_output=True,
        return_type=Analysis
    )
    def analyze_text(self, text: str) -> Analysis:
        return """
        Analyze this text and return:
        - summary (10+ chars)
        - sentiment (positive/negative/neutral)
        - key_points (non-empty list)
        
        Text: {text}
        """

agent = AnalysisAgent(client=client, default_model="gpt-4")
result = agent.run("Great product, highly recommend!")
print(f"Sentiment: {result.sentiment}")
print(f"Key points: {result.key_points}")
```

## Advanced Usage

### Streaming with Client

The client supports streaming responses:

```python
for chunk in client.generate_text(
    "Tell me a story about space exploration",
    model="gpt-3.5-turbo",
    stream=True
):
    print(chunk, end="", flush=True)
```

### Multi-Step Agent Workflows

Create agents with multiple processing steps:

```python
class AnalysisAgent(Agent):
    @think("analyze")
    def analyze_data(self, input_data: str) -> str:
        return f"Analyze this data: {input_data}"
    
    @act("process")
    def process_results(self, analysis: str) -> str:
        return f"Based on the analysis: {analysis}"

agent = AnalysisAgent(
    client=client,
    default_model="gpt-4",
    tool_confidence=0.8
)
```

## Best Practices

### Client Best Practices

1. **API Key Management**: Store API keys securely as environment variables
2. **Error Handling**: Implement proper error handling for API failures
3. **Model Selection**: Choose models based on task requirements and budget
4. **Context Management**: Manage conversation context efficiently

### Agent Best Practices

1. **Validation**: Use appropriate validation levels:
    - Plain text for simple responses
    - JSON output for basic structure
    - Pydantic models for strict validation
2. **Error Handling**: Always wrap validated calls in try/except blocks
3. **Tools**: Choose tool confidence thresholds based on task criticality
4. **Context**: Use context to share state between steps effectively

## Contribution

If you have suggestions or contributions to these guides, please refer to our [Contributing Guidelines](../community/CONTRIBUTING.md). We appreciate your input in improving our documentation and making ClientAI more accessible to all users.