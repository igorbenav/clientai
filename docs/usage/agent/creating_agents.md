# Creating Agents

Agents in ClientAI are AI-powered entities that can execute tasks ranging from simple operations to complex multi-step workflows. An agent combines:

  - A Large Language Model (LLM) for reasoning
  - Optional tools for specific capabilities
  - A workflow system for organizing tasks
  - State management for tracking progress

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Understanding Agent Types](#understanding-agent-types)
3. [Simple Single-Step Agents](#simple-single-step-agents)
4. [Multi-Step Agents](#multi-step-agents)
5. [Agent Configuration](#agent-configuration)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)

## Prerequisites

Before creating agents, ensure you have:

1. Installed ClientAI with your chosen provider:
   ```bash
   # For OpenAI
   pip install clientai[openai]
   ```

2. Initialize a ClientAI instance:
   ```python
   from clientai import ClientAI
   
   # OpenAI
   client = ClientAI('openai', api_key="your-api-key")
   
   # See provider-specific documentation for other options
   ```

3. Basic understanding of:
    - Language model capabilities and limitations
    - Python type hints and decorators

## Understanding Agent Types

Choose the right agent type based on your needs:

### Single-Step Agents
Best for:

- Simple, focused tasks (translation, summarization)
- Quick responses needed
- Minimal context requirements
- Independent operations

### Multi-Step Agents
Best for:

- Complex reasoning chains
- Tool-heavy workflows
- Context-dependent operations
- Tasks requiring multiple capabilities

## Simple Single-Step Agents

The `create_agent` factory function provides the fastest way to create task-specific agents:

### Basic Translation Agent
```python
translator = create_agent(
    client=client,
    role="translator",
    system_prompt="You are a French translator. Translate input to French.",
    model="gpt-4"
)
result = translator.run("Hello world!")
```

### Analysis Agent with Streaming
```python
analyzer = create_agent(
    client=client,
    role="analyzer",
    system_prompt="Analyze data and provide detailed insights.",
    model="gpt-4",
    step="think",    # Uses analysis-optimized parameters
    stream=True      # Enable streaming responses
)

# Stream the analysis
for chunk in analyzer.run("Sales data: [100, 150, 120]"):
    print(chunk, end="", flush=True)
```

### Complete Configuration Example
```python
# Analysis agent with "think" step type
analyzer = create_agent(
    client=client,
    role="analyzer",
    system_prompt="Analyze data and provide insights.",
    model="gpt-4",
    step="think",    # affects default parameters like temperature
                     # Can be: 
                     # - "think" (analysis, temp=0.7),
                     # - "act" (decisions, temp=0.2),
                     # - "observe" (data gathering, temp=0.1),
                     # -  "synthesize" (summarizing, temp=0.4)
    
    # ---------- optional parameters ----------
    temperature=0.7,            # Controls randomness in responses (0.0-1.0)
    top_p=0.9,                  # Controls diversity of responses(0.0-1.0)
    stream=False,               # Whether to stream responses chunk by chunk
    tools=[],                   # List of tools the agent can use (functions, ToolConfig)
    tool_selection_config=None, # Complete tool selection configuration
    tool_confidence=0.8,        # Threshold for tool selection confidence (0.0-1.0)
    tool_model="gpt-4",         # Specific model to use for tool selection decisions
    max_tools_per_step=3,       # Maximum number of tools to use in a single step
    **model_kwargs              # Any additional model-specific parameters
)
```

## Multi-Step Agents

For more complex tasks, create custom agents with multiple steps:

```python
class AnalysisAgent(Agent):
    @think
    def analyze(self, input_data: str) -> str:
        """First step: analyze the input."""
        return f"Please analyze this data: {input_data}"
    
    @act
    def recommend(self, analysis: str) -> str:
        """Second step: make recommendations."""
        return f"Based on the analysis, recommend actions for: {analysis}"
```

Initialize with configuration:
```python
agent = AnalysisAgent(
    client=client,
    default_model=ModelConfig(
        name="gpt-4",
        temperature=0.7,
        stream=True
    )
)
```

## Agent Configuration

Agents can be configured in several ways, with options ranging from simple to complex depending on your needs. The configuration primarily focuses on the model settings, which control how the agent interacts with the language model.

### Model Configuration Options

The default model configuration determines how your agent interacts with the AI provider. You have three ways to specify these settings, each offering different levels of control:

1. **Simple String Name**
    The most basic approach - just specify the model name when you only need default settings:

    ```python
    agent = MyAgent(client, default_model="gpt-4")
    ```

    This is ideal for quick prototypes or when the default parameters work well for your use case.

2. **Dictionary Configuration**
    When you need more control, use a dictionary to specify multiple parameters:

    ```python
    agent = MyAgent(client, default_model={
        "name": "gpt-4",
        "temperature": 0.7, # Control response creativity
        "top_p": 0.9,       # Control response diversity
        "stream": True      # Enable streaming responses
    })
    ```
    This approach is good for runtime configuration or when loading settings from configuration files.

3. **ModelConfig Object**
    The most flexible approach, providing type safety and additional functionality:

    ```python
    agent = MyAgent(client, default_model=ModelConfig(
        name="gpt-4",
        temperature=0.7,
        top_p=0.9,
        stream=False,
        extra_param="value"  # Provider-specific parameters
    ))
    ```

    Use this approach for production code.

## Best Practices

When creating agents, consider these key recommendations:

### Agent Type Selection
Choose the right agent based on your task:

- Single-step agents for:
    - Simple, independent operations (translation, summarization)
    - Quick responses with minimal setup
    - Basic transformations without complex logic

- Multi-step agents for:
    - Complex workflows needing multiple steps
    - Tasks requiring tool combinations
    - Operations needing intermediate results

### Configuration Guidelines
1. Model Selection
    - Use powerful models (e.g., GPT-4) for complex reasoning
    - Use faster models (e.g., qwen-2–7b) for simple transformations
    - Start with default configurations and adjust as needed

2. Basic Settings
   ```python
   agent = MyAgent(
       client=client,
       default_model="gpt-4",
       tools=[calculator, formatter], # Add tools only if needed
       tool_confidence=0.8            # Adjust based on task criticality
   )
   ```

### Error Handling
- Always wrap agent execution in try/except blocks
- Configure retries for critical operations
- Enable logging during development:
  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG)
  ```

For detailed information about workflow steps, see next: [Workflow Steps](workflow_steps.md).