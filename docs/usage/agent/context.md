# Context Management

Context management in ClientAI provides a structured way to maintain state and share information between workflow steps. The context system enables:

- State persistence across workflow steps
- Result tracking and access
- Data sharing between components
- Configuration management
- Error state tracking

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Understanding Context](#understanding-context)
3. [Working with Context](#working-with-context)
4. [Advanced Usage](#advanced-usage)
5. [Best Practices](#best-practices)

## Prerequisites

Before working with context, ensure you have:

1. Basic understanding of:
    - ClientAI agent concepts
    - Workflow steps
    - Tool usage patterns

2. Proper imports:
    ```python
    from clientai.agent import Agent
    from clientai.agent.context import AgentContext
    ```

## Understanding Context

The context system in ClientAI serves as a central state manager for agents, providing:

### Core Features

1. **State Management**
    - Persistent storage during workflow execution
    - Scoped access to stored data
    - Automatic cleanup of temporary data

2. **Result Tracking**
    - Storage of step execution results
    - Access to historical results
    - Tool execution outcome storage

3. **Configuration Access**
    - Agent configuration storage
    - Step-specific settings
    - Tool configuration management

## Working with Context

### Basic Context Operations

#### 1. Accessing Step Results
```python
class MyAgent(Agent):
    @think
    def analyze(self, input_data: str) -> str:
        """First step: analyze data."""
        return f"Analyzing: {input_data}"
    
    @act
    def process(self, prev_result: str) -> str:
        """Second step: access results."""
        # Get specific step result
        analysis = self.context.get_step_result("analyze")
        
        # Get all previous results
        all_results = self.context.get_all_results()
        
        return f"Processing analysis: {analysis}"
```

#### 2. Managing State
```python
class StateManagingAgent(Agent):
    @think
    def first_step(self, input_data: str) -> str:
        # Store data in context
        self.context.set_state("important_data", input_data)
        return "Processing step 1"
    
    @act
    def second_step(self, prev_result: str) -> str:
        # Retrieve data from context
        data = self.context.get_state("important_data")
        
        # Check if data exists
        if self.context.has_state("important_data"):
            return f"Found data: {data}"
        
        return "No data found"
```

#### 3. Configuration Access
```python
class ConfigAwareAgent(Agent):
    @think
    def configured_step(self, input_data: str) -> str:
        # Access agent configuration
        model = self.context.config.default_model
        
        # Access step configuration
        step_config = self.context.get_step_config("configured_step")
        
        return f"Using model: {model}"
```

### Context Lifecycle

The context system maintains state throughout an agent's workflow execution:

1. **Initialization**
   ```python
   class LifecycleAgent(Agent):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           # Context is automatically initialized
           
       @think
       def first_step(self, input_data: str) -> str:
           # Context is ready for use
           self.context.set_state("init_time", time.time())
           return "Step 1"
   ```

2. **Step Execution**
   ```python
   @act
   def execution_step(self, prev_result: str) -> str:
       # Results automatically stored
       current_step = self.context.current_step
       step_start = self.context.get_state("step_start_time")
       
       return "Processing"
   ```

3. **Cleanup**
   ```python
   def cleanup(self):
       """Optional cleanup method."""
       # Clear temporary data
       self.context.clear_state("temp_data")
       
       # Persist important data
       final_state = self.context.get_all_state()
       self.save_state(final_state)
   ```

## Advanced Usage

### 1. Custom Context Extensions
```python
from clientai.agent.context import AgentContext

class CustomContext(AgentContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_storage = {}
    
    def store_custom(self, key: str, value: Any) -> None:
        """Store custom data with validation."""
        if not isinstance(key, str):
            raise ValueError("Key must be string")
        self._custom_storage[key] = value
    
    def get_custom(self, key: str) -> Any:
        """Retrieve custom data."""
        return self._custom_storage.get(key)

class CustomAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = CustomContext()
```

### 2. Context Event Handling
```python
class EventAwareAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context.on_state_change(self._handle_state_change)
        self.context.on_result_added(self._handle_new_result)
    
    def _handle_state_change(self, key: str, value: Any) -> None:
        """Handle state changes."""
        logger.info(f"State changed: {key}")
    
    def _handle_new_result(self, step_name: str, result: Any) -> None:
        """Handle new results."""
        logger.info(f"New result from {step_name}")
```

### 3. Context Serialization
```python
class PersistentAgent(Agent):
    def save_context(self) -> None:
        """Save context to storage."""
        state = self.context.serialize()
        with open("agent_state.json", "w") as f:
            json.dump(state, f)
    
    def load_context(self) -> None:
        """Load context from storage."""
        with open("agent_state.json", "r") as f:
            state = json.load(f)
        self.context.deserialize(state)
```

## Best Practices

### 1. State Management
```python
# Good Practice
class WellManagedAgent(Agent):
    @think
    def first_step(self, input_data: str) -> str:
        # Use clear, descriptive keys
        self.context.set_state("analysis_start_time", time.time())
        self.context.set_state("raw_input", input_data)
        
        # Group related data
        self.context.set_state("analysis", {
            "input": input_data,
            "timestamp": time.time(),
            "status": "started"
        })
        
        return "Processing"
    
    @act
    def cleanup_step(self, prev_result: str) -> str:
        # Clear temporary data
        self.context.clear_state("analysis_start_time")
        
        # Keep important data
        analysis = self.context.get_state("analysis")
        return f"Completed analysis: {analysis}"
```

### 2. Result Access Patterns
```python
class ResultPatternAgent(Agent):
    @think
    def access_results(self, input_data: str) -> str:
        # Prefer specific result access
        last_result = self.context.get_step_result("specific_step")
        
        # Use get_all_results sparingly
        all_results = self.context.get_all_results()
        
        # Check result existence
        if self.context.has_step_result("specific_step"):
            return "Process result"
        
        return "Handle missing result"
```

### 3. Error Handling
```python
class ErrorAwareAgent(Agent):
    @think
    def safe_step(self, input_data: str) -> str:
        try:
            # Access state safely
            data = self.context.get_state("key", default="fallback")
            
            # Validate state before use
            if not self._validate_state(data):
                raise ValueError("Invalid state")
            
            return f"Processing: {data}"
            
        except (KeyError, ValueError) as e:
            logger.error(f"Context error: {e}")
            self.context.set_state("error_state", str(e))
            raise
```

### 4. Performance Considerations

1. Clear unnecessary state:
   ```python
   # Remove temporary data
   self.context.clear_state("temp_calculation")
   
   # Clear multiple states
   self.context.clear_states(["temp1", "temp2"])
   ```

2. Use efficient access patterns:
   ```python
   # Good: Direct access
   result = self.context.get_step_result("specific_step")
   
   # Avoid: Frequent full result access
   all_results = self.context.get_all_results()  # Use sparingly
   ```

3. Batch state updates:
   ```python
   # Good: Batch update
   self.context.set_states({
       "key1": "value1",
       "key2": "value2",
       "key3": "value3"
   })
   
   # Avoid: Multiple individual updates
   self.context.set_state("key1", "value1")
   self.context.set_state("key2", "value2")
   self.context.set_state("key3", "value3")
   ```

Now that you've mastered context management, check out our [Examples](../../examples/overview.md) section to see these concepts in action:

- How to build a simple Q&A bot that introduces core agent features
- Techniques for creating a task planner using `create_agent` and basic tools
- Methods for implementing a writing assistant with multi-step workflows
- Patterns for developing a sophisticated code analyzer with custom workflows

Each example demonstrates different aspects of ClientAI, from basic agent creation to complex systems combining steps, tools, and context management. Start with the [Simple Q&A Bot](../../examples/agent/simple_qa.md) to see ClientAI's fundamentals in practice, or jump straight to the [Code Analyzer](../../examples/agent/code_analyzer.md) for a more advanced implementation.