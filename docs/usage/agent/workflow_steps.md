# ClientAI Step Module Documentation

The Step Module is a core component of ClientAI's agent system, providing a structured way to build complex workflows through a series of specialized processing steps.

## Table of Contents

1. [Understanding Step Types](#understanding-step-types)
2. [Creating Steps](#creating-steps)
3. [Step Execution and Flow](#step-execution-and-flow)
4. [Managing Results](#managing-results)
5. [Advanced Configuration](#advanced-configuration)
6. [Error Handling](#error-handling)

## Understanding Step Types

ClientAI provides four specialized step types, each optimized for different aspects of information processing and decision making. Each type comes with pre-configured parameters that optimize the language model's behavior for that specific purpose.

### Think Steps
Think steps are designed for analysis and reasoning tasks. They use higher temperature settings to encourage creative and exploratory thinking, making them ideal for problem-solving and complex analysis.

```python
@think # Analysis & Reasoning
       # Temperature: 0.7 - Encourages creative thinking
       # Top_p: 0.9 - Allows for diverse idea generation
       # Best for: Complex analysis, planning, brainstorming
def analyze_data(self, input_data: str) -> str:
    """Performs detailed analysis of input data."""
    return f"Please analyze this data in detail: {input_data}"
```

### Act Steps
Act steps are optimized for decisive action and concrete decision-making. They use lower temperature settings to produce more focused and consistent outputs.

```python
@act # Decision Making & Action
     # Temperature: 0.2 - Promotes consistent decisions
     # Top_p: 0.8 - Maintains reasonable variation
     # Best for: Making choices, executing actions
def make_decision(self, input_data: str) -> str:
    """Makes a concrete decision based on analysis."""
    return f"Based on the analysis, decide the best course of action: {input_data}"
```

### Observe Steps
Observe steps are designed for accurate data collection and observation. They use very low temperature settings to maximize precision and accuracy.

```python
@observe # Data Collection & Observation
         # Temperature: 0.1 - Maximizes accuracy
         # Top_p: 0.5 - Ensures high precision
         # Best for: Data gathering, validation, fact-checking
def collect_data(self, input_data: str) -> str:
    """Gathers and validates specific information."""
    return f"Please extract and validate the following information: {input_data}"
```

### Synthesize Steps
Synthesize steps are optimized for combining information and creating summaries. They use moderate temperature settings to balance creativity with coherence.

```python
@synthesize # Summarization & Integration
            # Temperature: 0.4 - Balances creativity and focus
            # Top_p: 0.7 - Enables coherent synthesis
            # Best for: Summarizing, combining information
def combine_insights(self, input_data: str) -> str:
    """Synthesizes multiple pieces of information."""
    return f"Please synthesize the following information into a coherent summary: {input_data}"
```

## Creating Steps

Steps can be created with varying levels of configuration, from simple decorators to fully customized implementations. Here's a progression from basic to advanced step creation:

### Basic Step Definition
The simplest way to create a step is using a decorator with default settings. This approach uses the function name as the step name and the docstring as the description.

```python
class SimpleAgent(Agent):
    @think
    def analyze(self, input_data: str) -> str:
        """Analyzes the input data."""
        return f"Please analyze this data: {input_data}"
```

### Configured Step Definition
For more control, you can provide specific configurations to customize the step's behavior. This allows you to override default settings and specify exactly how the step should operate.

```python
class ConfiguredAgent(Agent):
    @think(
        name="detailed_analysis",                 # Custom step name
        description="Performs detailed analysis", # Step description
        send_to_llm=True,                         # Send output to LLM
        model=ModelConfig(                        # Step-specific model config
            name="gpt-4",
            temperature=0.7
        ),
        stream=True,        # Stream responses
        json_output=False,  # Plain text output
        use_tools=True      # Enable tool usage
    )
    def analyze(self, input_data: str) -> str:
        """Customized analysis step with specific configuration."""
        return f"Please perform a detailed analysis of: {input_data}"
```

### Error-Aware Step Definition
For critical operations, you can create steps with robust error handling and retry logic. This is particularly important for steps that interact with external services or perform critical operations.

```python
class RobustAgent(Agent):
    @think(step_config=StepConfig(
        enabled=True,           # Step can be enabled/disabled
        required=True,          # Failure stops workflow
        retry_count=2,          # Retry twice on failure
        timeout=10.0,           # 10 second timeout
        pass_result=True,       # Pass result to next step
        use_internal_retry=True # Use internal retry mechanism
    ))
    def critical_step(self, input_data: str) -> str:
        """Critical step with error handling and retry logic."""
        return f"Process this critical data with care: {input_data}"
```

## Step Execution and Flow

Understanding how steps execute and how data flows between them is crucial for building effective workflows. Here's a detailed look at the execution process:

### Basic Flow
Steps execute in sequence, with each step's output potentially becoming input for the next step. The execution engine handles the flow of data and interaction with the language model.

```python
class WorkflowAgent(Agent):
    @think
    def step1(self, input_data: str) -> str:
        # 1. Receives initial input or previous step result
        # 2. Returns string that becomes LLM prompt
        return f"Please analyze: {input_data}"
        # 3. LLM processes prompt
        # 4. Response stored in context.last_results["step1"]
        # 5. If pass_result=True, response becomes next step's input
    
    @act(send_to_llm=False)
    def step2(self, prev_result: str) -> str:
        # Manual processing - no LLM interaction
        # Return value becomes step result directly
        return prev_result.upper()
    
    @synthesize
    def step3(self, prev_result: str) -> str:
        # Gets step2's result as prev_result
        # Can access other results through context
        step1_result = self.context.get_step_result("step1")
        return f"Synthesize {prev_result} with {step1_result}"
```

### Controlling Data Flow
You can control how results flow between steps using the `pass_result` configuration. This gives you fine-grained control over what data each step receives.

```python
class FlowControlAgent(Agent):
    @think(step_config=StepConfig(pass_result=False))
    def analyze(self, input_data: str) -> str:
        # Result stored but not passed as next input
        return f"Analysis: {input_data}"
    
    @act
    def process(self, input_data: str) -> str:
        # Still gets original input
        # Must explicitly access analysis result if needed
        analysis = self.context.get_step_result("analyze")
        return f"Processing {input_data} using {analysis}"
```

## Managing Results

ClientAI provides multiple ways to access and manage step results, giving you flexibility in how you handle data between steps.

### Parameter-Based Access
The most straightforward way to access previous results is through step parameters. The number of parameters determines which results a step receives:

```python
class ResultAgent(Agent):
    @think
    def first(self, input_data: str) -> str:
        # Gets initial input
        return "First step result"
    
    @act
    def second(self, prev_result: str) -> str:
        # Gets first step's result
        return f"Using previous result: {prev_result}"
    
    @synthesize
    def third(self, latest: str, earlier: str) -> str:
        # latest = second step result
        # earlier = first step result
        return f"Combining latest ({latest}) with earlier ({earlier})"
```

### Context-Based Access
For more flexible access to results, you can use the context system. This allows you to access any result at any time:

```python
class ContextAgent(Agent):
    @think
    def flexible_step(self) -> str:
        # Access results through context
        current = self.context.current_input
        first_result = self.context.get_step_result("first")
        return f"""
        Working with current input ({current}) and first result ({first_result})
        """
```

## Advanced Configuration

Advanced configuration options allow you to fine-tune step behavior for specific needs.

### Model Configuration Per Step
Different steps may require different model configurations for optimal performance:

```python
class AdvancedAgent(Agent):
    @think(model=ModelConfig(
        name="gpt-4",        # Powerful model for complex analysis
        temperature=0.7,     # Creative thinking
        stream=True          # Stream responses
    ))
    def complex_step(self, input_data: str) -> str:
        """Complex analysis requiring a powerful model."""
        return f"Perform complex analysis of: {input_data}"

    @act(
        model="gpt-3.5",     # Simpler model for basic actions
        tool_model="llama-2" # Different model for tool selection
    )
    def simple_step(self, input_data: str) -> str:
        """Simple action using a more efficient model."""
        return f"Perform simple action on: {input_data}"
```

### Tool Usage Configuration
Configure how steps interact with tools:

```python
class ToolingAgent(Agent):
    @think(
        use_tools=True,
        tool_confidence=0.8,    # High confidence requirement
        tool_model="gpt-4",     # Specific model for tool selection
        max_tools_per_step=2    # Limit tool usage
    )
    def tool_step(self, input_data: str) -> str:
        """Step that carefully selects and uses tools."""
        return f"Analyze this data using appropriate tools: {input_data}"

    @act(tool_selection_config=ToolSelectionConfig(
        confidence_threshold=0.8,
        max_tools_per_step=3,
        prompt_template="Custom tool selection: {task}"
    ))
    def custom_tool_step(self, input_data: str) -> str:
        """Step with custom tool selection behavior."""
        return f"Process this data with custom tool selection: {input_data}"
```

## Error Handling

Robust error handling is crucial for reliable agent workflows. Here's how to implement comprehensive error handling:

### Basic Error Handling
Configure steps with appropriate error handling settings based on their importance:

```python
class ErrorAwareAgent(Agent):
    @think(step_config=StepConfig(
        required=True,     # Must succeed
        retry_count=2,     # Retry twice
        timeout=5.0        # 5 second timeout
    ))
    def must_succeed(self, input_data: str) -> str:
        """Critical step that must complete successfully."""
        return f"Critical processing of {input_data}"

    @act(step_config=StepConfig(
        required=False,   # Optional step
        retry_count=1     # One retry
    ))
    def can_fail(self, input_data: str) -> str:
        """Optional step that can fail without stopping workflow."""
        return f"Optional processing of {input_data}"
```

Robust error handling is essential for building reliable agent workflows. Your agent needs to handle various types of failures, from invalid inputs to failed LLM calls, while maintaining a clear record of what went wrong. Here's how to implement comprehensive error checking and recovery mechanisms.

#### Basic Result Checking
Monitoring step execution results is your first line of defense. This involves validating both the presence and quality of results from each step, with different handling strategies for critical versus optional steps. Here's a pattern that covers the essential checks:

```python
class ErrorCheckingAgent(Agent):
    def check_results(self) -> None:
        """Check results after workflow execution."""
        try:
            # Check critical steps
            critical_result = self.context.get_step_result("must_succeed")
            if not critical_result:
                raise WorkflowError("Critical step failed to complete")
            
            # Validate result format/content
            if not self._validate_result(critical_result):
                raise ValidationError(f"Invalid result format: {critical_result}")
            
            # Check optional steps
            optional_result = self.context.get_step_result("can_fail")
            if optional_result is None:
                logger.warning("Optional step failed but workflow continued")
                
        except (WorkflowError, ValidationError) as e:
            logger.error(f"Workflow validation failed: {e}")
            self._initiate_recovery()
        except Exception as e:
            logger.exception("Unexpected error during result checking")
            raise

    def _validate_result(self, result: str) -> bool:
        """Validate basic result requirements."""
        if not result:
            return False
        try:
            # Check for required components
            required_keywords = ["analysis", "recommendation"]
            return all(keyword in result.lower() for keyword in required_keywords)
        except Exception:
            return False
```

Key points to remember:

- Catch and handle specific exceptions appropriately - be explicit about what can fail and how to handle each case
- Log errors with sufficient context for debugging - include relevant state information and clear error messages
- Implement validation for critical results - don't assume outputs will always be valid
- Have clear recovery strategies for different error types - know what to do when things go wrong
- Use logging levels appropriately (error, warning, debug) - this helps with monitoring and debugging

## Best Practices

1. **Choose Step Types Wisely**
    - Use `@think` for complex analysis and reasoning
    - Use `@act` for decisive actions and choices
    - Use `@observe` for accurate data collection
    - Use `@synthesize` for combining information

2. **Manage Results Carefully**
    - Use parameter-based access when possible for clarity
    - Use context access for complex workflows
    - Be explicit about result flow with pass_result

3. **Handle Errors Appropriately**
    - Configure retry and timeout for unreliable operations
    - Mark critical steps as required
    - Implement proper error checking

4. **Optimize Performance**
    - Use appropriate models for each step's complexity
    - Configure tool selection carefully
    - Stream responses when appropriate

Remember that steps are building blocks - design them to be modular, focused, and reusable when possible. Clear documentation and appropriate error handling will make your workflows more maintainable and reliable.

For detailed information about tool usage, see next [Tools](tools.md).