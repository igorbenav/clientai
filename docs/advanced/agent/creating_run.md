# Creating Custom Run Methods in ClientAI

Custom run methods provide granular control over your agent's workflow execution. While ClientAI's default sequential execution works well for most cases, custom run methods let you implement sophisticated logic, handle errors gracefully, and maintain complex state across your workflow.

## Understanding Custom Run Methods

A custom run method replaces the default workflow execution in ClientAI. This gives you direct control over:

- When and how steps execute
- How data flows between steps 
- How results are processed
- How state is maintained
- How errors are handled

Let's build a code review assistant that demonstrates these capabilities by implementing it step by step.

## Step 1: Core Analysis Steps

First, let's define our main analysis steps:

```python
from clientai.agent import Agent, run, think, act, synthesize
from typing import Dict, Optional
import logging
import time

class CodeReviewAssistant(Agent):
    @think
    def analyze_structure(self, code: str) -> str:
        """Analyze code structure and organization."""
        return f"""
        Analyze this code's structure. Consider:
        - Code organization and flow
        - Function and variable naming
        - Module structure
        
        Code: {code}
        """
    
    @think
    def analyze_complexity(self, code: str) -> str:
        """Assess code complexity and maintainability."""
        return f"""
        Evaluate code complexity focusing on:
        - Cyclomatic complexity
        - Cognitive complexity
        - Maintainability concerns
        - Resource usage
        
        Code: {code}
        """
```

These initial steps use the `@think` decorator because they involve analytical processing. Each step has a clear, focused purpose and provides specific guidance to the LLM.

## Step 2: Security and Improvement Steps

Next, we'll add steps for security analysis and suggesting improvements:

```python
    @act
    def run_security_check(self, code: str) -> str:
        """Scan code for potential security issues."""
        return f"""
        Scan this code for security vulnerabilities, focusing on:
        - Input validation
        - Data sanitization
        - Authentication checks
        - Resource management
        
        Code: {code}
        """
    
    @act
    def suggest_refactoring(self, analysis: Dict[str, str]) -> str:
        """Suggest code refactoring improvements."""
        return f"""
        Based on the provided analyses, suggest specific refactoring improvements:
        
        Structural Analysis: {analysis['structure']}
        Complexity Analysis: {analysis['complexity']}
        Security Analysis: {analysis.get('security', 'No security analysis available')}
        
        Provide practical, actionable improvements prioritized by impact.
        """
```

These steps use the `@act` decorator because they involve taking action based on analysis. Note how `suggest_refactoring` accepts a dictionary containing all previous analyses.

## Step 3: Report Generation

The final step synthesizes all our findings into a report:

```python
    @synthesize
    def generate_report(self, data: Dict[str, str]) -> str:
        """Create final code review report."""
        return f"""
        Generate a comprehensive code review report with these sections:
        
        1. Executive Summary
        2. Structural Analysis
        3. Complexity Assessment
        4. Security Review
        5. Recommended Improvements
        6. Action Items
        
        Using this data:
        {data}
        
        Format the report in a professional, easy-to-read style.
        """
```

The `@synthesize` decorator indicates this step combines and formats results from previous steps.

## Step 4: Custom Run Implementation

Now let's implement our custom run method that orchestrates these steps:

```python
    @run
    def custom_run(self, input_data: str) -> str:
        """Execute comprehensive code review workflow."""
        try:
            logging.info("Starting code review workflow")
            self.context.state["start_time"] = time.time()
            self.context.state["original_code"] = input_data
            
            # Initialize results dictionary
            results = {}
            
            # Step 1: Structural Analysis
            logging.info("Analyzing code structure")
            structure_result = self.analyze_structure(input_data)
            if "invalid syntax" in structure_result.lower():
                return "Error: Code contains invalid syntax. Please fix syntax errors before review."
            results["structure"] = structure_result
```

This first part of our custom run method sets up logging, initializes state, and performs the initial structural analysis. Notice how we exit early if we detect syntax errors.

Let's continue with security and complexity analysis:

```python
            # Step 2: Security Analysis
            logging.info("Performing security scan")
            try:
                security_result = self.run_security_check(input_data)
                results["security"] = security_result
                
                if "critical vulnerability" in security_result.lower():
                    logging.warning("Critical security vulnerabilities detected")
                    self.context.state["has_security_issues"] = True
            except Exception as e:
                logging.error(f"Security analysis failed: {e}")
                results["security"] = "Security analysis failed - skipping"
            
            # Step 3: Complexity Analysis
            logging.info("Analyzing code complexity")
            complexity_result = self.analyze_complexity(input_data)
            results["complexity"] = complexity_result
```

This section shows proper error handling and state tracking. Note how security analysis failures don't stop the entire workflow.

Finally, let's handle improvements and report generation:

```python
            # Determine if code requires extensive refactoring
            needs_refactoring = (
                "high complexity" in complexity_result.lower() or
                "maintainability issues" in structure_result.lower() or
                self.context.state.get("has_security_issues", False)
            )
            
            # Generate improvements and final report
            logging.info("Generating improvement suggestions")
            self.context.state["refactoring_priority"] = "high" if needs_refactoring else "low"
            improvements = self.suggest_refactoring(results)
            results["improvements"] = improvements
            
            # Generate final report
            logging.info("Generating final report")
            final_report = self.generate_report(results)
            
            # Update completion time
            self.context.state["end_time"] = time.time()
            self.context.state["processing_time"] = (
                self.context.state["end_time"] - self.context.state["start_time"]
            )
            
            return final_report
            
        except Exception as e:
            logging.error(f"Workflow failed: {e}")
            return f"Error: Code review workflow failed - {str(e)}"
```

## Using the Code Review Assistant

Here's how to use our custom code review assistant:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the agent
agent = CodeReviewAssistant(
    client=client,
    default_model="gpt-4"
)

# Sample code to review
code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total
"""

# Run the review
result = agent.run(code)
print(result)

# Access workflow metrics
print(f"Processing time: {agent.context.state['processing_time']:.2f} seconds")
print(f"Refactoring priority: {agent.context.state['refactoring_priority']}")
```

## Key Features Demonstrated

This implementation shows several important aspects of creating effective custom run methods:

**Error Handling**: Each major step includes error handling that can gracefully recover from failures without stopping the entire workflow.

**State Management**: The context system maintains useful state like processing time, security status, and refactoring priorities.

**Conditional Processing**: The workflow makes intelligent decisions about processing based on results, such as exiting early for syntax errors or adjusting refactoring detail based on code quality.

**Logging**: Comprehensive logging provides visibility into the workflow's progress and helps with debugging issues.

Custom run methods let you create sophisticated workflows that adapt to different situations while maintaining clean, maintainable code structure. They're particularly valuable for complex tasks that need dynamic behavior or sophisticated state management.