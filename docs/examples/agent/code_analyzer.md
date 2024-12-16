# Building a Code Analysis Assistant with ClientAI

In this tutorial, we'll create a code analysis system using ClientAI and Ollama. Our assistant will analyze Python code structure, identify potential issues, suggest improvements, and generate documentation - all running on your local machine.

## Table of Contents

1. [Setup and Core Components](#getting-started)
2. [Analysis Tools](#checking-code-style)
3. [Building the Assistant](#registering-our-tools)
4. [Usage and Extensions](#creating-the-command-line-interface)

## Getting Started

First, let's grab all the tools we'll need. Here are our imports:

```python
import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import List

from clientai import ClientAI
from clientai.agent import (
    Agent,
    ToolConfig,
    act,
    observe,
    run,
    synthesize,
    think,
)
from clientai.ollama.manager import OllamaManager, OllamaServerConfig
```

There's quite a bit here, but don't worry - each piece has its purpose. The `ast` module is going to help us understand Python code by turning it into a tree structure we can analyze. We'll use `json` for data handling, and `re` for pattern matching when we check code style. The ClientAI imports give us the framework we need to build our AI-powered assistant.

## Structuring Our Results

When we analyze code, we need a clean way to organize what we find. Here's how we'll structure our results:

```python
@dataclass
class CodeAnalysisResult:
    """Results from code analysis."""
    complexity: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    issues: List[str]
```

Think of this as our report card for code analysis. The complexity score tells us how intricate the code is - higher numbers might mean it's getting too complicated. We keep track of functions and classes we find, which helps us understand the code's structure. The imports list shows us what external code is being used, and the issues list is where we'll note any problems we spot.

## Analyzing Code Structure

Now here's where things get interesting. We need to look inside the code and understand its structure. Here's how we do that:

```python
def analyze_python_code_original(code: str) -> CodeAnalysisResult:
    """Analyze Python code structure and complexity."""
    try:
        tree = ast.parse(code)
        functions = []
        classes = []
        imports = []
        complexity = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                complexity += sum(
                    1
                    for _ in ast.walk(node)
                    if isinstance(_, (ast.If, ast.For, ast.While))
                )
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in node.names:
                    imports.append(name.name)

        return CodeAnalysisResult(
            complexity=complexity,
            functions=functions,
            classes=classes,
            imports=imports,
            issues=[],
        )
    except Exception as e:
        return CodeAnalysisResult(
            complexity=0, functions=[], classes=[], imports=[], issues=[str(e)]
        )
```

This function is like a code detective. We hand it some Python code as a string, and it starts investigating. First, it uses `ast.parse()` to turn the code into a tree structure - imagine turning a book into a detailed outline. Then it walks through this tree, looking for interesting things.

When it finds a function, it doesn't just record its name - it also looks at how complex the function is by counting things like if statements, for loops, and while loops. Each of these makes the code a bit harder to understand, so we keep track of them.

We're also on the lookout for classes and import statements. This helps us understand how the code is organized and what external tools it's using.

## Checking Code Style

Next up is our style checker. Good code isn't just about working correctly - it should also be easy to read and maintain:

```python
def check_style_issues_original(code: str) -> List[str]:
    """Check for Python code style issues."""
    issues = []

    for i, line in enumerate(code.split("\n"), 1):
        if len(line.strip()) > 88:
            issues.append(f"Line {i} exceeds 88 characters")

    function_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    for match in re.finditer(function_pattern, code):
        name = match.group(1)
        if not name.islower():
            issues.append(f"Function '{name}' should use snake_case")

    return issues
```

This function is like a proofreader for code. It checks a couple of key style points. First, it makes sure lines aren't too long - we use 88 characters as our limit because that's a good balance between using space efficiently and keeping code readable.

It also checks function names. In Python, we like to use snake_case for function names (like `calculate_total` instead of `calculateTotal`). The function uses a regular expression pattern to find function definitions and checks if they follow this convention.

## Helping with Documentation

Documentation is crucial for good code, so we've built a helper for that too:

```python
def generate_docstring(code: str) -> str:
    """Generate docstring for Python code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                args = []
                if isinstance(node, ast.FunctionDef):
                    args = [a.arg for a in node.args.args]
                return f"""
                Suggested docstring for {node.name}:

                Args:
                {chr(4).join(f"{arg}: Description of {arg}" for arg in args)}
                Returns:
                    Description of return value

                Examples:
                    ```python
                    # Example usage of {node.name}
                    ```
                """
        return "No functions or classes found to document."
    except Exception as e:
        return f"Error generating docstring: {str(e)}"
```

This is like having a documentation assistant. It looks at functions and classes in your code and generates a template for documenting them. For functions, it automatically finds all the parameters and creates placeholders for describing what they do. It also reminds you to document the return value and include usage examples.

## Making Our Tools AI-Ready

Finally, we need to wrap our tools so they can work with the AI system. We do this by converting their output to JSON:

```python
def analyze_python_code(code: str) -> str:
    """Wrap analyze_python_code_original to return JSON string."""
    if not code:
        return json.dumps({"error": "No code provided"})
    result = analyze_python_code_original(code)
    return json.dumps({
        "complexity": result.complexity,
        "functions": result.functions,
        "classes": result.classes,
        "imports": result.imports,
        "issues": result.issues,
    })

def check_style_issues(code: str) -> str:
    """Wrap check_style_issues_original to return JSON string."""
    if not code:
        return json.dumps({"error": "No code provided"})
    issues = check_style_issues_original(code)
    return json.dumps({"issues": issues})
```

These wrapper functions make our tools more robust by adding input validation and converting their output to a format that the AI can easily understand. They're like translators that help our Python analysis tools communicate with the AI system.

Now that we have our core analysis tools built, let's turn them into something the AI can use. This is where things get really interesting - we're going to teach our AI assistant how to use these tools effectively.

## Registering Our Tools

First, we need to tell ClientAI about our analysis tools. Here's how we do that:

```python
def create_review_tools() -> List[ToolConfig]:
    """Create the tool configurations for code review."""
    return [
        ToolConfig(
            tool=analyze_python_code,
            name="code_analyzer",
            description=(
                "Analyze Python code structure and complexity. "
                "Expects a 'code' parameter with the Python code as a string."
            ),
            scopes=["observe"],
        ),
        ToolConfig(
            tool=check_style_issues,
            name="style_checker",
            description=(
                "Check Python code style issues. "
                "Expects a 'code' parameter with the Python code as a string."
            ),
            scopes=["observe"],
        ),
        ToolConfig(
            tool=generate_docstring,
            name="docstring_generator",
            description=(
                "Generate docstring suggestions for Python code. "
                "Expects a 'code' parameter with the Python code as a string."
            ),
            scopes=["act"],
        ),
    ]
```

Each `ToolConfig` is like a job description for our tools. We give each tool a name that the AI will use to reference it, a description that helps the AI understand when and how to use it, and a scope that determines when the tool can be used. We put our analysis tools in the "observe" scope because they gather information, while the docstring generator goes in the "act" scope because it produces new content.

## Building the Assistant

Now comes the fun part - creating our AI assistant. We'll design it to work in steps, kind of like how a human code reviewer would think:

```python
class CodeReviewAssistant(Agent):
    """An agent that performs comprehensive Python code review."""

    @observe(
        name="analyze_structure",
        description="Analyze code structure and style",
        stream=True,
    )
    def analyze_structure(self, code: str) -> str:
        """Analyze the code structure, complexity, and style issues."""
        self.context.state["code_to_analyze"] = code
        return """
        Please analyze this Python code structure and style:

        The code to analyze has been provided in the context as 'code_to_analyze'.
        Use the code_analyzer and style_checker tools to evaluate:
        1. Code complexity and structure metrics
        2. Style compliance issues
        3. Function and class organization
        4. Import usage patterns
        """
```

This first step is like the initial read-through of the code. The assistant uses our analysis tools to understand what it's looking at. Notice how we store the code in the context - this makes it available throughout the review process.

Next, we want our assistant to think about improvements:

```python
    @think(
        name="suggest_improvements",
        description="Suggest code improvements based on analysis",
        stream=True,
    )
    def suggest_improvements(self, analysis_result: str) -> str:
        """Generate improvement suggestions based on the analysis results."""
        current_code = self.context.state.get("current_code", "")
        return f"""
        Based on the code analysis of:

        ```python
        {current_code}
        ```

        And the analysis results:
        {analysis_result}

        Please suggest specific improvements for:
        1. Reducing complexity where identified
        2. Fixing style issues
        3. Improving code organization
        4. Optimizing import usage
        5. Enhancing readability
        6. Enhancing explicitness

        Provide concrete, actionable suggestions that maintain the code's functionality
        while improving its quality.
        """
```

This step is where the AI starts to form opinions about what could be better. It looks at both the code and the analysis results to make specific suggestions for improvement.

Then we have a step dedicated to documentation:

```python
    @act(
        name="improve_documentation",
        description="Generate improved documentation",
        stream=True,
    )
    def improve_docs(self, improvements: str) -> str:
        """Generate documentation improvements for the code."""
        current_code = self.context.state.get("current_code", "")
        return f"""
        For this code:

        ```python
        {current_code}
        ```

        And these suggested improvements:
        {improvements}

        Please provide comprehensive documentation improvements:
        1. Module-level documentation
        2. Class and function docstrings
        3. Inline comments for complex logic
        4. Usage examples
        """
```

The documentation step uses the context of both the original code and the suggested improvements to recommend better documentation. This way, the documentation reflects not just what the code does now, but also takes into account the planned improvements.

Finally, we wrap everything up in a comprehensive report:

```python
    @synthesize(
        name="create_report",
        description="Create final review report",
        stream=True,
    )
    def generate_report(self) -> str:
        """Generate a comprehensive code review report."""
        current_code = self.context.original_input
        return f"""
        Please create a comprehensive code review report for:

        ```python
        {current_code}
        ```

        Include these sections:

        1. Code Analysis Summary
        2. Suggested Improvements
        3. Documentation Recommendations
        4. Prioritized Action Items

        Format the report with clear headings and specific code examples where relevant.
        """
```

The report step pulls everything together into a clear, actionable document that developers can use to improve their code.

Now that we've built our assistant, let's get it up and running and see it in action. We'll set up a nice command-line interface that makes it easy to interact with our code reviewer.

## Creating the Command-Line Interface

Now let's create a user-friendly way to interact with our assistant:

```python
def main():
    logger = logging.getLogger(__name__)

    # Configure Ollama server
    config = OllamaServerConfig(
        host="127.0.0.1",
        port=11434,
        gpu_layers=35,
        cpu_threads=8,
    )

    # Use context manager for Ollama server
    with OllamaManager(config) as manager:
        # Initialize ClientAI with Ollama
        client = ClientAI("ollama", host=f"http://{config.host}:{config.port}")

        # Create code review assistant with tools
        assistant = CodeReviewAssistant(
            client=client,
            default_model="llama3",
            tools=create_review_tools(),
            tool_confidence=0.8,
            max_tools_per_step=2,
        )

        print("Code Review Assistant (Local AI)")
        print("Enter Python code to review, or 'quit' to exit.")
        print("End input with '###' on a new line.")

        while True:
            try:
                print("\n" + "=" * 50 + "\n")
                print("Enter code:")
                
                # Collect code input
                code_lines = []
                while True:
                    line = input()
                    if line == "###":
                        break
                    code_lines.append(line)

                code = "\n".join(code_lines)
                if code.lower() == "quit":
                    break

                # Process the code
                result = assistant.run(code, stream=True)

                # Handle both streaming and non-streaming results
                if isinstance(result, str):
                    print(result)
                else:
                    for chunk in result:
                        print(chunk, end="", flush=True)
                print("\n")

            except ValueError as e:
                logger.error(f"Error reviewing code: {e}")
                print(f"\nError: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print("\nAn unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main()
```

Let's break down what this interface does:

1. It sets up a local Ollama server for running our AI model
2. Creates our assistant with the tools we built
3. Provides a simple way to input code (type until you enter '###')
4. Handles the review process and displays results in real-time

## Seeing It in Action

Let's try our assistant with a real piece of code. Here's how you'd use it:

```python
# Run the assistant
python code_reviewer.py

# Enter some code to review:
def calculate_total(values,tax_rate):
    Total = 0
    for Val in values:
        if Val > 0:
            if tax_rate > 0:
                Total += Val + (Val * tax_rate)
            else:
                Total += Val
    return Total
###
```

The assistant will analyze this code and provide a comprehensive review that might include:

1. Structure Analysis:
    - Identifies nested if statements that increase complexity
    - Notes the inconsistent variable naming (Total, Val)

2. Style Issues:
    - Missing type hints for parameters
    - Missing docstring
    - Inconsistent capitalization in variable names

3. Documentation Suggestions:
    - Provides a template docstring with parameter descriptions
    - Suggests adding examples showing how to use the function

4. Improvement Recommendations:
    - Simplifying the nested conditions
    - Using consistent snake_case naming
    - Adding type hints and validation

## Extending the Assistant

Want to make the assistant better? Here are some ideas for extending it:

1. Add More Analysis Tools:
    - Security vulnerability scanning
    - Performance analysis
    - Type checking

2. Enhance Style Checking:
    - Add more PEP 8 rules
    - Check for common anti-patterns
    - Analyze variable naming patterns

3. Improve Documentation Analysis:
    - Check coverage of docstrings
    - Validate example code in docstrings
    - Generate more detailed usage examples

4. Add Auto-fixing Capabilities:
    - Automatic formatting
    - Simple refactoring suggestions
    - Documentation generation

The modular nature of our assistant makes it easy to add these enhancements - just create new tools and add them to the workflow.