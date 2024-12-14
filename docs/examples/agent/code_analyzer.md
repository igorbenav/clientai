# Building a Code Analysis Assistant with ClientAI

In this tutorial, we're going to build a code analysis assistant that can help developers improve their Python code. Our assistant will analyze code structure, identify potential issues, and suggest improvements. We'll use ClientAI's framework along with local AI models to create something that's both powerful and practical.

## Understanding the Foundation

Let's start with what we're actually building. A code analyzer needs to look at several aspects of code: its structure, complexity, style, and documentation. To do this effectively, we'll use Python's abstract syntax tree (AST) module to parse and analyze code programmatically.

First, we need a way to represent our analysis results. Here's how we structure this:

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

This dataclass gives us a clean way to organize our findings. The complexity score helps us understand how intricate the code is, while the other fields track the various components we find in the code.

Now let's write the core analysis function that examines Python code:

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
                complexity += sum(1 for _ in ast.walk(node) 
                                if isinstance(_, (ast.If, ast.For, ast.While)))
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

This function does the heavy lifting of our analysis. When we pass it a string of Python code, it uses the AST module to parse the code into a tree structure that we can examine. We walk through this tree looking for different types of nodes that represent functions, classes, and imports.

For each function we find, we also calculate its complexity. We do this by counting control structures like if statements, for loops, and while loops. This gives us a rough measure of how complex the function is - more control structures generally mean more complex code that might need simplification.

Next, we need to look at code style. Python has some well-established style conventions, and we want to check if code follows them:

```python
def check_style_issues(code: str) -> str:
    """Check Python code style issues."""
    issues = []
    
    for i, line in enumerate(code.split("\n"), 1):
        if len(line.strip()) > 88:
            issues.append(f"Line {i} exceeds 88 characters")
            
    function_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    for match in re.finditer(function_pattern, code):
        name = match.group(1)
        if not name.islower():
            issues.append(f"Function '{name}' should use snake_case")
            
    return json.dumps({"issues": issues})
```

This style checker looks at a couple of key aspects of Python style. First, it checks line length - lines that are too long can be hard to read and understand. Second, it looks at function naming conventions. In Python, we typically use snake_case for function names (like `calculate_total` rather than `calculateTotal`).

Now that we have our core analysis functions, let's integrate them with ClientAI. Here's how we package them as tools for the AI to use:

```python
def create_review_tools() -> List[ToolConfig]:
    """Create tool configurations for the assistant."""
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
    ]
```

The `ToolConfig` wrapper provides metadata that helps the AI understand and use our tools effectively:

- `name`: A unique identifier the AI uses to reference the tool
- `description`: Helps the AI understand what the tool does and how to use it
- `scopes`: Controls when the tool can be used in the workflow - in this case, during observation steps when the AI is gathering information

Now let's look at how we build our assistant that uses these tools:

```python
class CodeReviewAssistant(Agent):
    """Code review assistant implementation."""
    
    @observe(
        name="analyze_structure",
        description="Analyze code structure and style",
        stream=True,
    )
    def analyze_structure(self, code: str) -> str:
        """First step: Analyze code structure."""
        self.context.state["code_to_analyze"] = code
        
        return """
        Please analyze this Python code structure and style:
        
        Use the code_analyzer and style_checker tools to evaluate:
        1. Code complexity and structure metrics
        2. Style compliance issues
        3. Function and class organization
        4. Import usage patterns
        """
```

Our assistant inherits from ClientAI's Agent class, which provides the framework for creating AI-powered tools. The `@observe` decorator marks this method as an observation step - a step where we gather information about the code we're analyzing.

Inside the method, we store the code in the assistant's context. This makes it available to our tools and other steps in the process. Then we return a prompt that tells the AI what we want it to analyze.

Let's add a step for suggesting improvements:

```python
    @think(
        name="suggest_improvements",
        description="Generate improvement suggestions",
        stream=True,
    )
    def suggest_improvements(self, analysis_result: str) -> str:
        """Second step: Generate improvement suggestions."""
        current_code = self.context.state.get("current_code", "")
        
        return f"""
        Based on the code analysis of:

        ```python
        {current_code}
        ```

        And the analysis results:
        {analysis_result}

        Please suggest improvements for:
        1. Reducing complexity
        2. Fixing style issues
        3. Improving organization
        4. Enhancing readability
        """
```

This step takes the results from our analysis and asks the AI to generate specific suggestions for improvement. We're using the `@think` decorator here because this step involves processing information and making recommendations rather than just gathering data.

Finally, we need a way to use our assistant. Here's how we set up the command-line interface:

```python
def main():
    """Run the code analysis assistant."""
    config = OllamaServerConfig(
        host="127.0.0.1",
        port=11434,
        gpu_layers=35,
        cpu_threads=8,
    )
    
    with OllamaManager(config) as manager:
        client = ClientAI("ollama", host=f"http://{config.host}:{config.port}")
        
        assistant = CodeReviewAssistant(
            client=client,
            default_model="llama3",
            tools=create_review_tools(),
            tool_confidence=0.8,
            max_tools_per_step=2,
        )
```

This sets up our connection to the local AI model and creates our assistant. We configure it to use our analysis tools and set parameters for how confidently it should use those tools.

When you use the assistant, you can enter Python code and get back detailed analysis and suggestions:

```python
def example(x,y):
    if x > 0:
        if y > 0:
            return x+y
    return 0
```

The assistant will analyze this code and point out several things:

- The nested if statements increase complexity
- The function name is good (it uses snake_case)
- It's missing type hints and a docstring
- The logic could be simplified

The beauty of this system is that it combines static analysis (our Python tools) with AI-powered insights to provide comprehensive code review feedback. The AI can explain issues in a way that's easy to understand and suggest specific improvements based on best practices.

You can build on this foundation by adding more types of analysis, improving the style checks, or even adding support for automatically fixing some of the issues it finds. The key is that we've created a flexible framework that can grow with your needs.