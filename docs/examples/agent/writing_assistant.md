# Building an AI Writing Assistant with ClientAI

Let's create a sophisticated writing assistant using ClientAI's agent framework. This assistant will analyze text, suggest improvements, and rewrite content while maintaining context throughout the process. We'll build it step by step and explain each component.

## Setting Up

First, you'll need to install ClientAI with Groq support. Open your terminal and run:

```bash
pip install clientai[groq]
```

You'll need a Groq API key. Create a `.env` file in your project directory and add your key:

```plaintext
GROQ_API_KEY=your_groq_api_key_here
```

## Creating the Writing Assistant

Let's create our assistant in `writing_assistant.py`. We'll break down each component to understand how it works.

First, let's import our dependencies:

```python
from clientai import ClientAI
from clientai.agent import Agent, think, act, synthesize
from clientai.agent.config import ToolConfig
```

Now let's create a helpful formatting tool that our assistant can use:

```python
def format_text(text: str, style: str = "paragraph") -> str:
    """Format text in different styles."""
    text = text.strip()
    
    if style == "bullet":
        lines = text.split('\n')
        return '\n'.join(f"• {line.strip()}" for line in lines if line.strip())
    elif style == "numbered":
        lines = text.split('\n')
        return '\n'.join(f"{i}. {line.strip()}" for i, line in enumerate(lines, 1) if line.strip())
    else:
        return ' '.join(line.strip() for line in text.split('\n') if line.strip())
```

This tool helps format text in different styles - as paragraphs, bullet points, or numbered lists. The assistant can use this when analyzing or presenting improvements.

Next, let's create our WritingAssistant class:

```python
class WritingAssistant(Agent):
    """
    An AI writing assistant that helps improve text by:
    1. Analyzing the input text
    2. Suggesting improvements 
    3. Applying those improvements
    """
    
    @think(
        # Using explicit name
        name="analyze_text",
        # Using explicit description, docstring will be ignored
        description="Identify issues in the provided text", 
        model={
            "name": "llama-3.2-3b-preview",
            "temperature": 0.7
        }
    )
    def analyze(self, text: str) -> str:
        """This docstring is ignored since description is provided in decorator."""
        return f"""
        Identify specific issues in this text that need improvement:
        
        {text}
        
        List only the actual problems in the text as bullet points. Be specific and brief.
        """
    
    # No name specified - will use function name "suggest"
    # No description specified - will use function docstring
    @act
    def suggest(self, analysis: str) -> str:
        """Generate specific suggestions for improving the analyzed text."""
        return f"""
        Based on these identified issues:
        {analysis}
        
        Looking at this text:
        {self.context.current_input}
        
        Provide 3 specific ways to further improve this text.
        Each suggestion should directly address an issue from the analysis.
        Format as a numbered list of brief, actionable changes.
        """

    @synthesize(
        description="Create improved version of the text"  # Explicit description - docstring ignored
        # No name - will use function name "improve"
    )
    def improve(self, suggestions: str) -> str:
        """This docstring is ignored since description is provided in decorator."""
        return f"""
        Using these improvement suggestions:
        {suggestions}
        
        Rewrite this original text:
        {self.context.current_input}
        
        IMPORTANT: Your response should be ONLY the improved text itself.
        Do not include any preamble like "Here's the improved version" or any other commentary.
        Write just the improved text as a single cohesive paragraph.
        """
```

The WritingAssistant works in three steps:

1. `analyze_text`: Identifies specific issues in the text
2. `suggest`: Generates actionable improvement suggestions
3. `improve`: Creates an improved version incorporating the suggestions

Let's create a function to set up our assistant:

```python
def create_assistant(api_key: str = None):
    """
    Create and configure the writing assistant.
    
    Sets up:
    1. The AI client with appropriate provider (Groq)
    2. The writing assistant with default model configuration
    3. Available tools with proper scoping
    
    Args:
        api_key: Optional Groq API key (can also use environment variable)
        
    Returns:
        WritingAssistant: Configured assistant ready to process text
    """
    # Try to get API key from parameter or environment
    groq_api_key = api_key or os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "Groq API key must be provided either as a parameter or "
            "through the GROQ_API_KEY environment variable"
        )
    
    # Initialize the AI client with Groq
    client = ClientAI('groq', api_key=groq_api_key)
    
    # Create the writing assistant with:
    # - Base model: llama-3.2-3b-preview for general text processing
    # - Formatting tool: Available during analysis phase
    return WritingAssistant(
        client,
        default_model="llama-3.2-3b-preview",
        tools=[
            ToolConfig(
                # The tool function to register
                format_text,
                # Name used by LLM to reference the tool
                name="format",
                # Only available during analysis ("think") phase
                scopes=["think"],
                # Clear description helps LLM understand when to use the tool
                description="Format text as paragraphs, bullet points, or numbered lists."
                            "Requires 'text' parameter and optional 'style' parameter "
                            "('paragraph', 'bullet', 'numbered')."
            )
        ]
    )
```

Now let's create a simple interface to use our assistant:

```python
def main():
    try:
        # Create our assistant
        assistant = create_assistant()
        
        print("AI Writing Assistant (type 'quit' to exit)")
        print("Enter your text and watch it get improved!\n")
        
        while True:
            # Get user input
            text = input("\nEnter text to improve (or 'quit'): ").strip()
            
            if text.lower() == 'quit':
                break
                
            # Improve the text
            print("\nAnalyzing and improving text...")
            improved = assistant.run(text)
            
            # Show results
            print("\nOriginal text:")
            print(text)
            
            print("\nAnalysis:")
            print(assistant.context.last_results["analyze_text"])
            
            print("\nSuggestions:")
            print(assistant.context.last_results["suggest"])
            
            print("\nImproved text:")
            print(improved)
            
    except ValueError as e:
        print(f"\nError: {e}")
        print("Please make sure your Groq API key is properly set.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
```

## Using Your Writing Assistant

To use the assistant, simply run the Python file:

```bash
python writing_assistant.py
```

You'll see a prompt where you can enter text. The assistant will:

1. Analyze the text for issues
2. Suggest specific improvements
3. Provide an improved version
4. Show you the full analysis and reasoning

## Taking It Further

This foundation can be extended in many ways:

- Add support for different writing styles (formal, casual, technical)
- Include grammar checking tools
- Save before/after versions of improved texts
- Create a web interface
- Add support for longer documents
- Integrate with document editors

The modular design makes it easy to add new capabilities while maintaining the clear three-step improvement process.