# Building a Simple Q&A Bot with ClientAI

Let's build a straightforward Q&A bot using ClientAI's `create_agent` function. This approach gives us powerful features like context management and response streaming while keeping the code minimal and easy to understand.

## Setting Up

Before we start coding, you'll need to install ClientAI with OpenAI support. Open your terminal and run:

```bash
pip install clientai[openai]
```

You'll also need an OpenAI API key. Create a `.env` file in your project directory and add your key:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

## Creating the Bot

Let's create our bot in a file called `qa_bot.py`. We'll break down each part of the code and understand what it does.

First, let's import what we need:

```python
from clientai import ClientAI
from clientai.agent import create_agent
from typing import Iterator, Union
```

Now let's write the function that creates our bot:

```python
def create_bot(api_key: str = None):
    """Create a simple Q&A bot."""
    # Initialize the AI client
    client = ClientAI('openai', api_key=api_key)
    
    # Create an agent with a helpful personality
    system_prompt = """
    You are a friendly and helpful assistant. Your role is to:
    - Answer questions clearly and concisely
    - Maintain a conversational tone
    - Ask for clarification when needed
    """
    
    return create_agent(
        client=client,
        role="assistant",
        system_prompt=system_prompt,
        model="gpt-4",  # Or use "gpt-3.5-turbo" for a more economical option
        stream=True,    # Enable real-time response streaming
        temperature=0.7 # Add some creativity to responses
    )
```

The `create_bot` function does two important things. First, it sets up a connection to OpenAI through ClientAI. Then it creates an agent with a specific personality defined in the system prompt. The agent will use GPT-4 (though you can switch to GPT-3.5 to save costs), stream its responses in real-time, and use a moderate temperature setting to balance creativity and accuracy.

Next, we need a way to display the bot's responses. Since we're using streaming, we need to handle both regular and streaming responses:

```python
def display_response(response: Union[str, Iterator[str]]):
    """Display the bot's response, handling both streaming and non-streaming."""
    if isinstance(response, str):
        print(response)
    else:
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
```

This function checks whether it received a complete string or a stream of text chunks. For streams, it prints each chunk as it arrives, creating that nice "thinking in real-time" effect.

Finally, let's create the main interaction loop:

```python
def main():
    # Create our bot
    bot = create_bot()
    
    print("Simple Q&A Bot (type 'quit' to exit, 'clear' to reset)")
    print("Watch the bot think in real-time!\n")
    
    while True:
        # Get user input
        question = input("\nYou: ").strip()
        
        # Handle commands
        if question.lower() == 'quit':
            break
        elif question.lower() == 'clear':
            bot.reset_context()
            print("Memory cleared!")
            continue
        
        # Get and display response
        print("\nBot: ", end="")
        response = bot.run(question)
        display_response(response)

if __name__ == "__main__":
    main()
```

The main loop creates a simple command-line interface where users can ask questions, clear the conversation history, or quit the program. When a question is asked, it runs it through the agent and displays the response in real-time.

## Using Your Bot

Running the bot is as simple as executing the Python file:

```bash
python qa_bot.py
```

When you run it, you'll see a welcome message and a prompt for your first question. The bot will maintain context between questions, so you can have natural back-and-forth conversations. If you want to start fresh, just type 'clear'.

## Making It Your Own

The bot is quite flexible and can be customized in several ways. Want a more creative bot? Increase the temperature to 0.9. Need more precise answers? Lower it to 0.2. You can even change the bot's personality by modifying the system prompt - make it funny, professional, or anything in between.

If you're watching costs, switch to "gpt-3.5-turbo" instead of "gpt-4". And if you prefer instant complete responses rather than streaming, just set `stream=False` in the create_agent call.

## Taking It Further

This simple bot can grow with your needs. You might want to add error handling for when the API has issues, or save conversations to files for later reference. You could even create a web interface or add support for different AI providers. The foundation we've built here makes all of these enhancements straightforward to add.