# ClientAI Tutorial: Building a Simple Q&A Bot

In this tutorial, we'll create an interactive question-answering bot using the ClientAI package. Our bot will maintain conversation context, provide real-time responses, and demonstrate the core features of working with AI providers through a unified interface.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Setting Up the Project](#2-setting-up-the-project)
3. [Building the Q&A Bot](#3-building-the-qa-bot)
4. [Creating the User Interface](#4-creating-the-user-interface)
5. [Running the Bot](#5-running-the-bot)
6. [Further Improvements](#6-further-improvements)

## 1. Introduction

ClientAI makes it easy to work with various AI providers through a single, consistent interface. In this tutorial, we'll use it to build a Q&A bot that does more than just ask and answer questions. Our bot will maintain conversation history, stream responses in real-time for a more engaging experience, and handle different AI providers seamlessly.

The end result will be a practical bot that you can use for testing different AI providers, experimenting with conversation flows, or as a foundation for more complex chatbot applications.

## 2. Setting Up the Project

Let's start by setting up our development environment. First, create a new directory for your project:

```bash
mkdir simple_qa_bot
cd simple_qa_bot
```

Now we need to install ClientAI. We'll install it with support for OpenAI because that's what we'll use:

```bash
pip install clientai[openai]
```

Before we start coding, let's set up our API keys. Create a `.env` file in your project directory:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
```

This keeps our sensitive information separate from our code and makes it easy to switch between different development environments.

## 3. Building the Q&A Bot

At the heart of our project is the SimpleQABot class. This class will handle all interactions with the AI provider and maintain our conversation state. Let's start building it:

```python
from typing import Optional
from clientai import ClientAI
import time

class SimpleQABot:
    """A basic question-answering bot that demonstrates core ClientAI functionality."""
    
    def __init__(self, provider: str = 'openai', api_key: Optional[str] = None):
        """Initialize the Q&A bot with specified provider."""
        self.client = ClientAI(provider, api_key=api_key)
        self.context = []
```

Our initialization is straightforward but flexible. We can specify which AI provider to use and optionally provide an API key. If no key is provided, ClientAI will look for it in environment variables.

The conversation context is stored in a simple list. This allows our bot to remember previous interactions and maintain coherent conversations. Each interaction will be stored as a message with a role (either 'user' or 'assistant') and its content.

Now let's add the core question-answering functionality:

```python
def ask(self, question: str, stream: bool = False) -> str:
    """Ask a question and get a response."""
    # Add the question to conversation context
    self.context.append({"role": "user", "content": question})
    
    # Get response from AI
    response = self.client.chat(
        messages=self.context,
        model="gpt-3.5-turbo",  # Default model
        stream=stream
    )
    
    # Handle streaming vs non-streaming response
    if stream:
        full_response = ""
        for chunk in response:
            print(chunk, end="", flush=True)
            full_response += chunk
            time.sleep(0.02)  # Small delay for readability
        print()
        final_response = full_response
    else:
        final_response = response
        
    # Add response to context for future questions
    self.context.append({"role": "assistant", "content": final_response})
    
    return final_response
```

The `ask` method is where the magic happens. When a question is asked:
1. We add it to our conversation context
2. Send the entire context to the AI provider
3. Either stream the response word by word or return it all at once
4. Store the response in our context for future reference

We've added a small delay when streaming responses. This makes the output more readable and gives the impression of the bot "thinking" as it responds.

We also need a way to start fresh conversations:

```python
def clear_context(self):
    """Clear the conversation history."""
    self.context = []
```

This simple method resets our conversation context, allowing users to start new conversations without previous context influencing the responses.

## 4. Creating the User Interface

Now let's create a simple but effective command-line interface for our bot. Create a main.py file:

```python
def main():
    # Initialize bot
    bot = SimpleQABot('openai')
    
    print("Simple Q&A Bot (type 'quit' to exit, 'clear' to clear context)")
    print("Streaming mode is ON - watch the bot think!\n")
    
    while True:
        # Get user input
        question = input("\nYou: ")
        
        # Handle commands
        if question.lower() == 'quit':
            break
        elif question.lower() == 'clear':
            bot.clear_context()
            print("Context cleared!")
            continue
            
        # Get and display response
        print("\nBot: ", end="")
        response = bot.ask(question, stream=True)

if __name__ == "__main__":
    main()
```

Our interface is straightforward but includes some nice features:

- Clear prompts for user input
- Command handling for quitting and clearing context
- Real-time response streaming
- Visual separation between user and bot messages

## 5. Running the Bot

Using the bot is as simple as running the main script:

```bash
python main.py
```

Here's what you'll see:

```
Simple Q&A Bot (type 'quit' to exit, 'clear' to clear context)
Streaming mode is ON - watch the bot think!

You: What is Python?
Bot: Python is a high-level, interpreted programming language known for its 
simple and readable syntax...

You: Why is it popular?
Bot: Python's popularity comes from several key factors...

You: clear
Context cleared!

You: quit
```

## 6. Further Improvements

While our Q&A bot is already functional, there are many ways to enhance it. You could add error handling to gracefully manage API failures and rate limits. The context management could be expanded to include conversation saving and loading.

The bot's interface could be improved with a web frontend, or you could add support for different response formats. You might also want to experiment with different AI providers and models to find the best fit for your needs.

For a more robust implementation, consider using ClientAI's agent framework. This would give you access to:

- Automatic tool selection for complex tasks
- Structured workflow management
- Enhanced context handling
- Better error recovery

The modular design of our bot makes it easy to add these improvements incrementally. You can start with the features that matter most to your use case and build from there.