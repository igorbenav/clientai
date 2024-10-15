# Chat Functionality in ClientAI

This guide covers how to leverage ClientAI's chat functionality. You'll learn about creating chat conversations, managing context, and handling chat-specific features across supported providers.

## Table of Contents

1. [Basic Chat Interaction](#basic-chat-interaction)
2. [Managing Conversation Context](#managing-conversation-context)
3. [Advanced Chat Features](#advanced-chat-features)
4. [Provider-Specific Chat Capabilities](#provider-specific-chat-capabilities)
5. [Best Practices](#best-practices)

## Basic Chat Interaction

To use the chat functionality in ClientAI, use the `chat` method:

```python
from clientai import ClientAI

client = ClientAI('openai', api_key="your-openai-api-key")

messages = [
    {"role": "user", "content": "Hello, who are you?"}
]

response = client.chat(messages, model="gpt-3.5-turbo")
print(response)

# Continue the conversation
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "What can you help me with?"})

response = client.chat(messages, model="gpt-3.5-turbo")
print(response)
```

This example demonstrates a simple back-and-forth conversation.

## Managing Conversation Context

Effective context management is crucial for coherent conversations:

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant specializing in Python programming."},
    {"role": "user", "content": "How do I use list comprehensions in Python?"}
]

response = client.chat(conversation, model="gpt-3.5-turbo")
print(response)

conversation.append({"role": "assistant", "content": response})
conversation.append({"role": "user", "content": "Can you give an example?"})

response = client.chat(conversation, model="gpt-3.5-turbo")
print(response)
```

This example shows how to maintain context across multiple exchanges, including a system message to set the assistant's role.

## Advanced Chat Features

### Streaming Chat Responses

For real-time conversation, you can stream chat responses:

```python
conversation = [
    {"role": "user", "content": "Tell me a long story about space exploration"}
]

for chunk in client.chat(conversation, model="gpt-3.5-turbo", stream=True):
    print(chunk, end="", flush=True)
```

### Temperature and Top-p Sampling

Adjust the creativity and randomness of responses:

```python
response = client.chat(
    conversation,
    model="gpt-3.5-turbo",
    temperature=0.7,
    top_p=0.9
)
```

## Provider-Specific Chat Capabilities

Different providers may offer unique chat features:

### OpenAI

```python
openai_client = ClientAI('openai', api_key="your-openai-api-key")

response = openai_client.chat(
    [{"role": "user", "content": "Translate 'Hello, world!' to Japanese"}],
    model="gpt-4"
)
```

### Replicate

```python
replicate_client = ClientAI('replicate', api_key="your-replicate-api-key")

response = replicate_client.chat(
    [{"role": "user", "content": "Explain quantum computing"}],
    model="meta/llama-2-70b-chat:latest"
)
```

### Ollama

```python
ollama_client = ClientAI('ollama', host="http://localhost:11434")

response = ollama_client.chat(
    [{"role": "user", "content": "What are the three laws of robotics?"}],
    model="llama2"
)
```

## Best Practices

1. **Context Management**: Keep track of the conversation history, but be mindful of token limits.

   ```python
   max_context_length = 10
   if len(conversation) > max_context_length:
       conversation = conversation[-max_context_length:]
   ```

2. **Error Handling**: Implement robust error handling for chat interactions:

   ```python
   try:
       response = client.chat(conversation, model="gpt-3.5-turbo")
   except Exception as e:
       print(f"An error occurred during chat: {e}")
       response = "I'm sorry, I encountered an error. Could you please try again?"
   ```

3. **User Input Validation**: Validate and sanitize user inputs to prevent potential issues:

   ```python
   def sanitize_input(user_input):
       # Implement appropriate sanitization logic
       return user_input.strip()

   user_message = sanitize_input(input("Your message: "))
   conversation.append({"role": "user", "content": user_message})
   ```

4. **Graceful Fallbacks**: Implement fallback mechanisms for when the AI doesn't understand or can't provide a suitable response:

   ```python
   if not response or response.lower() == "i don't know":
       response = "I'm not sure about that. Could you please rephrase or ask something else?"
   ```

5. **Model Selection**: Choose appropriate models based on the complexity of your chat application:

   ```python
   model = "gpt-4" if complex_conversation else "gpt-3.5-turbo"
   response = client.chat(conversation, model=model)
   ```

6. **Conversation Resetting**: Provide options to reset or start new conversations:

   ```python
   def reset_conversation():
       return [{"role": "system", "content": "You are a helpful assistant."}]

   # Usage
   conversation = reset_conversation()
   ```

By following these guidelines and exploring the various features available, you can create sophisticated chat applications using ClientAI across different AI providers.