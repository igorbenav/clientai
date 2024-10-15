# Examples Overview

Welcome to the Examples section of the ClientAI documentation. This section provides practical, real-world examples of how to use ClientAI in various applications. Whether you're a beginner looking to get started or an experienced developer seeking inspiration for more complex projects, these examples will demonstrate the versatility and power of ClientAI.

## Featured Examples

Our examples cover a range of applications, from simple text generation to more complex AI-driven systems. Here's an overview of what you'll find in this section:

1. **AI Dungeon Master**: A text-based RPG that uses multiple AI providers to create an interactive storytelling experience.

    - [AI Dungeon Master Tutorial](ai_dungeon_master.md)

2. **Chatbot Assistant**: A simple chatbot that can answer questions and engage in conversation using ClientAI.

    - Soon

3. **Sentiment Analyzer**: An application that analyzes the sentiment of given text using different AI models.

    - Soon

## Usage

Each example is documented on its own page, where you'll find:

- A detailed explanation of the example's purpose and functionality
- Step-by-step instructions for implementing the example
- Code snippets and full source code
- Explanations of key ClientAI features used in the example
- Tips for customizing and extending the example

### Quick Start Example

Here's a simple example to get you started with ClientAI:

```python
from clientai import ClientAI

# Initialize the client
client = ClientAI('openai', api_key="your-openai-api-key")

# Generate a short story
prompt = "Write a short story about a robot learning to paint."
response = client.generate_text(prompt, model="gpt-3.5-turbo")

print(response)
```

For more general usage instructions, please refer to our [Quickstart Guide](../quick-start.md).

## Customizing Examples

Feel free to use these examples as starting points for your own projects. You can modify and extend them to suit your specific needs. If you create an interesting project using ClientAI, we'd love to hear about it!

## Contributing

We welcome contributions to our examples collection! If you've created an example that you think would be valuable to others, please consider submitting it. Check out our [Contributing Guidelines](../community/CONTRIBUTING.md) for more information on how to contribute.

## Feedback

Your feedback helps us improve our examples and documentation. If you have suggestions for new examples, improvements to existing ones, or any other feedback, please let us know through GitHub issues or our community channels.

---

Explore each example to see ClientAI in action and learn how to implement AI-driven features in your own projects.