# Installing

## Requirements

Before installing ClientAI, ensure you have the following prerequisites:

* **Python:** Version 3.9 or newer.
* **Core Dependencies:** ClientAI has minimal core dependencies, which will be automatically installed.
* **Provider-Specific Libraries:** Depending on which AI providers you plan to use, you may need to install additional libraries:
    * For OpenAI: `openai` library
    * For Replicate: `replicate` library
    * For Ollama: `ollama` library
    * For Groq: `groq` library

## Installing

ClientAI offers flexible installation options to suit your needs:

### Basic Installation

To install the core ClientAI package without any provider-specific dependencies:

```sh
pip install clientai
```

Or, if using poetry:

```sh
poetry add clientai
```

### Installation with Specific Providers

To install ClientAI with support for specific providers:

```sh
pip install clientai[openai]  # For OpenAI support
pip install clientai[replicate]  # For Replicate support
pip install clientai[ollama]  # For Ollama support
pip install clientai[groq]  # For Groq support
```

Or with poetry:

```sh
poetry add "clientai[openai]"
poetry add "clientai[replicate]"
poetry add "clientai[ollama]"
poetry add "clientai[groq]"
```

### Full Installation

To install ClientAI with support for all providers:

```sh
pip install clientai[all]
```

Or with poetry:

```sh
poetry add "clientai[all]"
```
