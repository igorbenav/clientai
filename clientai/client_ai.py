from importlib import import_module
from typing import Any, Generic, List, Optional, cast

from ._constants import (
    GROQ_INSTALLED,
    OLLAMA_INSTALLED,
    OPENAI_INSTALLED,
    REPLICATE_INSTALLED,
)
from ._typing import AIGenericResponse, Message, P, S, T


class ClientAI(Generic[P, T, S]):
    """
    A unified client for interacting with a single AI provider
    (OpenAI, Replicate, Ollama, or Groq).

    This class provides a consistent interface for common
    AI operations such as text generation and chat
    for the chosen AI provider.

    Type Parameters:
        P: The type of the AI provider.
        T: The type of the full response for non-streaming operations.
        S: The type of each chunk in streaming operations.

    Attributes:
        provider: The initialized AI provider.
        system_prompt: The default system prompt for all interactions.
        temperature: The default temperature value for all interactions.
        top_p: The default top-p value for all interactions.

    Args:
        provider_name: The name of the AI provider to use
                      ('openai', 'replicate', 'ollama', or 'groq').
        system_prompt: Optional system prompt to guide the model's behavior
                      across all interactions.
        temperature: Optional default temperature value for all interactions.
                    Controls randomness in output. Provider defaults:
                    - OpenAI: 0.0 to 2.0 (default: 1.0)
                    - Ollama: 0.0 to 2.0 (default: 0.8)
                    - Replicate: Model-dependent
                    - Groq: 0.0 to 2.0 (default: 1.0)
        top_p: Optional default top-p value for all interactions.
               Controls diversity via nucleus sampling. Provider defaults:
               - OpenAI: 0.0 to 1.0 (default: 1.0)
               - Ollama: 0.0 to 1.0 (default: 0.9)
               - Replicate: Model-dependent
               - Groq: 0.0 to 1.0 (default: 1.0)
        **kwargs: Provider-specific initialization parameters.
                 - For OpenAI, Replicate, Groq: Use api_key="your-key"
                 - For Ollama: Use host="your-host" (default: http://localhost:11434)

    Raises:
        ValueError: If an unsupported provider name is given or if
                    temperature/top_p are outside valid ranges.
        ImportError: If the specified provider is not installed.

    Examples:
        Initialize with OpenAI:
        ```python
        ai = ClientAI('openai', api_key="your-openai-key")
        ```

        Initialize with custom parameters:
        ```python
        ai = ClientAI(
            'openai',
            api_key="your-openai-key",
            system_prompt="You are a helpful assistant",
            temperature=0.8,
            top_p=0.9
        )
        ```

        Initialize Ollama with custom host:
        ```python
        ai = ClientAI('ollama', host="http://localhost:11434")
        ```
    """

    def __init__(
        self,
        provider_name: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ):
        prov_name = provider_name
        if prov_name not in ["openai", "replicate", "ollama", "groq"]:
            raise ValueError(f"Unsupported provider: {prov_name}")

        if (
            prov_name == "openai"
            and not OPENAI_INSTALLED
            or prov_name == "replicate"
            and not REPLICATE_INSTALLED
            or prov_name == "ollama"
            and not OLLAMA_INSTALLED
            or prov_name == "groq"
            and not GROQ_INSTALLED
        ):
            raise ImportError(
                f"The {prov_name} provider is not installed. "
                f"Please install it with 'pip install clientai[{prov_name}]'."
            )

        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p

        try:
            provider_module = import_module(
                f".{prov_name}.provider", package="clientai"
            )
            provider_class = getattr(provider_module, "Provider")
            if prov_name in ["openai", "replicate", "groq"]:
                self.provider = cast(
                    P, provider_class(api_key=kwargs.get("api_key"))
                )
            elif prov_name == "ollama":
                self.provider = cast(
                    P, provider_class(host=kwargs.get("host"))
                )
        except ImportError as e:
            raise ImportError(
                f"Error importing {prov_name} provider module: {str(e)}"
            ) from e

    def generate_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        **kwargs: Any,
    ) -> AIGenericResponse:
        """
        Generate text based on a given prompt using
        the specified AI model and provider.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the AI model to use.
            system_prompt: Optional system prompt to override the default one.
                         If None, uses the one specified in initialization.
            temperature: Optional temperature value to override the default.
                        Controls randomness in output. Provider ranges:
                        - OpenAI: 0.0 to 2.0 (default: 1.0)
                        - Ollama: 0.0 to 2.0 (default: 0.8)
                        - Replicate: Model-dependent
                        - Groq: 0.0 to 2.0 (default: 1.0)
            top_p: Optional top-p value to override the default.
                   Controls diversity via nucleus sampling. Provider ranges:
                   - OpenAI: 0.0 to 1.0 (default: 1.0)
                   - Ollama: 0.0 to 1.0 (default: 0.9)
                   - Replicate: Model-dependent
                   - Groq: 0.0 to 1.0 (default: 1.0)
            return_full_response: If True, returns the full structured
                                  response. If False, returns only
                                  the generated text.
            stream: If True, returns an iterator for streaming responses.
            json_output: If True, format the response as valid JSON.
                        Provider implementations:
                        - OpenAI/Groq: Uses response_format={
                              "type": "json_object"
                          }
                        - Replicate: Adds output="json" to parameters
                        - Ollama: Uses format="json" parameter
            **kwargs: Additional keyword arguments
                      specific to the provider's API.

        Returns:
            AIGenericResponse: The generated text response, full response
                               structure, or iterator for streaming responses.

        Raises:
            ValueError: If temperature or top_p are outside valid ranges.

        Examples:
            Basic text generation:
            ```python
            response = ai.generate_text(
                "Tell me a joke",
                model="gpt-3.5-turbo",
            )
            ```

            Generate creative text with custom parameters:
            ```python
            response = ai.generate_text(
                "Write a story about space",
                model="gpt-3.5-turbo",
                temperature=0.8,
                top_p=0.9
            )
            ```

            Generate JSON output:
            ```python
            response = ai.generate_text(
                "Generate a user profile with name and age",
                model="gpt-3.5-turbo",
                json_output=True
            )
            ```

            Stream response:
            ```python
            for chunk in ai.generate_text(
                "Write a long story",
                model="gpt-3.5-turbo",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```
        """

        effective_system_prompt = system_prompt or self.system_prompt
        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )
        effective_top_p = top_p if top_p is not None else self.top_p

        return self.provider.generate_text(
            prompt,
            model,
            system_prompt=effective_system_prompt,
            temperature=effective_temperature,
            top_p=effective_top_p,
            return_full_response=return_full_response,
            stream=stream,
            json_output=json_output,
            **kwargs,
        )

    def chat(
        self,
        messages: List[Message],
        model: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        **kwargs: Any,
    ) -> AIGenericResponse:
        """
        Engage in a chat conversation using
        the specified AI model and provider.

        Args:
            messages: A list of message dictionaries, each containing
                     'role' and 'content'.
            model: The name or identifier of the AI model to use.
            system_prompt: Optional system prompt to override the default one.
                         If None, uses the one specified in initialization.
            temperature: Optional temperature value to override the default.
                        Controls randomness in output. Provider ranges:
                        - OpenAI: 0.0 to 2.0 (default: 1.0)
                        - Ollama: 0.0 to 2.0 (default: 0.8)
                        - Replicate: Model-dependent
                        - Groq: 0.0 to 2.0 (default: 1.0)
            top_p: Optional top-p value to override the default.
                   Controls diversity via nucleus sampling. Provider ranges:
                   - OpenAI: 0.0 to 1.0 (default: 1.0)
                   - Ollama: 0.0 to 1.0 (default: 0.9)
                   - Replicate: Model-dependent
                   - Groq: 0.0 to 1.0 (default: 1.0)
            return_full_response: If True, returns the full structured
                                  response. If False, returns only the
                                  assistant's message.
            stream: If True, returns an iterator for streaming responses.
            json_output: If True, format the response as valid JSON.
                        Provider implementations:
                        - OpenAI/Groq: Uses response_format={
                              "type": "json_object"
                          }
                        - Replicate: Adds output="json" to parameters
                        - Ollama: Uses format="json" parameter
            **kwargs: Additional keyword arguments
                      specific to the provider's API.

        Returns:
            AIGenericResponse: The chat response, full response structure,
                             or an iterator for streaming responses.

        Raises:
            ValueError: If temperature or top_p are outside valid ranges.

        Examples:
            Basic chat:
            ```python
            messages = [
                {"role": "user", "content": "What is AI?"}
            ]
            response = ai.chat(
                messages,
                model="gpt-3.5-turbo",
            )
            ```

            Chat with system prompt and custom temperature:
            ```python
            response = ai.chat(
                messages,
                model="gpt-3.5-turbo",
                system_prompt="You are a helpful AI teacher",
                temperature=0.8,
                top_p=0.9
            )
            ```

            Chat with JSON output:
            ```python
            messages = [
                {"role": "user", "content": "Generate a user profile"}
            ]
            response = ai.chat(
                messages,
                model="gpt-3.5-turbo",
                json_output=True
            )
            ```

            Stream chat response:
            ```python
            for chunk in ai.chat(
                messages,
                model="gpt-3.5-turbo",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```
        """
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if top_p is not None and not 0.0 <= top_p <= 1.0:
            raise ValueError("Top-p must be between 0.0 and 1.0")

        effective_system_prompt = system_prompt or self.system_prompt
        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )
        effective_top_p = top_p if top_p is not None else self.top_p

        return self.provider.chat(
            messages,
            model,
            system_prompt=effective_system_prompt,
            temperature=effective_temperature,
            top_p=effective_top_p,
            return_full_response=return_full_response,
            stream=stream,
            json_output=json_output,
            **kwargs,
        )
