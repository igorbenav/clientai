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

    Args:
        provider_name: The name of the AI provider to use
                       ('openai', 'replicate', 'ollama', or 'groq').
        system_prompt: Optional system prompt to guide the model's behavior
                       across all interactions.
        temperature: Optional default temperature value for all interactions.
                    Controls randomness in the output (usually 0.0-2.0).
        top_p: Optional default top-p value for all interactions.
               Controls diversity via nucleus sampling (usually 0.0-1.0).
        **kwargs (Any): Provider-specific initialization parameters.

    Raises:
        ValueError: If an unsupported provider name is given.
        ImportError: If the specified provider is not installed.

    Example:
        Initialize with OpenAI:
        ```python
        ai = ClientAI('openai', api_key="your-openai-key")
        ```

        Initialize with custom generation parameters:
        ```python
        ai = ClientAI(
            'openai',
            api_key="your-openai-key",
            temperature=0.8,
            top_p=0.9
        )
        ```
    """

    def __init__(
        self,
        provider_name: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
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
        **kwargs: Any,
    ) -> AIGenericResponse:
        """
        Generate text based on a given prompt
        using the specified AI model and provider.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the AI model to use.
            system_prompt: Optional system prompt to override the default one.
                           If None, uses the one specified in initialization.
            temperature: Optional temperature value to override the default.
                        Controls randomness (usually 0.0-2.0).
            top_p: Optional top-p value to override the default.
                  Controls diversity via nucleus sampling (usually 0.0-1.0).
            return_full_response: If True, returns the full structured
                                  response. If False, returns only the
                                  generated text.
            stream: If True, returns an iterator for streaming responses.
            **kwargs: Additional keyword arguments specific to
                      the chosen provider's API.

        Returns:
            AIGenericResponse:
                The generated text response, full response structure,
                or an iterator for streaming responses.

        Example:
            Generate text with default settings:
            ```python
            response = ai.generate_text(
                "Tell me a joke",
                model="gpt-3.5-turbo",
            )
            ```

            Generate creative text with high temperature:
            ```python
            response = ai.generate_text(
                "Write a story about space",
                model="gpt-3.5-turbo",
                temperature=0.8,
                top_p=0.9
            )
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
        **kwargs: Any,
    ) -> AIGenericResponse:
        """
        Engage in a chat conversation using
        the specified AI model and provider.

        Args:
            messages: A list of message dictionaries, each
                      containing 'role' and 'content'.
            model: The name or identifier of the AI model to use.
            system_prompt: Optional system prompt to override the default one.
                           If None, uses the one specified in initialization.
            temperature: Optional temperature value to override the default.
                        Controls randomness (usually 0.0-2.0).
            top_p: Optional top-p value to override the default.
                  Controls diversity via nucleus sampling (usually 0.0-1.0).
            return_full_response: If True, returns the full structured
                                  response. If False, returns the
                                  assistant's message.
            stream: If True, returns an iterator for streaming responses.
            **kwargs: Additional keyword arguments specific to
                      the chosen provider's API.

        Returns:
            AIGenericResponse:
                The chat response, full response structure,
                or an iterator for streaming responses.

        Example:
            Chat with default settings:
            ```python
            messages = [
                {"role": "user", "content": "What is AI?"}
            ]
            response = ai.chat(
                messages,
                model="gpt-3.5-turbo",
            )
            ```

            Creative chat with custom temperature:
            ```python
            response = ai.chat(
                messages,
                model="gpt-3.5-turbo",
                temperature=0.8,
                top_p=0.9
            )
            ```
        """
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
            **kwargs,
        )
