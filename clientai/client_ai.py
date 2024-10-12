from importlib import import_module
from typing import Any, Generic, List, cast

from ._constants import OLLAMA_INSTALLED, OPENAI_INSTALLED, REPLICATE_INSTALLED
from ._typing import AIGenericResponse, Message, P, S, T


class ClientAI(Generic[P, T, S]):
    """
    A unified client for interacting with a single AI provider
    (OpenAI, Replicate, or Ollama).

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
                       ('openai', 'replicate', or 'ollama').
        **kwargs (Any): Provider-specific initialization parameters.

    Raises:
        ValueError: If an unsupported provider name is given.
        ImportError: If the specified provider is not installed.

    Examples:
        Initialize with OpenAI:
        ```python
        ai = ClientAI('openai', api_key="your-openai-key")
        ```

        Initialize with Replicate:
        ```python
        ai = ClientAI('replicate', api_key="your-replicate-key")
        ```

        Initialize with Ollama:
        ```python
        ai = ClientAI('ollama', host="your-ollama-host")
        ```
    """

    def __init__(self, provider_name: str, **kwargs):
        prov_name = provider_name
        if prov_name not in ["openai", "replicate", "ollama"]:
            raise ValueError(f"Unsupported provider: {prov_name}")

        if (
            prov_name == "openai"
            and not OPENAI_INSTALLED
            or prov_name == "replicate"
            and not REPLICATE_INSTALLED
            or prov_name == "ollama"
            and not OLLAMA_INSTALLED
        ):
            raise ImportError(
                f"The {prov_name} provider is not installed. "
                f"Please install it with 'pip install clientai[{prov_name}]'."
            )

        try:
            provider_module = import_module(
                f".{prov_name}.provider", package="clientai"
            )
            provider_class = getattr(provider_module, "Provider")
            if prov_name in ["openai", "replicate"]:
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
            return_full_response: If True, returns the full structured response
                                  If False, returns only the generated text.
            stream: If True, returns an iterator for streaming responses.
            **kwargs: Additional keyword arguments specific to
                      the chosen provider's API.

        Returns:
            AIGenericResponse:
                The generated text response, full response structure,
                or an iterator for streaming responses.

        Examples:
            Generate text using OpenAI (text only):
            ```python
            response = ai.generate_text(
                "Tell me a joke",
                model="gpt-3.5-turbo",
            )
            ```

            Generate text using OpenAI (full response):
            ```python
            response = ai.generate_text(
                "Tell me a joke",
                model="gpt-3.5-turbo",
                return_full_response=True
            )
            ```

            Generate text using OpenAI (streaming):
            ```python
            for chunk in ai.generate_text(
                "Tell me a joke",
                model="gpt-3.5-turbo",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```

            Generate text using Replicate:
            ```python
            response = ai.generate_text(
                "Explain quantum computing",
                model="meta/llama-2-70b-chat:latest",
            )
            ```

            Generate text using Ollama:
            ```python
            response = ai.generate_text(
                "What is the capital of France?",
                model="llama2",
            )
            ```
        """
        return self.provider.generate_text(
            prompt,
            model,
            return_full_response=return_full_response,
            stream=stream,
            **kwargs,
        )

    def chat(
        self,
        messages: List[Message],
        model: str,
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
            return_full_response: If True, returns the full structured response
                                  If False, returns the assistant's message.
            stream: If True, returns an iterator for streaming responses.
            **kwargs: Additional keyword arguments specific to
                      the chosen provider's API.

        Returns:
            AIGenericResponse:
                The chat response, full response structure,
                or an iterator for streaming responses.

        Examples:
            Chat using OpenAI (message content only):
            ```python
            messages = [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "What is its population?"}
            ]
            response = ai.chat(
                messages,
                model="gpt-3.5-turbo",
            )
            ```

            Chat using OpenAI (full response):
            ```python
            response = ai.chat(
                messages,
                model="gpt-3.5-turbo",
                return_full_response=True
            )
            ```

            Chat using OpenAI (streaming):
            ```python
            for chunk in ai.chat(
                messages,
                model="gpt-3.5-turbo",
                stream=True
            ):
                print(chunk, end="", flush=True)
            ```

            Chat using Replicate:
            ```python
            messages = [
                {"role": "user", "content": "Explain the concept of AI."}
            ]
            response = ai.chat(
                messages,
                model="meta/llama-2-70b-chat:latest",
            )
            ```

            Chat using Ollama:
            ```python
            messages = [
                {"role": "user", "content": "What are the laws of robotics?"}
            ]
            response = ai.chat(messages, model="llama2")
            ```
        """
        return self.provider.chat(
            messages,
            model,
            return_full_response=return_full_response,
            stream=stream,
            **kwargs,
        )
