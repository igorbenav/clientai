from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ._common_types import GenericResponse, Message


class AIProvider(ABC):
    """
    Abstract base class for AI providers.
    """

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> GenericResponse:
        """
        Generate text based on a given prompt.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the AI model to use.
            system_prompt: Optional system prompt to guide model behavior.
            return_full_response: If True, return the full response object
                                  instead of just the generated text.
            stream: If True, return an iterator for streaming responses.
            json_output: If True, format the response as valid JSON.
                        Each provider uses its native JSON support mechanism.
            temperature: Optional temperature value controlling randomness.
                        Usually between 0.0 and 2.0, with lower values making
                        the output more focused and deterministic, and higher
                        values making it more creative and variable.
            top_p: Optional nucleus sampling parameter controlling diversity.
                  Usually between 0.0 and 1.0, with lower values making the
                  output more focused on likely tokens, and higher values
                  allowing more diverse selections.
            **kwargs: Additional keyword arguments specific to
                      the provider's API.

        Returns:
            GenericResponse:
                The generated text response, full response object,
                or an iterator for streaming responses.

        Note:
            When json_output is True:
            - OpenAI/Groq use response_format={"type": "json_object"}
            - Replicate adds output="json" to input parameters
            - Ollama uses format="json" parameter

            Temperature ranges:
            - OpenAI: 0.0 to 2.0 (default: 1.0)
            - Ollama: 0.0 to 2.0 (default: 0.8)
            - Replicate: Model-dependent
            - Groq: 0.0 to 2.0 (default: 1.0)

            Top-p ranges:
            - OpenAI: 0.0 to 1.0 (default: 1.0)
            - Ollama: 0.0 to 1.0 (default: 0.9)
            - Replicate: Model-dependent
            - Groq: 0.0 to 1.0 (default: 1.0)
        """
        pass  # pragma: no cover

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        model: str,
        system_prompt: Optional[str] = None,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> GenericResponse:
        """
        Engage in a chat conversation.

        Args:
            messages: A list of message dictionaries, each containing
                      'role' and 'content'.
            model: The name or identifier of the AI model to use.
            system_prompt: Optional system prompt to guide model behavior.
            return_full_response: If True, return the full response object
                                  instead of just the chat content.
            stream: If True, return an iterator for streaming responses.
            json_output: If True, format the response as valid JSON.
                        Each provider uses its native JSON support mechanism.
            temperature: Optional temperature value controlling randomness.
                        Usually between 0.0 and 2.0, with lower values making
                        the output more focused and deterministic, and higher
                        values making it more creative and variable.
            top_p: Optional nucleus sampling parameter controlling diversity.
                  Usually between 0.0 and 1.0, with lower values making the
                  output more focused on likely tokens, and higher values
                  allowing more diverse selections.
            **kwargs: Additional keyword arguments specific to
                      the provider's API.

        Returns:
            GenericResponse:
                The chat response, either as a string, a dictionary,
                or an iterator for streaming responses.

        Note:
            When json_output is True:
            - OpenAI/Groq use response_format={"type": "json_object"}
            - Replicate adds output="json" to input parameters
            - Ollama uses format="json" parameter

            Temperature ranges:
            - OpenAI: 0.0 to 2.0 (default: 1.0)
            - Ollama: 0.0 to 2.0 (default: 0.8)
            - Replicate: Model-dependent
            - Groq: 0.0 to 2.0 (default: 1.0)

            Top-p ranges:
            - OpenAI: 0.0 to 1.0 (default: 1.0)
            - Ollama: 0.0 to 1.0 (default: 0.9)
            - Replicate: Model-dependent
            - Groq: 0.0 to 1.0 (default: 1.0)
        """
        pass  # pragma: no cover
