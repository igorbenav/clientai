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
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        model: str,
        system_prompt: Optional[str] = None,
        return_full_response: bool = False,
        stream: bool = False,
        json_output: bool = False,
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
        """
        pass
