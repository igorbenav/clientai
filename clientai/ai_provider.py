from abc import ABC, abstractmethod
from typing import Any, List

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
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenericResponse:
        """
        Generate text based on a given prompt.

        Args:
            prompt: The input prompt for text generation.
            model: The name or identifier of the AI model to use.
            return_full_response: If True, return the full response object
                                  instead of just the generated text.
            stream: If True, return an iterator for streaming responses.
            **kwargs: Additional keyword arguments specific to
                      the provider's API.

        Returns:
            GenericResponse:
                The generated text response, full response object,
                or an iterator for streaming responses.
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        model: str,
        return_full_response: bool = False,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenericResponse:
        """
        Engage in a chat conversation.

        Args:
            messages: A list of message dictionaries, each containing
                      'role' and 'content'.
            model: The name or identifier of the AI model to use.
            return_full_response: If True, return the full response object
                                  instead of just the chat content.
            stream: If True, return an iterator for streaming responses.
            **kwargs: Additional keyword arguments specific to
                      the provider's API.

        Returns:
            GenericResponse:
                The chat response, either as a string, a dictionary,
                or an iterator for streaming responses.
        """
        pass
