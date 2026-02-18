"""Abstract base class for LLM provider wrappers."""

from abc import ABC, abstractmethod

from memory import Message


class BaseProvider(ABC):
    """Common interface that all LLM provider wrappers must implement."""

    @abstractmethod
    async def complete(self, messages: list[Message], system: str) -> str:
        """Generate a reply given a conversation history and a system prompt.

        Args:
            messages: Ordered list of prior messages (oldest first).
            system:   System-level instruction for the model.

        Returns:
            The model's reply text.

        Raises:
            PermissionError: If the provider returns a 403 status.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier, e.g. ``"openai"``."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier in use, e.g. ``"gpt-4o-mini"``."""
        ...
