from typing import Any, Dict, Optional

import pytest

from clientai.agent.memory import Memory, recall, remember


class AgentTestMemory(Memory[str, str]):
    """Test implementation of Memory interface."""

    def __init__(self):
        self._storage: Dict[str, str] = {}

    def store(self, key: str, value: str, **kwargs: Any) -> None:
        """Store a value in memory."""
        self._storage[key] = value

    def retrieve(
        self, key: str, default: Optional[str] = None, **kwargs: Any
    ) -> Optional[str]:
        """Retrieve a value from memory."""
        return self._storage.get(key, default)

    def remove(self, key: str, **kwargs: Any) -> None:
        """Remove a value from memory."""
        self._storage.pop(key, None)

    def clear(self, **kwargs: Any) -> None:
        """Clear all values from memory."""
        self._storage.clear()


class TestMemorySystem:
    """Test suite for memory system functionality."""

    @pytest.fixture
    def memory(self):
        """Fixture providing a test memory implementation."""
        return AgentTestMemory()

    def test_basic_memory_operations(self, memory):
        """Test basic memory store and retrieve operations."""
        # Test store and retrieve
        memory.store("key1", "value1")
        assert memory.retrieve("key1") == "value1"
        assert memory.retrieve("nonexistent") is None

        # Test storing multiple values
        memory.store("key2", "value2")
        assert memory.retrieve("key1") == "value1"
        assert memory.retrieve("key2") == "value2"

        # Test remove operation
        memory.remove("key1")
        assert memory.retrieve("key1") is None
        assert memory.retrieve("key2") == "value2"

        # Test clear operation
        memory.clear()
        assert memory.retrieve("key2") is None
        assert memory.retrieve("key1") is None

    def test_remember_decorator(self, memory):
        """Test @remember decorator functionality."""

        class TestAgent:
            def __init__(self):
                self.memory = memory

            @remember("test_key")
            def test_method(self, value: str) -> str:
                return f"processed_{value}"

        agent = TestAgent()
        result = agent.test_method("data")

        assert result == "processed_data"
        assert memory.retrieve("test_key") == "processed_data"

    def test_recall_decorator(self, memory):
        """Test @recall decorator functionality."""
        # Store a value first
        memory.store("recall_key", "stored_value")

        class TestAgent:
            def __init__(self):
                self.memory = memory

            @recall("recall_key")
            def test_method(self, recalled_value: str, extra: str) -> str:
                return f"{recalled_value}_{extra}"

        agent = TestAgent()
        result = agent.test_method(extra="added")

        assert result == "stored_value_added"

    def test_recall_decorator_with_default(self, memory):
        """Test @recall decorator with default value."""

        class TestAgent:
            def __init__(self):
                self.memory = memory

            @recall("nonexistent_key", default="default_value")
            def test_method(self, recalled_value: str, extra: str) -> str:
                return f"{recalled_value}_{extra}"

        agent = TestAgent()
        result = agent.test_method(extra="added")

        assert result == "default_value_added"

    def test_remember_decorator_with_kwargs(self, memory):
        """Test @remember decorator with additional kwargs."""

        class TestAgent:
            def __init__(self):
                self.memory = memory

            @remember("test_key", category="test")
            def test_method(self, value: str) -> str:
                return f"processed_{value}"

        agent = TestAgent()
        result = agent.test_method("data")

        assert result == "processed_data"
        assert memory.retrieve("test_key") == "processed_data"
