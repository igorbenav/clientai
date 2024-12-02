from typing import Any, Dict, Optional
from unittest.mock import Mock

import pytest

from clientai.agent.memory import (
    LLMMemoryManager,
    Memory,
    MemoryDecision,
    MemoryType,
    recall,
    remember,
    smart_recall,
    smart_remember,
)


class AgentTestMemory(Memory[str, str]):
    """Test implementation of Memory interface."""

    def __init__(self):
        self._storage: Dict[str, str] = {}

    def store(self, key: str, value: str, **kwargs: Any) -> None:
        self._storage[key] = value

    def retrieve(
        self, key: str, default: Optional[str] = None, **kwargs: Any
    ) -> Optional[str]:
        return self._storage.get(key, default)

    def remove(self, key: str, **kwargs: Any) -> None:
        self._storage.pop(key, None)

    def clear(self, **kwargs: Any) -> None:
        self._storage.clear()


class TestMemorySystem:
    """Test suite for memory system functionality."""

    @pytest.fixture
    def memory(self):
        return AgentTestMemory()

    @pytest.fixture
    def mock_llm_client(self):
        client = Mock()
        client.generate_text.return_value = (
            '{"decisions": ['
            '{"memory_type": "working", '
            '"store": true, '
            '"key": "test", '
            '"importance": 0.8, '
            '"reason": "test", '
            '"metadata": {}}'
            "]}"
        )
        return client

    def test_basic_memory_operations(self, memory):
        """Test basic memory store and retrieve operations."""
        memory.store("key1", "value1")
        assert memory.retrieve("key1") == "value1"
        assert memory.retrieve("nonexistent") is None

        memory.store("key2", "value2")
        memory.remove("key1")
        assert memory.retrieve("key1") is None
        assert memory.retrieve("key2") == "value2"

        memory.clear()
        assert memory.retrieve("key2") is None

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

    def test_smart_remember_decorator(self, mock_llm_client):
        """Test @smart_remember decorator with LLM integration."""

        class TestAgent:
            def __init__(self):
                self.memory_manager = LLMMemoryManager(
                    mock_llm_client,
                    "test-model",
                    {MemoryType.WORKING: AgentTestMemory()},
                )
                self.memories = {MemoryType.WORKING: AgentTestMemory()}
                self.context = Mock()
                self.context.state = {}

            @smart_remember()
            def test_method(self, value: str) -> str:
                return f"processed_{value}"

        agent = TestAgent()
        result = agent.test_method("data")

        assert result == "processed_data"
        assert mock_llm_client.generate_text.called

    def test_smart_recall_decorator(self, mock_llm_client):
        """Test @smart_recall decorator with LLM integration."""
        # Update mock response for retrieval
        mock_llm_client.generate_text.return_value = """
        {
            "retrieval": [
                {
                    "memory_type": "working",
                    "search": true,
                    "strategy": "similar",
                    "reason": "test",
                    "criteria": {"threshold": 0.8}
                }
            ]
        }
        """

        class TestAgent:
            def __init__(self):
                self.memory_manager = LLMMemoryManager(
                    mock_llm_client,
                    "test-model",
                    {MemoryType.WORKING: AgentTestMemory()},
                )
                self.context = Mock()
                self.context.state = {}

            @smart_recall("What happened yesterday?")
            def test_method(
                self, retrieved_memories: Dict[MemoryType, Any]
            ) -> str:
                return str(retrieved_memories)

        agent = TestAgent()
        result = agent.test_method()

        assert mock_llm_client.generate_text.called
        assert isinstance(result, str)

    def test_memory_type_handling(self):
        """Test memory type handling and validation."""
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"

    def test_llm_memory_manager_initialization(self, mock_llm_client):
        """Test LLM memory manager initialization and configuration."""
        working_memory = AgentTestMemory()
        episodic_memory = AgentTestMemory()

        manager = LLMMemoryManager(
            client=mock_llm_client,
            model="test-model",
            memories={
                MemoryType.WORKING: working_memory,
                MemoryType.EPISODIC: episodic_memory,
            },
        )

        assert manager.client == mock_llm_client
        assert manager.model == "test-model"
        assert len(manager.memories) == 2

    def test_memory_decision_execution(self, mock_llm_client):
        """Test memory decision execution through LLM manager."""
        manager = LLMMemoryManager(
            client=mock_llm_client,
            model="test-model",
            memories={MemoryType.WORKING: AgentTestMemory()},
        )

        decisions = manager.decide_storage(
            content="test_content", context={"current_task": "testing"}
        )

        assert len(decisions) == 1
        assert isinstance(decisions[0], MemoryDecision)
        assert MemoryType.WORKING in decisions[0].store_in

    def test_memory_retrieval_decision(self, mock_llm_client):
        """Test memory retrieval decision making."""
        mock_llm_client.generate_text.return_value = """
        {
            "retrieval": [
                {
                    "memory_type": "working",
                    "search": true,
                    "strategy": "similar",
                    "reason": "test",
                    "criteria": {"threshold": 0.8}
                }
            ]
        }"""

        manager = LLMMemoryManager(
            client=mock_llm_client,
            model="test-model",
            memories={MemoryType.WORKING: AgentTestMemory()},
        )

        results = manager.decide_retrieval(
            query="test query", task="test_task", context={}
        )

        assert isinstance(results, dict)
        assert mock_llm_client.generate_text.called

    def test_memory_prompt_templates(self, mock_llm_client):
        """Test memory prompt template functionality."""
        manager = LLMMemoryManager(
            client=mock_llm_client,
            model="test-model",
            memories={MemoryType.WORKING: AgentTestMemory()},
        )

        manager.update_prompt(
            "storage",
            template="Store: {content}",
            variables={"content": "Content to store"},
        )

        manager.update_prompt(
            "retrieval",
            template="Retrieve: {query}",
            variables={"query": "Query to search"},
        )

        with pytest.raises(ValueError):
            manager.update_prompt("nonexistent", "invalid")
