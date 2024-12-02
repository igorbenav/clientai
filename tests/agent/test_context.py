from typing import Any, Dict

import pytest

from clientai.agent.core.context import AgentContext
from clientai.agent.memory import Memory


class MockMemory(Memory[str, Any]):
    def __init__(self):
        self._storage: Dict[str, Any] = {}

    def store(self, key: str, value: Any, **kwargs: Any) -> None:
        self._storage[key] = value

    def retrieve(self, key: str, default: Any = None, **kwargs: Any) -> Any:
        return self._storage.get(key, default)

    def remove(self, key: str, **kwargs: Any) -> None:
        self._storage.pop(key, None)

    def clear(self, **kwargs: Any) -> None:
        self._storage.clear()


class TestContextManagement:
    @pytest.fixture
    def context(self):
        return AgentContext()

    @pytest.fixture
    def memory(self):
        return MockMemory()

    def test_context_initialization(self, context):
        """Test initial state of context."""
        assert len(context.memory) == 0
        assert len(context.state) == 0
        assert len(context.last_results) == 0
        assert context.current_input is None
        assert context.iteration == 0

    def test_step_result_management(self, context):
        """Test storing and retrieving step results."""
        context.set_step_result("step1", "result1")
        context.set_step_result("step2", "result2")

        assert context.get_step_result("step1") == "result1"
        assert context.get_step_result("step2") == "result2"
        assert context.get_step_result("nonexistent") is None

    def test_iteration_counting(self, context):
        """Test iteration counter functionality."""
        assert context.iteration == 0

        count1 = context.increment_iteration()
        assert count1 == 1
        assert context.iteration == 1

        count2 = context.increment_iteration()
        assert count2 == 2
        assert context.iteration == 2

    def test_state_management(self, context):
        """Test state dictionary operations."""
        context.state["key1"] = "value1"
        context.state["key2"] = {"nested": "value2"}

        assert context.state["key1"] == "value1"
        assert context.state["key2"]["nested"] == "value2"

        context.state.clear()
        assert len(context.state) == 0

    def test_memory_integration(self, context, memory):
        """Test memory system integration."""
        memory.store("key1", "value1")
        memory.store("key2", {"data": "value2"})

        context.memory.append(
            {"key": "key1", "value": memory.retrieve("key1")}
        )
        context.memory.append(
            {"key": "key2", "value": memory.retrieve("key2")}
        )

        assert len(context.memory) == 2
        assert context.memory[0]["value"] == "value1"
        assert context.memory[1]["value"]["data"] == "value2"

    def test_context_clear(self, context):
        """Test clearing all context data."""
        context.state["key"] = "value"
        context.memory.append({"data": "test"})
        context.set_step_result("step", "result")
        context.current_input = "input"
        context.increment_iteration()

        context.clear()

        assert len(context.state) == 0
        assert len(context.memory) == 0
        assert len(context.last_results) == 0
        assert context.current_input is None
        assert context.iteration == 0

    def test_result_persistence(self, context):
        """Test persistence of step results across iterations."""
        context.set_step_result("persistent_step", "result")
        context.increment_iteration()

        assert context.get_step_result("persistent_step") == "result"

        context.increment_iteration()
        assert context.get_step_result("persistent_step") == "result"

    def test_complex_state_handling(self, context):
        """Test handling of complex nested state structures."""
        complex_state = {
            "level1": {
                "level2": {"data": [1, 2, 3], "metadata": {"type": "test"}}
            }
        }

        context.state["complex"] = complex_state

        assert context.state["complex"]["level1"]["level2"]["data"][1] == 2
        assert (
            context.state["complex"]["level1"]["level2"]["metadata"]["type"]
            == "test"
        )

        context.state["complex"]["level1"]["level2"]["data"].append(4)
        assert len(context.state["complex"]["level1"]["level2"]["data"]) == 4

    def test_concurrent_updates(self, context):
        """Test handling concurrent updates to context."""
        from concurrent.futures import ThreadPoolExecutor

        def update_func(key: str, value: str):
            context.state[key] = value
            context.set_step_result(key, value)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(3):
                futures.append(
                    executor.submit(update_func, f"key{i}", f"value{i}")
                )

        for i in range(3):
            assert context.state[f"key{i}"] == f"value{i}"
            assert context.get_step_result(f"key{i}") == f"value{i}"

    def test_memory_limit_handling(self, context):
        """Test handling of memory size limits."""
        for i in range(1000):
            context.memory.append({"index": i, "data": f"data{i}"})

        assert len(context.memory) <= 1000

        assert context.memory[0]["index"] == 0

    def test_error_handling(self, context):
        """Test error handling in context operations."""
        assert context.get_step_result("nonexistent_step") is None

        context.state["test_key"] = "test_value"
        del context.state["test_key"]
        assert "test_key" not in context.state

        with pytest.raises(KeyError):
            del context.state["nonexistent_key"]

        large_list = [{"data": i} for i in range(10000)]
        for item in large_list:
            context.memory.append(item)
        assert len(context.memory) <= 10000

        context.set_step_result("step1", "result1")
        context.state["state1"] = "value1"
        context.memory.append({"mem1": "value1"})

        context.clear()
        assert isinstance(context.last_results, dict)
        assert isinstance(context.memory, list)
        assert isinstance(context.state, dict)
        assert len(context.last_results) == 0
        assert len(context.memory) == 0
        assert len(context.state) == 0

        with pytest.raises(KeyError):
            context.last_results["nonexistent_step"]

        context.state["dict_value"] = {"key": "value"}
        assert isinstance(context.state["dict_value"], dict)

        context.set_step_result("test_step", {"result": "value"})
        assert isinstance(context.get_step_result("test_step"), dict)
