import pytest

from clientai.agent.core.context import AgentContext


class TestContextManagement:
    @pytest.fixture
    def context(self):
        return AgentContext()

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

    def test_historical_data_management(self, context):
        """Test storing historical context data."""
        context.memory.append({"step": "step1", "result": "value1"})
        context.memory.append({"step": "step2", "result": {"data": "value2"}})

        assert len(context.memory) == 2
        assert context.memory[0]["result"] == "value1"
        assert context.memory[1]["result"]["data"] == "value2"

    def test_state_management(self, context):
        """Test state dictionary operations."""
        context.state["key1"] = "value1"
        context.state["key2"] = {"nested": "value2"}

        assert context.state["key1"] == "value1"
        assert context.state["key2"]["nested"] == "value2"

        context.state.clear()
        assert len(context.state) == 0

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

        context.memory.append(
            {
                "step": "persistent_step",
                "result": context.get_step_result("persistent_step"),
                "iteration": context.iteration,
            }
        )

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

        context.memory.append(
            {
                "type": "state_change",
                "key": "complex",
                "value": complex_state.copy(),
                "iteration": context.iteration,
            }
        )

    def test_concurrent_updates(self, context):
        """Test handling concurrent updates to context."""
        from concurrent.futures import ThreadPoolExecutor

        def update_func(key: str, value: str):
            context.state[key] = value
            context.set_step_result(key, value)
            context.memory.append(
                {
                    "type": "concurrent_update",
                    "key": key,
                    "value": value,
                    "iteration": context.iteration,
                }
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(3):
                futures.append(
                    executor.submit(update_func, f"key{i}", f"value{i}")
                )

        for i in range(3):
            assert context.state[f"key{i}"] == f"value{i}"
            assert context.get_step_result(f"key{i}") == f"value{i}"

    def test_history_limit_handling(self, context):
        """Test handling of history size limits."""
        MAX_HISTORY = 1000

        for i in range(MAX_HISTORY + 100):
            context.memory.append({"index": i, "data": f"data{i}"})
            if len(context.memory) > MAX_HISTORY:
                context.memory.pop(0)

        assert len(context.memory) == MAX_HISTORY
        assert context.memory[0]["index"] == 100
