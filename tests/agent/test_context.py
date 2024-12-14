from datetime import datetime

import pytest

from clientai.agent.core import AgentContext


@pytest.fixture
def context():
    """Provide a fresh AgentContext instance for each test."""
    return AgentContext()


def test_context_initialization():
    """Test context initialization with various configurations."""
    # Default initialization
    context = AgentContext()
    assert len(context.memory) == 0
    assert len(context.state) == 0
    assert context.current_input is None
    assert context.iteration == 0
    assert context.max_history_size == 10

    # Custom initialization
    context = AgentContext(max_history_size=5)
    assert context.max_history_size == 5

    # Initialize with non-default values
    context = AgentContext(
        memory=[{"step": "init"}], state={"key": "value"}, max_history_size=3
    )
    assert len(context.memory) == 1
    assert context.state["key"] == "value"
    assert context.max_history_size == 3


def test_context_input_setting_comprehensive(context):
    """Test comprehensive input setting behavior and edge cases."""
    # Basic input setting
    context.set_input("first input")
    assert context.current_input == "first input"
    assert context.original_input == "first input"

    # Test with different input types
    test_inputs = [
        123,
        {"key": "value"},
        ["list", "of", "items"],
        None,
        "",
        True,
    ]

    for input_data in test_inputs:
        context.set_input(input_data)
        assert context.current_input == input_data
        assert context.original_input == input_data

    # Test input history creation
    context.set_input("first")
    context.set_step_result("step1", "result1")
    context.set_step_result("step2", "result2")
    context.set_input("second")

    last_history = context.conversation_history[-1]
    assert last_history["input"] == "first"
    assert len(last_history["results"]) == 2
    assert all(
        k in last_history
        for k in ["input", "results", "iteration", "timestamp"]
    )


def test_context_result_management_comprehensive(context):
    """Test comprehensive result storage and retrieval behavior."""
    # Test various result types
    test_results = {
        "string_result": "test",
        "int_result": 123,
        "dict_result": {"key": "value"},
        "list_result": [1, 2, 3],
        "none_result": None,
        "bool_result": True,
    }

    for step_name, result in test_results.items():
        context.set_step_result(step_name, result)
        assert context.get_step_result(step_name) == result

    # Test overwriting results
    context.set_step_result("test_step", "original")
    context.set_step_result("test_step", "updated")
    assert context.get_step_result("test_step") == "updated"

    # Test getting non-existent results
    assert context.get_step_result("nonexistent") is None

    # Test result persistence across inputs
    context.set_step_result("persistent", "value")
    context.set_input("new input")
    assert len(context.last_results) == 0
    assert "persistent" not in context.last_results


def test_context_history_management_comprehensive():
    """Test comprehensive history management scenarios."""
    # Test with different max sizes
    context = AgentContext(max_history_size=3)

    # Add interactions
    for i in range(5):  # Add 5 interactions but only keep last 3
        context.set_input(f"input{i}")
        context.set_step_result("step", f"result{i}")
        context.set_input(f"input{i+1}")

    assert len(context.conversation_history) == 3
    # The last three entries should be for input2, input3, and input4
    assert context.conversation_history[0]["input"] == "input2"
    assert context.conversation_history[1]["input"] == "input3"
    assert context.conversation_history[2]["input"] == "input4"

    # Test history data integrity
    context = AgentContext(max_history_size=3)
    test_data = [
        ("input1", {"step1": "result1"}),
        ("input2", {"step2": "result2"}),
        ("input3", {"step3": "result3"}),
    ]

    # Add data to history
    for input_data, results in test_data:
        context.set_input(input_data)
        for step, result in results.items():
            context.set_step_result(step, result)
        context.set_input("next")

    # Verify all data is preserved correctly
    assert len(context.conversation_history) == 3
    for i, (input_data, results) in enumerate(test_data):
        history_entry = context.conversation_history[i]
        assert history_entry["input"] == input_data
        assert all(
            history_entry["results"][k] == v for k, v in results.items()
        )


def test_context_history_formatting_comprehensive(context):
    """Test comprehensive history formatting scenarios."""

    def create_test_interaction(index: int):
        context.set_input(f"input{index}")
        context.set_step_result(f"step{index}a", f"result{index}a")
        context.set_step_result(f"step{index}b", f"result{index}b")
        context.set_input("next")

    # Create multiple interactions
    for i in range(3):
        create_test_interaction(i)

    # Test raw history retrieval
    raw_history = context.get_recent_history(raw=True)
    assert len(raw_history) == 3

    # Test formatted history
    formatted = context.get_recent_history()
    assert isinstance(formatted, str)
    assert all(f"input{i}" in formatted for i in range(3))
    assert all(f"result{i}a" in formatted for i in range(3))
    assert all(f"result{i}b" in formatted for i in range(3))

    # Test partial history retrieval
    partial = context.get_recent_history(n=1)
    assert "input0" not in partial
    assert "input2" in partial

    # Test empty history formatting
    context.clear_all()
    empty = context.get_recent_history()
    assert "No interactions available" in empty

    # Test single interaction formatting
    context.set_input("single")
    single = context.get_recent_history()
    assert "No previous interactions" in single


def test_context_iteration_management(context):
    """Test iteration counter management."""
    assert context.iteration == 0

    # Test multiple increments
    iterations = 5
    expected_values = list(range(1, iterations + 1))
    actual_values = [context.increment_iteration() for _ in range(iterations)]
    assert actual_values == expected_values

    # Test reset behavior
    context.clear()
    assert context.iteration == 0

    # Test persistence across inputs
    context.increment_iteration()
    current_iteration = context.iteration
    context.set_input("new input")
    assert context.iteration == current_iteration


def test_context_timestamp_handling():
    """Test timestamp handling in history entries."""
    context = AgentContext()

    # Create an interaction and verify timestamp exists
    # and is properly formatted
    context.set_input("test")
    context.set_step_result("step", "result")
    context.set_input("next")

    history_entry = context.conversation_history[-1]
    assert "timestamp" in history_entry
    timestamp = history_entry["timestamp"]
    # Verify it can be parsed as a datetime
    try:
        datetime.fromisoformat(timestamp)
    except ValueError:
        pytest.fail("Timestamp is not in valid ISO format")

    # Verify timestamp appears in formatted history
    formatted = context.get_recent_history()
    assert timestamp[:10] in formatted  # Check date part appears in formatting


def test_context_memory_management(context):
    """Test memory management and cleanup."""
    # Test memory usage patterns
    large_data = ["data" for _ in range(1000)]

    # Add large data to different context areas
    context.memory.extend([{"large": large_data} for _ in range(5)])
    context.state["large"] = large_data
    context.set_step_result("large_step", large_data)

    # Verify cleanup
    context.clear()
    assert len(context.memory) == 0
    assert len(context.state) == 0
    assert len(context.last_results) == 0

    # Test memory cleanup with history retention
    context.set_input("test")
    context.set_step_result("large_step", large_data)
    context.set_input("next")

    context.clear()
    assert len(context.conversation_history) == 1
    assert (
        context.conversation_history[-1]["results"]["large_step"] == large_data
    )


def test_context_max_history_size():
    """Test max history size validation and management."""
    # Test setting max history size
    context = AgentContext(max_history_size=5)
    assert context.max_history_size == 5

    # Test updating max history size
    context.set_max_history_size(3)
    assert context.max_history_size == 3

    # Test that negative values raise ValueError
    with pytest.raises(ValueError):
        context.set_max_history_size(-1)

    # Test that zero is allowed
    context.set_max_history_size(0)
    assert context.max_history_size == 0
