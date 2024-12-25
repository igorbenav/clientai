from typing import Iterator
from unittest.mock import Mock

import pytest

from clientai.agent.core import AgentContext, WorkflowManager
from clientai.agent.exceptions import StepError
from clientai.agent.steps.base import Step, StepConfig
from clientai.agent.steps.decorators import (
    act,
    observe,
    run,
    synthesize,
    think,
)
from clientai.agent.steps.types import StepType


def test_workflow_initialization():
    """Test workflow manager initialization."""
    workflow = WorkflowManager()
    assert len(workflow.get_steps()) == 0
    assert workflow._custom_run is None


def test_workflow_step_registration():
    """Test registering steps with the workflow."""

    class TestAgent:
        @think("analyze")
        def analyze(self, input_data: str) -> str:
            return f"Analyzing: {input_data}"  # pragma: no cover

        @act("process")
        def process(self, analysis: str) -> str:
            return f"Processing: {analysis}"  # pragma: no cover

        @observe("gather")
        def gather(self, query: str) -> str:
            return f"Gathering: {query}"  # pragma: no cover

    workflow = WorkflowManager()
    agent = TestAgent()
    workflow.register_class_steps(agent)

    steps = workflow.get_steps()
    assert len(steps) == 3
    assert "analyze" in steps
    assert "process" in steps
    assert "gather" in steps

    # Verify step types
    assert steps["analyze"].step_type == StepType.THINK
    assert steps["process"].step_type == StepType.ACT
    assert steps["gather"].step_type == StepType.OBSERVE


def test_workflow_execution_parameter_passing():
    """Test parameter passing between workflow steps."""
    mock_engine = Mock()
    mock_engine.execute_step.side_effect = ["First", "Second", "Third"]

    class TestAgent:
        def __init__(self):
            self.context = AgentContext()
            self.execution_engine = mock_engine

        @think
        def step1(self, input_data: str) -> str:
            return "First step"  # pragma: no cover

        @act
        def step2(self, prev_result: str) -> str:
            return "Second step"  # pragma: no cover

        @synthesize
        def step3(self, latest: str) -> str:
            return "Final step"  # pragma: no cover

    workflow = WorkflowManager()
    agent = TestAgent()
    workflow.register_class_steps(agent)

    agent.context.set_input("test input")
    result = workflow.execute(
        agent=agent, input_data="test input", engine=mock_engine
    )
    assert result == "Third"

    # Verify parameters are passed correctly
    calls = mock_engine.execute_step.call_args_list
    assert len(calls) == 3

    # First step gets input data
    first_call = calls[0]
    assert len(first_call.args) >= 2
    assert isinstance(first_call.args[0], Step)
    assert "test input" in str(first_call.args)

    # Second step gets result from first step
    second_call = calls[1]
    assert len(second_call.args) >= 2
    assert isinstance(second_call.args[0], Step)

    # Third step gets result from second step
    third_call = calls[2]
    assert len(third_call.args) >= 2
    assert isinstance(third_call.args[0], Step)


def test_custom_run_method():
    """Test workflow with custom run method."""
    mock_engine = Mock()

    class TestAgent:
        def __init__(self):
            self.context = AgentContext()
            self.execution_engine = mock_engine

        @run
        def custom_run(self, input_data: str) -> str:
            return f"Custom: {input_data}"

        @think
        def analyze(self, data: str) -> str:
            return "Analysis"  # pragma: no cover

    workflow = WorkflowManager()
    agent = TestAgent()
    workflow.register_class_steps(agent)

    result = workflow.execute(
        agent=agent, input_data="test", engine=mock_engine
    )
    assert result == "Custom: test"
    assert (
        mock_engine.execute_step.call_count == 0
    )  # Custom run bypasses normal execution


def test_streaming_configuration():
    """Test streaming configuration handling."""
    mock_engine = Mock()
    mock_engine.execute_step.return_value = iter(["chunk1", "chunk2"])

    class TestAgent:
        def __init__(self):
            self.context = AgentContext()

        @think(stream=True)
        def analyze(self, data: str) -> str:
            return "Analysis"  # pragma: no cover

        @act(stream=False)
        def process(self, analysis: str) -> str:
            return "Result"  # pragma: no cover

    workflow = WorkflowManager()
    agent = TestAgent()
    workflow.register_class_steps(agent)

    # Test with stream override
    result = workflow.execute(
        agent=agent,
        input_data="test",
        engine=mock_engine,
        stream_override=True,
    )
    assert isinstance(result, Iterator)

    # Test without override
    mock_engine.execute_step.return_value = "Regular result"
    result = workflow.execute(
        agent=agent, input_data="test", engine=mock_engine
    )
    assert isinstance(result, str)


def test_step_type_filtering():
    """Test filtering steps by type."""

    class TestAgent:
        @think
        def think1(self, data: str) -> str:
            return "Think 1"  # pragma: no cover

        @think
        def think2(self, data: str) -> str:
            return "Think 2"  # pragma: no cover

        @act
        def act1(self, data: str) -> str:
            return "Act 1"  # pragma: no cover

    workflow = WorkflowManager()
    agent = TestAgent()
    workflow.register_class_steps(agent)

    think_steps = workflow.get_steps_by_type(StepType.THINK)
    act_steps = workflow.get_steps_by_type(StepType.ACT)

    assert len(think_steps) == 2
    assert len(act_steps) == 1
    assert all(
        step.step_type == StepType.THINK for step in think_steps.values()
    )
    assert all(step.step_type == StepType.ACT for step in act_steps.values())


def test_error_handling():
    """Test comprehensive error handling scenarios."""
    mock_engine = Mock()

    class TestAgent:
        def __init__(self):
            self.context = AgentContext()
            self.execution_engine = mock_engine

        @think("step1")
        def required_step1(self, data: str) -> str:
            return "Required 1"  # pragma: no cover

        @act("optional")
        def optional_step(self, data: str) -> str:
            return "Optional"  # pragma: no cover

        @synthesize("final")  # Adding final step to ensure optional isn't last
        def final_step(self, data: str) -> str:
            return "Final"  # pragma: no cover

    workflow = WorkflowManager()
    agent = TestAgent()
    workflow.register_class_steps(agent)

    # Configure the optional step to be non-required
    optional_step = workflow.get_step("optional")
    object.__setattr__(
        optional_step, "config", StepConfig(required=False, pass_result=True)
    )

    # First test - all steps succeed
    mock_engine.execute_step.side_effect = [
        "First Result",
        "Second Result",
        "Final Result",
    ]
    result = workflow.execute(
        agent=agent, input_data="test", engine=mock_engine
    )
    assert result == "Final Result"

    # Second test - required step fails
    mock_engine.execute_step.side_effect = StepError("Required step failed")
    agent.context.clear()
    with pytest.raises(StepError, match="Required step failed"):
        workflow.execute(agent=agent, input_data="test", engine=mock_engine)

    # Third test - optional step fails
    def side_effect(step, *args, **kwargs):
        if step.name == "optional":
            raise StepError("Optional step failed")
        elif step.name == "step1":
            return "First Success"
        else:
            return "Final Success"

    mock_engine.execute_step.side_effect = side_effect
    agent.context.clear()
    result = workflow.execute(
        agent=agent, input_data="test", engine=mock_engine
    )
    assert result == "Final Success"


def test_parameter_validation():
    """Test parameter validation between steps."""
    mock_engine = Mock()

    class TestAgent:
        def __init__(self):
            self.context = AgentContext()
            self.execution_engine = mock_engine

        @think
        def step1(self) -> str:
            return "No params"  # pragma: no cover

        @act
        def step2(self, result: str, extra: str, more: str) -> str:
            return "Three params"  # pragma: no cover

    workflow = WorkflowManager()
    agent = TestAgent()
    workflow.register_class_steps(agent)

    # Should raise error due to insufficient previous results
    mock_engine.execute_step.side_effect = ["First Result"]

    with pytest.raises(ValueError) as exc_info:
        workflow.execute(agent=agent, input_data="test", engine=mock_engine)
    assert "parameters" in str(exc_info.value)


def test_workflow_state_management():
    """Test workflow state management and context updates."""
    mock_engine = Mock()
    mock_engine.execute_step.side_effect = ["Result 1", "Result 2"]

    class TestAgent:
        def __init__(self):
            self.context = AgentContext()

        @think
        def step1(self, data: str) -> str:
            return "First"  # pragma: no cover

        @act
        def step2(self, prev: str) -> str:
            return "Second"  # pragma: no cover

    workflow = WorkflowManager()
    agent = TestAgent()
    workflow.register_class_steps(agent)

    workflow.execute(agent=agent, input_data="test", engine=mock_engine)

    # Verify context updates
    assert "step1" in agent.context.last_results
    assert "step2" in agent.context.last_results
    assert agent.context.last_results["step1"] == "Result 1"
    assert agent.context.last_results["step2"] == "Result 2"


def test_invalid_step_configurations():
    """Test handling of invalid step configurations."""

    def dummy_func(x: str) -> str:
        return x  # pragma: no cover

    # Test with empty name
    with pytest.raises(ValueError) as exc_info:
        Step(func=dummy_func, step_type=StepType.THINK, name="")
    assert "Step name cannot be empty" in str(exc_info.value)

    # Test with invalid identifier
    with pytest.raises(ValueError) as exc_info:
        Step(func=dummy_func, step_type=StepType.THINK, name="123-invalid")
    assert "must be a valid Python identifier" in str(exc_info.value)

    # Test with non-callable
    with pytest.raises(ValueError) as exc_info:
        Step(func="not-a-function", step_type=StepType.THINK, name="test_step")
    assert "func must be a callable" in str(exc_info.value)


def test_workflow_reset_and_reuse():
    """Test workflow reset and reuse functionality."""
    mock_engine = Mock()

    class TestAgent:
        def __init__(self):
            self.context = AgentContext()

        @think
        def step(self, data: str) -> str:
            return "Result"  # pragma: no cover

    workflow = WorkflowManager()
    agent = TestAgent()
    workflow.register_class_steps(agent)

    # First execution
    workflow.execute(agent=agent, input_data="test1", engine=mock_engine)
    assert agent.context.iteration == 1

    # Reset and second execution
    workflow.reset()
    assert len(workflow.get_steps()) == 0

    workflow.register_class_steps(agent)
    workflow.execute(agent=agent, input_data="test2", engine=mock_engine)
    assert agent.context.iteration == 1  # Should reset to 1 after clear
