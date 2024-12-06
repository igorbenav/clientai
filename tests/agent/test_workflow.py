from unittest.mock import Mock

import pytest

from clientai.agent.config.steps import StepConfig
from clientai.agent.core.execution import StepExecutionEngine
from clientai.agent.core.workflow import WorkflowManager
from clientai.agent.steps import StepType, act, run, think
from clientai.exceptions import ClientAIError


class TestWorkflowManager:
    """Test suite for workflow management functionality."""

    @pytest.fixture
    def workflow_manager(self):
        return WorkflowManager()

    @pytest.fixture
    def mock_engine(self):
        engine = Mock(spec=StepExecutionEngine)
        engine.execute_step.side_effect = lambda step, agent, data: data
        return engine

    @pytest.fixture
    def mock_agent(self):
        agent = Mock()
        agent.context.last_results = {}
        agent.custom_run = Mock(return_value="custom_output")
        return agent

    def test_step_registration(self, workflow_manager):
        """Test step registration and retrieval."""

        class TestAgent:
            @think("test_step")
            def test_method(self, data: str) -> str:
                return data

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        assert len(workflow_manager.get_steps()) == 1
        assert workflow_manager.get_step("test_step") is not None

    def test_workflow_execution(
        self, workflow_manager, mock_engine, mock_agent
    ):
        """Test workflow execution with multiple steps."""

        class TestAgent:
            @think("step1")
            def step1(self, data: str) -> str:
                return f"step1_{data}"

            @act("step2")
            def step2(self, data: str) -> str:
                return f"step2_{data}"

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        result = workflow_manager.execute(mock_agent, "input", mock_engine)

        assert mock_engine.execute_step.call_count == 2
        assert result == "input"

    def test_custom_run_method(self, workflow_manager, mock_agent):
        """Test workflow execution with custom run method."""
        original_custom_run = mock_agent.custom_run

        class TestAgent:
            @run()
            def custom_run(self, input_data: str) -> str:
                """Custom run method that accepts input_data."""
                return original_custom_run(self, input_data)

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        result = workflow_manager.execute(mock_agent, "input", None)
        assert result == "custom_output"

    def test_step_dependency_handling(self, workflow_manager):
        """Test step dependency detection and handling."""

        class TestAgent:
            @think("producer")
            def produce(self, data: str) -> str:
                return data

            @act("consumer")
            def consume(self, produced: str) -> str:
                return produced

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        steps = workflow_manager.get_steps()
        assert len(steps) == 2

    def test_error_handling(self, workflow_manager, mock_engine, mock_agent):
        """Test error handling during workflow execution."""
        mock_engine.execute_step.side_effect = ClientAIError("Test error")

        class TestAgent:
            @think("error_step")
            def error_step(self, data: str) -> str:
                return data

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        with pytest.raises(ClientAIError):
            workflow_manager.execute(mock_agent, "input", mock_engine)

    def test_step_type_ordering(self, workflow_manager):
        """Test ordering of different step types in workflow."""

        class TestAgent:
            @think("thinking_step")
            def think(self, data: str) -> str:
                return data

            @act("action_step")
            def action(self, data: str) -> str:
                return data

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        steps = list(workflow_manager.get_steps().values())
        assert len([s for s in steps if s.step_type == StepType.THINK]) == 1
        assert len([s for s in steps if s.step_type == StepType.ACT]) == 1

    def test_workflow_reset(self, workflow_manager):
        """Test workflow reset functionality."""

        class TestAgent:
            @think("step1")
            def step1(self, data: str) -> str:
                return data

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        assert len(workflow_manager.get_steps()) == 1

        workflow_manager.reset()
        assert len(workflow_manager.get_steps()) == 0
        assert workflow_manager._custom_run is None

    def test_get_steps_by_type(self, workflow_manager):
        """Test retrieval of steps by type."""

        class TestAgent:
            @think("think_step")
            def think(self, data: str) -> str:
                return data

            @act("act_step")
            def act(self, data: str) -> str:
                return data

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        think_steps = workflow_manager.get_steps_by_type(StepType.THINK)
        act_steps = workflow_manager.get_steps_by_type(StepType.ACT)

        assert len(think_steps) == 1
        assert len(act_steps) == 1
        assert "think_step" in think_steps
        assert "act_step" in act_steps

    def test_duplicate_step_handling(self, workflow_manager):
        """Test handling of duplicate step names."""

        class TestAgent:
            @think("duplicate")
            def step1(self, data: str) -> str:
                return data

            @act("duplicate")
            def step2(self, data: str) -> str:
                return data

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        assert len(workflow_manager.get_steps()) == 1
        assert workflow_manager.get_step("duplicate") is not None

    def test_conditional_workflow_execution(
        self, workflow_manager, mock_engine, mock_agent
    ):
        """Test conditional workflow execution based on step configuration."""
        mock_engine.reset_mock()

        class TestAgent:
            @think(
                "step1",
                step_config=StepConfig(enabled=False, pass_result=False),
            )
            def disabled_step(self, data: str) -> str:
                return data

            @act(
                "step2", step_config=StepConfig(enabled=True, pass_result=True)
            )
            def enabled_step(self, data: str) -> str:
                return data

        agent = TestAgent()
        workflow_manager.register_class_steps(agent)

        executed_steps = []

        def side_effect(step, agent, data):
            if step.config.enabled:
                executed_steps.append(step.name)
                return data
            return None

        mock_engine.execute_step.side_effect = side_effect

        workflow_manager.execute(mock_agent, "input", mock_engine)

        assert mock_engine.execute_step.call_count == 2
        assert len(executed_steps) == 1
        assert executed_steps[0] == "step2"

        calls = mock_engine.execute_step.call_args_list
        assert calls[0][0][0].name == "step1"
        assert calls[1][0][0].name == "step2"
