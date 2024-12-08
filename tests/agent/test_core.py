from unittest.mock import Mock

import pytest

from clientai.agent import Agent, ModelConfig, ToolConfig, act, observe, think
from clientai.agent.core.context import AgentContext
from clientai.agent.exceptions import ToolError
from clientai.agent.tools import tool
from clientai.exceptions import ClientAIError


@tool()
def mock_calculator(x: int, y: int) -> int:
    """A simple calculator tool for testing."""
    return x + y


class MockTestingAgent(Agent):
    """Test agent implementation."""

    think_called: bool = False
    act_called: bool = False
    observe_called: bool = False

    @think
    def analyze_data(self, data: str) -> str:
        self.think_called = True
        return f"Analyzed: {data}"

    @act
    def process_analysis(self, analysis: str) -> str:
        self.act_called = True
        return f"Processed: {analysis}"

    @observe
    def check_result(self, result: str) -> str:
        self.observe_called = True
        return f"Checked: {result}"

    def reset(self) -> None:
        super().reset()
        self.think_called = False
        self.act_called = False
        self.observe_called = False


@pytest.fixture
def mock_client():
    client = Mock()
    client.generate_text = Mock(side_effect=lambda prompt, **kwargs: prompt)
    client.provider = Mock()
    client.provider.__class__.__module__ = "clientai.openai.provider"
    return client


@pytest.fixture
def basic_agent(mock_client):
    return MockTestingAgent(client=mock_client, default_model="test-model")


@pytest.fixture
def agent_with_tools(mock_client):
    return MockTestingAgent(
        client=mock_client,
        default_model="test-model",
        tools=[ToolConfig(mock_calculator, ["think"])],
    )


def test_agent_initialization(basic_agent):
    assert isinstance(basic_agent.context, AgentContext)
    assert basic_agent._default_model.name == "test-model"
    assert len(basic_agent.workflow_manager.get_steps()) == 3


def test_agent_with_model_config():
    config = ModelConfig(name="test-model", temperature=0.7, max_tokens=100)
    client = Mock()
    agent = MockTestingAgent(client=client, default_model=config)
    assert agent._default_model.name == "test-model"
    assert agent._default_model.get_parameters()["temperature"] == 0.7
    assert agent._default_model.get_parameters()["max_tokens"] == 100


def test_agent_workflow_execution(basic_agent):
    basic_agent.run("test data")
    assert basic_agent.think_called
    assert basic_agent.act_called
    assert basic_agent.observe_called
    assert (
        "Analyzed: test data"
        in basic_agent.context.last_results["analyze_data"]
    )
    assert (
        "Processed: Analyzed: test data"
        in basic_agent.context.last_results["process_analysis"]
    )
    assert (
        "Checked: Processed: Analyzed: test data"
        in basic_agent.context.last_results["check_result"]
    )


def test_agent_context_management(basic_agent):
    basic_agent.run("test data")
    assert len(basic_agent.context.last_results) > 0
    assert "analyze_data" in basic_agent.context.last_results
    assert "process_analysis" in basic_agent.context.last_results
    assert "check_result" in basic_agent.context.last_results


def test_agent_tool_registration(agent_with_tools):
    tools = agent_with_tools.get_tools("think")
    assert len(tools) == 1
    assert tools[0].name == "mock_calculator"


def test_agent_tool_execution(agent_with_tools):
    result = agent_with_tools.use_tool("mock_calculator", 2, 3)
    assert result == 5


def test_agent_reset(basic_agent):
    basic_agent.run("test data")
    assert len(basic_agent.context.last_results) > 0
    assert basic_agent.think_called
    basic_agent.reset()
    assert len(basic_agent.context.last_results) == 0
    assert not basic_agent.think_called
    assert not basic_agent.act_called
    assert not basic_agent.observe_called


def test_invalid_tool_execution(basic_agent):
    with pytest.raises(ToolError, match="Tool 'nonexistent_tool' not found"):
        basic_agent.use_tool("nonexistent_tool")


def test_client_error_handling(basic_agent):
    basic_agent._client.generate_text.side_effect = ClientAIError("API Error")
    with pytest.raises(ClientAIError):
        basic_agent.run("test data")


def test_agent_with_custom_model_kwargs():
    client = Mock()
    agent = MockTestingAgent(
        client=client,
        default_model="test-model",
        temperature=0.8,
        max_tokens=150,
    )
    assert agent._default_model_kwargs["temperature"] == 0.8
    assert agent._default_model_kwargs["max_tokens"] == 150
