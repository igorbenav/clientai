from unittest.mock import Mock

import pytest

from clientai.agent.config import ToolConfig
from clientai.agent.core.factory import create_agent
from clientai.agent.exceptions import AgentError
from clientai.agent.tools import ToolSelectionConfig


@pytest.fixture
def mock_client():
    return Mock()


def test_basic_agent_creation(mock_client):
    """Test basic agent creation with minimal configuration."""
    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
    )

    assert agent is not None
    assert agent._client == mock_client
    assert agent._default_model.name == "gpt-4"
    assert len(agent.workflow_manager.get_steps()) == 1


def test_agent_creation_with_tools(mock_client):
    """Test agent creation with tool configuration."""

    def mock_tool(x: int) -> int:
        """Test tool."""
        return x * 2  # pragma: no cover

    # Test with basic tool
    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
        tools=[mock_tool],
    )
    assert len(agent.get_tools()) == 1

    # Test with ToolConfig
    tool_config = ToolConfig(tool=mock_tool, scopes=["think"])
    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
        tools=[tool_config],
    )
    assert len(agent.get_tools("think")) == 1


def test_agent_creation_with_step_types(mock_client):
    """Test agent creation with different step types."""
    step_types = ["think", "act", "observe", "synthesize"]

    for step_type in step_types:
        agent = create_agent(
            client=mock_client,
            role="test_role",
            system_prompt="Test prompt",
            model="gpt-4",
            step=step_type,
        )
        steps = agent.workflow_manager.get_steps()
        assert len(steps) == 1
        step = next(iter(steps.values()))
        assert step.step_type.name.lower() == step_type


def test_agent_creation_with_tool_selection_config(mock_client):
    """Test agent creation with tool selection configuration."""
    tool_config = ToolSelectionConfig(
        confidence_threshold=0.8, max_tools_per_step=2
    )

    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
        tool_selection_config=tool_config,
    )

    assert agent._tool_selection_config.confidence_threshold == 0.8
    assert agent._tool_selection_config.max_tools_per_step == 2


def test_agent_creation_with_model_params(mock_client):
    """Test agent creation with various model parameters."""
    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
        temperature=0.7,
        top_p=0.9,
        stream=True,
        extra_param="value",
    )

    # Check model configuration parameters
    config_dict = agent._default_model.to_dict()
    assert config_dict["name"] == "gpt-4"
    assert config_dict["temperature"] == 0.7
    assert config_dict["top_p"] == 0.9
    assert config_dict["stream"] is True
    assert config_dict["extra_param"] == "value"


def test_agent_creation_errors(mock_client):
    """Test error cases in agent creation."""
    # Test missing required parameters
    with pytest.raises(AgentError, match="Role must be a non-empty string"):
        create_agent(
            client=mock_client,
            role="",
            system_prompt="Test prompt",
            model="gpt-4",
        )

    with pytest.raises(
        AgentError, match="System prompt must be a non-empty string"
    ):
        create_agent(
            client=mock_client,
            role="test_role",
            system_prompt="",
            model="gpt-4",
        )

    # Test invalid tool configuration
    with pytest.raises(AgentError):
        create_agent(
            client=mock_client,
            role="test_role",
            system_prompt="Test prompt",
            model="gpt-4",
            tool_selection_config=ToolSelectionConfig(),
            tool_confidence=0.8,  # Conflict
        )

    # Test invalid temperature
    with pytest.raises(AgentError):
        create_agent(
            client=mock_client,
            role="test_role",
            system_prompt="Test prompt",
            model="gpt-4",
            temperature=2.0,  # Invalid value
        )


def test_agent_streaming_configuration(mock_client):
    """Test agent creation with streaming configuration."""
    # Test streaming enabled
    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
        stream=True,
    )
    steps = agent.workflow_manager.get_steps()
    step = next(iter(steps.values()))
    assert step.stream is True

    # Test streaming disabled
    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
        stream=False,
    )
    steps = agent.workflow_manager.get_steps()
    step = next(iter(steps.values()))
    assert step.stream is False


def test_agent_tool_model_configuration(mock_client):
    """Test agent creation with tool model configuration."""
    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
        tool_model="different-model",
    )

    assert agent._tool_model.name == "different-model"

    # Test tool model inherits from default when not specified
    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
    )

    assert agent._tool_model.name == "gpt-4"


def test_custom_step_creation(mock_client):
    """Test agent creation with custom step type."""
    agent = create_agent(
        client=mock_client,
        role="test_role",
        system_prompt="Test prompt",
        model="gpt-4",
        step="custom_step",
    )

    steps = agent.workflow_manager.get_steps()
    assert len(steps) == 1
    step = next(iter(steps.values()))
    assert step.step_type.name == "ACT"  # Custom steps default to ACT type


def test_sanitized_step_names(mock_client):
    """Test step name sanitization in agent creation."""
    # Test with special characters
    agent = create_agent(
        client=mock_client,
        role="test@role#123",
        system_prompt="Test prompt",
        model="gpt-4",
    )

    steps = agent.workflow_manager.get_steps()
    step_name = next(iter(steps.keys()))
    assert step_name.isidentifier()  # Should be a valid Python identifier
    assert "@" not in step_name
    assert "#" not in step_name
