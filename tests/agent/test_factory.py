from unittest.mock import Mock

import pytest

from clientai.agent import Agent, ToolConfig, ToolSelectionConfig
from clientai.agent.core.factory import create_agent
from clientai.exceptions import ClientAIError


def add_numbers(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y


@pytest.fixture
def mock_client():
    client = Mock()
    client.generate_text = Mock(side_effect=lambda prompt, **kwargs: prompt)
    client.provider = Mock()
    client.provider.__class__.__module__ = "clientai.openai.provider"
    return client


def test_basic_agent_creation(mock_client):
    """Test creating a basic agent with minimal configuration."""
    agent = create_agent(
        client=mock_client,
        role="test",
        system_prompt="Test prompt",
        model="test-model",
    )

    assert isinstance(agent, Agent)
    assert agent._default_model.name == "test-model"
    assert len(agent.workflow_manager.get_steps()) == 1
    assert "test_step" in agent.workflow_manager.get_steps()


def test_agent_with_tools(mock_client):
    """Test creating an agent with function tools."""
    agent = create_agent(
        client=mock_client,
        role="calculator",
        system_prompt="Test prompt",
        model="test-model",
        tools=[add_numbers, multiply_numbers],
    )

    tools = agent.get_tools()
    assert len(tools) == 2
    tool_names = {tool.name for tool in tools}
    assert "add_numbers" in tool_names
    assert "multiply_numbers" in tool_names


def test_agent_with_tool_config(mock_client):
    """Test creating an agent with ToolConfig objects."""
    tool_config = ToolConfig(tool=add_numbers, scopes=["think"])
    agent = create_agent(
        client=mock_client,
        role="calculator",
        system_prompt="Test prompt",
        model="test-model",
        tools=[tool_config],
    )

    tools = agent.get_tools("think")
    assert len(tools) == 1
    assert tools[0].name == "add_numbers"


def test_agent_with_tool_selection_config(mock_client):
    """Test creating an agent with tool selection configuration."""
    tool_config = ToolSelectionConfig(
        confidence_threshold=0.8, max_tools_per_step=2
    )
    agent = create_agent(
        client=mock_client,
        role="calculator",
        system_prompt="Test prompt",
        model="test-model",
        tool_selection_config=tool_config,
        tools=[add_numbers],
    )

    assert agent._tool_selection_config.confidence_threshold == 0.8
    assert agent._tool_selection_config.max_tools_per_step == 2


def test_agent_with_step_type(mock_client):
    """Test creating agents with different step types."""
    think_agent = create_agent(
        client=mock_client,
        role="analyzer",
        system_prompt="Test prompt",
        model="test-model",
        step="think",
    )

    act_agent = create_agent(
        client=mock_client,
        role="executor",
        system_prompt="Test prompt",
        model="test-model",
        step="act",
    )

    # Check that different step types have different default temperatures
    assert think_agent._default_model.get_parameters().get(
        "temperature", 0
    ) != act_agent._default_model.get_parameters().get("temperature", 0)


def test_agent_with_custom_model_params(mock_client):
    """Test creating an agent with custom model parameters."""
    agent = create_agent(
        client=mock_client,
        role="test",
        system_prompt="Test prompt",
        model="test-model",
        temperature=0.8,
        top_p=0.9,
        max_tokens=100,
    )

    params = agent._default_model.get_parameters()
    assert params["temperature"] == 0.8
    assert params["top_p"] == 0.9
    assert params["max_tokens"] == 100


def test_streaming_agent(mock_client):
    """Test creating an agent with streaming enabled."""
    agent = create_agent(
        client=mock_client,
        role="test",
        system_prompt="Test prompt",
        model="test-model",
        stream=True,
    )

    assert agent._default_model.stream is True


@pytest.mark.parametrize(
    "invalid_input",
    [
        {"role": "", "error": "Role must be a non-empty string"},
        {"role": None, "error": "Role must be a non-empty string"},
        {
            "system_prompt": "",
            "error": "System prompt must be a non-empty string",
        },
        {
            "system_prompt": None,
            "error": "System prompt must be a non-empty string",
        },
        {"model": "", "error": "Model must be a non-empty string"},
        {"model": None, "error": "Model must be a non-empty string"},
        {"temperature": 1.5, "error": "Temperature must be between 0 and 1"},
        {"temperature": -0.1, "error": "Temperature must be between 0 and 1"},
        {"top_p": 1.5, "error": "Top_p must be between 0 and 1"},
        {"top_p": -0.1, "error": "Top_p must be between 0 and 1"},
    ],
)
def test_invalid_inputs(mock_client, invalid_input):
    """Test that invalid inputs raise appropriate errors."""
    error_msg = invalid_input.pop("error")
    base_args = {
        "client": mock_client,
        "role": "test",
        "system_prompt": "Test prompt",
        "model": "test-model",
    }

    with pytest.raises(ValueError, match=error_msg):
        create_agent(**{**base_args, **invalid_input})


def test_conflicting_tool_configs(mock_client):
    """Test that conflicting tool configurations raise an error."""
    with pytest.raises(
        ValueError,
        match="Cannot specify both tool_selection_config "
        "and individual tool parameters",
    ):
        create_agent(
            client=mock_client,
            role="test",
            system_prompt="Test prompt",
            model="test-model",
            tool_selection_config=ToolSelectionConfig(),
            tool_confidence=0.8,
        )


def test_agent_execution(mock_client):
    """Test that the created agent can execute successfully."""
    agent = create_agent(
        client=mock_client,
        role="test",
        system_prompt="Test prompt",
        model="test-model",
    )

    result = agent.run("test input")
    assert isinstance(result, str)
    assert len(agent.context.last_results) == 1


def test_client_error_handling(mock_client):
    """Test that client errors are handled properly."""
    mock_client.generate_text.side_effect = ClientAIError("API Error")
    agent = create_agent(
        client=mock_client,
        role="test",
        system_prompt="Test prompt",
        model="test-model",
    )

    with pytest.raises(ClientAIError):
        agent.run("test input")


def test_invalid_step_type(mock_client):
    """Test that invalid step types raise an error."""
    with pytest.raises(ValueError, match="Step must be one of"):
        create_agent(
            client=mock_client,
            role="test",
            system_prompt="Test prompt",
            model="test-model",
            step="invalid_step_type!",
        )
