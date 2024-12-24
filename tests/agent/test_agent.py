from unittest.mock import Mock, patch

import pytest

from clientai.agent.config import ModelConfig
from clientai.agent.core import Agent
from clientai.agent.core.context import AgentContext
from clientai.agent.exceptions import AgentError, ToolError, WorkflowError
from clientai.agent.steps.decorators import act, observe, think
from clientai.agent.tools import ToolSelectionConfig


class MockAgent(Agent):
    @think
    def analyze(self, input_data: str) -> str:
        return f"Analyzing: {input_data}"

    @act
    def process(self, analysis: str) -> str:
        return f"Processing: {analysis}"


@pytest.fixture
def mock_client():
    return Mock()


@pytest.fixture
def base_agent(mock_client):
    return MockAgent(
        client=mock_client,
        default_model=ModelConfig(
            name="gpt-4",
            temperature=0.7,
        ),
    )


def test_agent_initialization(mock_client):
    """Test comprehensive agent initialization with various configurations."""
    # Test basic initialization
    agent = MockAgent(client=mock_client, default_model="gpt-4")
    assert agent._client == mock_client
    assert agent._default_model.name == "gpt-4"

    # Test with ModelConfig
    model_config = ModelConfig(name="gpt-4", temperature=0.7)
    agent = MockAgent(client=mock_client, default_model=model_config)
    assert agent._default_model.temperature == 0.7

    # Test with tool configuration
    tool_config = ToolSelectionConfig(confidence_threshold=0.8)
    agent = MockAgent(
        client=mock_client,
        default_model="gpt-4",
        tool_selection_config=tool_config,
    )
    assert agent._tool_selection_config.confidence_threshold == 0.8

    # Test custom history size
    agent = MockAgent(
        client=mock_client, default_model="gpt-4", max_history_size=5
    )
    assert agent.context.max_history_size == 5


def test_agent_initialization_errors(mock_client):
    """Test agent initialization error cases."""
    # Test empty model name
    with pytest.raises(AgentError, match="default_model must be specified"):
        MockAgent(client=mock_client, default_model="")

    # Test invalid model configuration
    with pytest.raises(AgentError):
        MockAgent(client=mock_client, default_model={"invalid": "config"})

    # Test conflicting tool configurations
    with pytest.raises(AgentError):
        MockAgent(
            client=mock_client,
            default_model="gpt-4",
            tool_selection_config=ToolSelectionConfig(),
            tool_confidence=0.8,  # Conflict
        )


def test_agent_run_method(base_agent):
    """Test agent's run method with various scenarios."""
    mock_client = base_agent._client
    mock_client.generate_text.return_value = "Generated response"

    with patch.object(
        base_agent.execution_engine, "_current_agent", base_agent
    ):
        # Test basic execution
        result = base_agent.run("test input")
        assert isinstance(result, str)
        assert mock_client.generate_text.called

        # Test with different input types
        result = base_agent.run(123)  # Should handle non-string input
        assert isinstance(result, str)

        # Test error propagation
        mock_client.generate_text.side_effect = Exception("API Error")
        with pytest.raises(WorkflowError):
            base_agent.run("error test")


def test_agent_streaming(base_agent):
    """Test agent's streaming functionality comprehensively."""
    mock_client = base_agent._client

    with patch.object(
        base_agent.execution_engine, "_current_agent", base_agent
    ):
        # Test basic streaming
        mock_client.generate_text.side_effect = lambda *args, **kwargs: iter(
            ["chunk1", "chunk2"]
        )
        result = base_agent.run("test input", stream=True)
        assert hasattr(result, "__iter__")
        chunks = list(result)
        assert chunks == ["chunk1", "chunk2"]

        # Test streaming override
        mock_client.generate_text.side_effect = (
            lambda *args, **kwargs: "non-stream response"
        )
        result = base_agent.run("test input", stream=False)
        assert isinstance(result, str)

        # Test error in stream
        mock_client.generate_text.side_effect = lambda *args, **kwargs: iter(
            []
        )
        result = base_agent.run("test input", stream=True)
        assert list(result) == []


def test_agent_workflow_execution(base_agent):
    """Test complete workflow execution with different patterns."""
    mock_client = base_agent._client

    with patch.object(
        base_agent.execution_engine, "_current_agent", base_agent
    ):
        # Test sequential execution
        mock_client.generate_text.side_effect = [
            "Analysis result",
            "Final result",
        ]
        result = base_agent.run("test input")
        assert isinstance(result, str)
        assert mock_client.generate_text.call_count == 2

        # Verify execution order
        calls = mock_client.generate_text.call_args_list
        assert "Analyzing" in calls[0][0][0]
        assert "Processing" in calls[1][0][0]

        # Test workflow with error
        mock_client.generate_text.side_effect = [
            "Analysis result",
            Exception("Step error"),
        ]
        with pytest.raises(WorkflowError):
            base_agent.run("error test")


class ComplexAgent(Agent):
    """Test agent with more complex workflow patterns."""

    @think
    def analyze(self, input_data: str) -> str:
        return f"Analyzing: {input_data}"

    @observe
    def gather(self, analysis: str) -> str:
        return f"Gathering data for: {analysis}"

    @act  # Changed from synthesize to avoid parameter order issues
    def summarize(self, data: str) -> str:
        return f"Summary of {data}"


def test_complex_workflow_execution(mock_client):
    """Test execution of more complex workflow patterns."""
    agent = ComplexAgent(client=mock_client, default_model="gpt-4")

    with patch.object(agent.execution_engine, "_current_agent", agent):
        mock_client.generate_text.side_effect = [
            "Analysis result",
            "Gathered data",
            "Final summary",
        ]

        result = agent.run("test input")
        assert isinstance(result, str)
        assert mock_client.generate_text.call_count == 3

        # Verify step execution order
        calls = mock_client.generate_text.call_args_list
        assert "Analyzing" in str(calls[0])
        assert "Gathering" in str(calls[1])
        assert "Summary" in str(calls[2])


def test_agent_context_management(mock_client):
    """Test agent's context management and state tracking."""
    agent = MockAgent(client=mock_client, default_model="gpt-4")

    # Test context initialization
    assert isinstance(agent.context, AgentContext)
    assert len(agent.context.memory) == 0

    # Test state management
    agent.context.state["test_key"] = "test_value"
    assert agent.context.state["test_key"] == "test_value"

    # Test context reset
    agent.reset_context()
    assert "test_key" not in agent.context.state
    assert len(agent.context.memory) == 0

    # Test history tracking
    with patch.object(agent.execution_engine, "_current_agent", agent):
        mock_client.generate_text.return_value = "response"
        # First interaction
        agent.run("test input 1")
        # Second interaction
        agent.run("test input 2")
        # Check conversation history - should have at least first interaction
        assert len(agent.context.conversation_history) >= 1
        # Verify the history content
        latest_interaction = agent.context.conversation_history[-1]
        assert "test input 1" == latest_interaction["input"]


def test_agent_tool_handling(mock_client):
    """Test agent's tool management capabilities."""

    def mock_tool(x: int) -> int:
        """Test tool."""
        return x * 2

    # Test tool registration
    agent = MockAgent(
        client=mock_client, default_model="gpt-4", tools=[mock_tool]
    )

    assert len(agent.get_tools()) == 1

    # Test tool execution
    result = agent.use_tool("mock_tool", 5)
    assert result == 10

    # Test tool scope filtering
    think_tools = agent.get_tools("think")
    assert len(think_tools) > 0

    # Test invalid tool execution
    with pytest.raises(ToolError):
        agent.use_tool("nonexistent_tool")
