from typing import Optional

import pytest

from clientai.agent.config.tools import ToolConfig
from clientai.agent.tools import Tool, tool
from clientai.agent.tools.registry import ToolRegistry
from clientai.agent.tools.types import ToolScope


@tool()
def calculator(x: int, y: int) -> int:
    """Simple calculator tool."""
    return x + y


@tool(name="custom_printer", description="Custom printing tool")
def printer(message: str, prefix: Optional[str] = None) -> str:
    """Print with optional prefix."""
    return f"{prefix}: {message}" if prefix else message


class TestToolDecorator:
    def test_basic_tool_creation(self):
        """Test basic tool creation with minimal configuration."""

        @tool()
        def sample(x: int) -> int:
            """Sample tool."""
            return x * 2

        assert isinstance(sample, Tool)
        assert sample.name == "sample"
        assert "Sample tool" in sample.description
        assert "x: int" in sample.signature_str
        assert "-> int" in sample.signature_str

    def test_tool_with_custom_config(self):
        """Test tool creation with custom name and description."""

        @tool(name="custom_tool", description="Custom description")
        def sample(x: int) -> int:
            return x * 2

        assert sample.name == "custom_tool"
        assert sample.description == "Custom description"

    def test_tool_execution(self):
        """Test that decorated tool can still be executed."""
        result = calculator(2, 3)
        assert result == 5

    def test_tool_with_default_params(self):
        """Test tool with default parameters."""
        result = printer("hello", prefix="test")
        assert result == "test: hello"
        result = printer("hello")
        assert result == "hello"


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_register_tool(self, registry):
        """Test basic tool registration."""
        config = ToolConfig(calculator, ["think"])
        registry.register(config)
        assert "calculator" in registry
        assert len(registry) == 1

    def test_register_duplicate_tool(self, registry):
        """Test that registering duplicate tools raises error."""
        config = ToolConfig(calculator, ["think"])
        registry.register(config)
        with pytest.raises(
            ValueError, match="Tool 'calculator' already registered"
        ):
            registry.register(config)

    def test_get_tool(self, registry):
        """Test retrieving registered tool."""
        config = ToolConfig(calculator, ["think"])
        registry.register(config)
        tool = registry.get("calculator")
        assert tool is not None
        assert tool.name == "calculator"

    def test_get_nonexistent_tool(self, registry):
        """Test retrieving non-existent tool."""
        assert registry.get("nonexistent") is None

    def test_get_tools_by_scope(self, registry):
        """Test retrieving tools by scope."""
        registry.register(ToolConfig(calculator, ["think"]))
        registry.register(ToolConfig(printer, ["act"]))

        think_tools = registry.get_for_scope("think")
        assert len(think_tools) == 1
        assert think_tools[0].name == "calculator"

        act_tools = registry.get_for_scope("act")
        assert len(act_tools) == 1
        assert act_tools[0].name == "custom_printer"

    def test_tool_with_multiple_scopes(self, registry):
        """Test tool registration with multiple scopes."""
        config = ToolConfig(calculator, ["think", "act"])
        registry.register(config)

        think_tools = registry.get_for_scope("think")
        act_tools = registry.get_for_scope("act")

        assert len(think_tools) == 1
        assert len(act_tools) == 1
        assert think_tools[0].name == "calculator"
        assert act_tools[0].name == "calculator"

    def test_all_scope(self, registry):
        """Test tool registration with ALL scope."""
        config = ToolConfig(calculator, ["all"])
        registry.register(config)

        for scope in ToolScope:
            tools = registry.get_for_scope(scope.value)
            assert len(tools) == 1
            assert tools[0].name == "calculator"

    def test_tool_signature_formatting(self, registry):
        """Test tool signature string formatting."""
        registry.register(ToolConfig(calculator, ["think"]))
        tool = registry.get("calculator")
        assert tool is not None
        assert "calculator(x: int, y: int) -> int" == tool.signature_str

    def test_tool_with_optional_params_signature(self, registry):
        """Test signature formatting with optional parameters."""
        registry.register(ToolConfig(printer, ["think"]))
        tool = registry.get("custom_printer")
        assert tool is not None
        assert (
            "custom_printer(message: str, prefix: Optional) -> str"
            == tool.signature_str
        )


class TestToolExecution:
    @pytest.fixture
    def registry(self):
        reg = ToolRegistry()
        reg.register(ToolConfig(calculator, ["all"]))
        reg.register(ToolConfig(printer, ["all"]))
        return reg

    def test_basic_execution(self, registry):
        """Test basic tool execution through registry."""
        tool = registry.get("calculator")
        assert tool is not None
        result = tool(2, 3)
        assert result == 5

    def test_execution_with_kwargs(self, registry):
        """Test tool execution with keyword arguments."""
        tool = registry.get("custom_printer")
        assert tool is not None
        result = tool(message="hello", prefix="test")
        assert result == "test: hello"

    def test_execution_with_missing_args(self, registry):
        """Test tool execution with missing required arguments."""
        tool = registry.get("calculator")
        assert tool is not None
        with pytest.raises(TypeError):
            tool(2)

    def test_execution_with_invalid_args(self, registry):
        """Test tool execution with invalid argument types."""
        tool = registry.get("calculator")
        assert tool is not None
        with pytest.raises(TypeError):
            tool("not a number", 3)
