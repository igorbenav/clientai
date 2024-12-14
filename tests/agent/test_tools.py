from typing import Any, Dict, Optional

import pytest

from clientai.agent.config.tools import ToolConfig
from clientai.agent.tools import Tool, tool
from clientai.agent.tools.registry import ToolRegistry, is_tool
from clientai.agent.tools.types import ToolScope, ToolSignature


def test_tool_creation_basic():
    """Test basic tool creation from function."""

    def calculator(x: int, y: int) -> int:
        """Calculate sum."""
        return x + y

    tool = Tool.create(
        func=calculator, name="Calculator", description="Adds two numbers"
    )

    assert tool.name == "Calculator"
    assert tool.description == "Adds two numbers"
    assert callable(tool.func)
    assert tool(2, 3) == 5


def test_tool_creation_defaults():
    """Test tool creation with default values."""

    def calculator(x: int, y: int) -> int:
        """Calculate sum."""
        return x + y

    tool = Tool.create(calculator)

    assert tool.name == "calculator"
    assert tool.description == "Calculate sum."
    assert callable(tool.func)


def test_tool_creation_no_docstring():
    """Test tool creation with no docstring."""

    def calculator(x: int, y: int) -> int:
        return x + y

    tool = Tool.create(calculator)

    assert tool.name == "calculator"
    assert tool.description == "No description available"


def test_tool_decorator_basic():
    """Test basic tool decorator usage."""

    @tool
    def calculator(x: int, y: int) -> int:
        """Calculate sum."""
        return x + y

    assert isinstance(calculator, Tool)
    assert calculator.name == "calculator"
    assert calculator(2, 3) == 5


def test_tool_decorator_with_params():
    """Test tool decorator with custom parameters."""

    @tool(name="CustomCalc", description="Custom calculator")
    def calculator(x: int, y: int) -> int:
        return x + y

    assert isinstance(calculator, Tool)
    assert calculator.name == "CustomCalc"
    assert calculator.description == "Custom calculator"
    assert calculator(2, 3) == 5


def test_tool_signature():
    """Test tool signature analysis."""

    def complex_func(
        x: int, y: str = "default", z: Optional[float] = None
    ) -> Dict[str, Any]:
        """Complex function."""
        return {"x": x, "y": y, "z": z}

    tool = Tool.create(complex_func)

    assert isinstance(tool.signature, ToolSignature)
    assert tool.signature.name == "complex_func"
    assert len(tool.signature.parameters) == 3
    assert tool.signature_str.startswith("complex_func(")


def test_tool_registry_basic():
    """Test basic tool registry functionality."""
    registry = ToolRegistry()

    def calculator(x: int, y: int) -> int:
        """Calculate sum."""
        return x + y

    tool = Tool.create(calculator)
    config = ToolConfig(tool=tool)
    registry.register(config)

    assert len(registry) == 1
    assert "calculator" in registry
    assert registry.get("calculator") is not None
    assert len(registry.get_for_scope(None)) == 1


def test_tool_registry_duplicate():
    """Test registering duplicate tools."""
    registry = ToolRegistry()

    def calculator(x: int) -> int:
        return x * 2

    tool = Tool.create(calculator)
    config = ToolConfig(tool=tool)
    registry.register(config)

    with pytest.raises(
        ValueError, match="Tool 'calculator' already registered"
    ):
        registry.register(config)


def test_tool_scoping_single():
    """Test tool scope with single scope."""
    registry = ToolRegistry()

    def calculator(x: int) -> int:
        return x * 2

    tool = Tool.create(calculator)
    config = ToolConfig(tool=tool, scopes=frozenset([ToolScope.THINK]))
    registry.register(config)

    assert len(registry.get_for_scope("think")) == 1
    assert len(registry.get_for_scope("act")) == 0
    assert len(registry.get_for_scope("observe")) == 0


def test_tool_scoping_multiple():
    """Test tool scope with multiple scopes."""
    registry = ToolRegistry()

    def calculator(x: int) -> int:
        return x * 2

    tool = Tool.create(calculator)
    config = ToolConfig(
        tool=tool, scopes=frozenset([ToolScope.THINK, ToolScope.ACT])
    )
    registry.register(config)

    assert len(registry.get_for_scope("think")) == 1
    assert len(registry.get_for_scope("act")) == 1
    assert len(registry.get_for_scope("observe")) == 0


def test_tool_scoping_all():
    """Test tool scope with ALL scope."""
    registry = ToolRegistry()

    def calculator(x: int) -> int:
        return x * 2

    tool = Tool.create(calculator)
    config = ToolConfig(tool=tool, scopes=frozenset([ToolScope.ALL]))
    registry.register(config)

    assert len(registry.get_for_scope("think")) == 1
    assert len(registry.get_for_scope("act")) == 1
    assert len(registry.get_for_scope("observe")) == 1
    assert len(registry.get_for_scope("synthesize")) == 1


def test_tool_execution_success():
    """Test successful tool execution."""

    def calculator(x: int, y: int) -> int:
        return x + y

    tool = Tool.create(calculator)

    assert tool(2, 3) == 5
    assert tool(x=2, y=3) == 5
    assert tool(y=3, x=2) == 5


def test_tool_execution_errors():
    """Test tool execution errors."""

    def calculator(x: int, y: int) -> int:
        return x + y

    tool = Tool.create(calculator)

    with pytest.raises(TypeError):
        tool("not a number", 3)

    with pytest.raises(TypeError):
        tool(1, "not a number")

    with pytest.raises(TypeError):
        tool()  # Missing arguments


def test_tool_string_representation():
    """Test tool string representation."""

    def calculator(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    tool = Tool.create(calculator, name="Calculator")

    str_rep = str(tool)
    assert "Calculator" in str_rep
    assert "Signature" in str_rep
    assert "Add two numbers" in str_rep


def test_is_tool_function():
    """Test is_tool function with various inputs."""

    def calculator(x: int) -> int:
        return x * 2

    tool = Tool.create(calculator)

    assert is_tool(tool)
    assert not is_tool(calculator)
    assert not is_tool(42)
    assert not is_tool("not a tool")
    assert not is_tool(None)


def test_tool_type_hints_validation():
    """Test validation of type hints in tool creation."""

    # Missing return type hint
    def bad_func1(x: int):  # type: ignore
        return x * 2

    # Missing parameter type hint
    def bad_func2(x) -> int:  # type: ignore
        return x * 2

    # Correct type hints
    def good_func(x: int) -> int:
        return x * 2

    Tool.create(good_func)  # Should work

    # Note: Depending on implementation, these might raise ValueError or pass
    Tool.create(bad_func1)
    Tool.create(bad_func2)


def test_tool_format_info():
    """Test tool information formatting."""

    def complex_func(
        x: int, y: str = "default", z: Optional[float] = None
    ) -> Dict[str, Any]:
        """Complex function with multiple parameters."""
        return {"x": x, "y": y, "z": z}

    tool = Tool.create(
        complex_func,
        name="ComplexTool",
        description="A complex tool for testing",
    )

    formatted = tool.format_tool_info()
    assert "ComplexTool" in formatted
    assert "x: int" in formatted
    assert "Optional[float]" in formatted
    assert "A complex tool for testing" in formatted


def test_tool_registry_get_for_invalid_scope():
    """Test registry behavior with invalid scope."""
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Invalid scope"):
        registry.get_for_scope("invalid_scope")
