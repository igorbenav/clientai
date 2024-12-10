from typing import Any, Dict, List, Optional, Set

from ..config.tools import ToolConfig
from .base import Tool
from .types import ToolScope


class ToolRegistry:
    """Registry for managing and organizing tools by name and scope.

    A centralized registry that maintains a collection of tools with
    efficient lookup by name and scope. It ensures unique tool names
    and proper scope indexing for quick access to tools available in
    different execution contexts.

    Attributes:
        _tools: Dictionary mapping tool names to Tool instances.
        _scope_index: Dictionary mapping scopes to sets of tool names.

    Example:
        ```python
        registry = ToolRegistry()

        # Register a tool with configuration
        config = ToolConfig(
            tool=calculator_func,
            scopes=["think", "act"],
            name="Calculator"
        )
        registry.register(config)

        # Get tools for a scope
        think_tools = registry.get_for_scope("think")

        # Check if tool exists
        if "Calculator" in registry:
            tool = registry.get("Calculator")
        ```
    """

    def __init__(self) -> None:
        """
        Initialize an empty tool registry.

        Creates empty storage for tools and initializes scope indexing for
        all available tool scopes.
        """
        self._tools: Dict[str, Tool] = {}
        self._scope_index: Dict[ToolScope, Set[str]] = {
            scope: set() for scope in ToolScope
        }

    def register(self, tool_config: ToolConfig) -> None:
        """Register a new tool with the registry.

        Creates a Tool instance if needed and adds it to the registry with
        proper scope indexing. Handles scope inheritance for tools marked
        as available in all scopes.

        Args:
            tool_config: Configuration specifying the tool and its properties.

        Raises:
            ValueError: If a tool with the same name is already registered.

        Example:
            ```python
            registry = ToolRegistry()
            registry.register(ToolConfig(
                tool=my_tool,
                scopes=["think"],
                name="MyTool"
            ))
            ```
        """
        tool = (
            tool_config.tool
            if isinstance(tool_config.tool, Tool)
            else Tool.create(
                func=tool_config.tool,
                name=tool_config.name,
                description=tool_config.description,
            )
        )

        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool

        for scope in tool_config.scopes:
            self._scope_index[scope].add(tool.name)
            if scope == ToolScope.ALL:
                for s in ToolScope:
                    self._scope_index[s].add(tool.name)

    def get(self, name: str) -> Optional[Tool]:
        """Retrieve a tool by its name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The requested Tool instance, or None if not found.

        Example:
            ```python
            tool = registry.get("Calculator")
            if tool:
                result = tool(5, 3)
            ```
        """
        return self._tools.get(name)

    def get_for_scope(self, scope: Optional[str] = None) -> List[Tool]:
        """Get all tools available in a specific scope.

        Args:
            scope: The scope to filter tools by. If None, returns all tools.

        Returns:
            List of Tool instances available in the specified scope.

        Raises:
            ValueError: If the specified scope is invalid.

        Example:
            ```python
            think_tools = registry.get_for_scope("think")
            all_tools = registry.get_for_scope(None)
            ```
        """
        if scope is None:
            return list(self._tools.values())

        tool_scope = ToolScope.from_str(scope)
        return [self._tools[name] for name in self._scope_index[tool_scope]]

    def __contains__(self, name: str) -> bool:
        """
        Check if a tool is registered by name.

        Args:
            name: The name of the tool to check.

        Returns:
            True if the tool is registered, False otherwise.

        Example:
            ```python
            if "Calculator" in registry:
                tool = registry.get("Calculator")
            ```
        """
        return name in self._tools

    def __len__(self) -> int:
        """
        Get the total number of registered tools.

        Returns:
            Number of tools in the registry.

        Example:
            ```python
            print(f"Registry contains {len(registry)} tools")
            ```
        """
        return len(self._tools)


def is_tool(obj: Any) -> bool:
    """Check if an object is a Tool instance.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a Tool instance, False otherwise.

    Example:
        ```python
        if is_tool(obj):
            result = obj(5, 3)  # We know obj is a Tool
        ```
    """
    return isinstance(obj, Tool)
