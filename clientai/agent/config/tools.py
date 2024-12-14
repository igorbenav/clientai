from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Optional, Union

from ..tools.types import ToolScope
from ..types import ToolProtocol


@dataclass
class ToolConfig:
    """Configuration for registering tools with an agent.

    Defines how a tool should be registered and used
    within an agent's workflow, including its availability
    in different workflow scopes and metadata.

    Attributes:
        tool: The callable function or tool protocol instance to register
        scopes: Set of workflow scopes where the tool can be used
        name: Optional custom name for the tool
        description: Optional description of tool's functionality

    Example:
        Basic tool configuration:
        ```python
        def add(x: int, y: int) -> int:
            return x + y

        # Configure tool for all scopes
        config = ToolConfig(
            tool=add,
            name="Calculator",
            description="Adds two numbers"
        )

        # Configure tool for specific scopes
        config = ToolConfig(
            tool=add,
            scopes=["think", "act"],
            name="Calculator"
        )
        ```

    Notes:
        - If no scopes are specified, defaults to ["all"]
        - Scopes can be strings or ToolScope enum values
        - Name defaults to function name if not provided
    """

    tool: Union[Callable, ToolProtocol]
    scopes: FrozenSet[ToolScope] = field(
        default_factory=lambda: frozenset({ToolScope.ALL})
    )
    name: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and convert scope specifications.

        Converts string scope specifications to ToolScope enum values and
        ensures they are stored in a frozenset for immutability.

        Example:
            Scope conversion:
            ```python
            # Using string scopes
            config = ToolConfig(tool=func, scopes=["think", "act"])

            # Using enum scopes
            config = ToolConfig(
                tool=func,
                scopes=[ToolScope.THINK, ToolScope.ACT]
            )
            ```
        """
        if isinstance(self.scopes, (list, set)):  # noqa: UP038
            self.scopes = frozenset(
                ToolScope.from_str(s.strip()) if isinstance(s, str) else s
                for s in self.scopes
            )
