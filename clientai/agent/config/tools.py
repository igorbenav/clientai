from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Optional, Union

from ..tools.types import ToolScope
from ..types import ToolProtocol


@dataclass
class ToolConfig:
    tool: Union[Callable, ToolProtocol]
    scopes: FrozenSet[ToolScope] = field(
        default_factory=lambda: frozenset({ToolScope.ALL})
    )
    name: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.scopes, (list, set)):  # noqa: UP038
            self.scopes = frozenset(
                ToolScope.from_str(s.strip()) if isinstance(s, str) else s
                for s in self.scopes
            )
