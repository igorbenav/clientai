from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FormatOptions:
    """Configuration options for agent string formatting."""

    max_description_length: int = 60
    indent: str = "  "
    tree_chars: Optional[Dict[str, str]] = None

    def __post_init__(self):
        default_chars = {
            "vertical": "│",
            "horizontal": "─",
            "branch": "├",
            "corner": "└",
            "top": "╭",
            "bottom": "╰",
        }
        self.tree_chars = self.tree_chars or default_chars
