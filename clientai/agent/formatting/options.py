from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FormatOptions:
    """Configuration options for formatting agent string representations.

    Controls how agent information is formatted into strings,
    including indentation, line drawing characters, and text
    formatting constraints.

    Attributes:
        max_description_length: Maximum length for truncating descriptions
        indent: String to use for each level of indentation
        tree_chars: Optional mapping of tree drawing characters for formatting.
            Supported keys:
            - "vertical": Vertical line character (│)
            - "horizontal": Horizontal line character (─)
            - "branch": Branch character (├)
            - "corner": Corner character (└)
            - "top": Top junction character (╭)
            - "bottom": Bottom junction character (╰)

    Example:
        Basic formatting options:
        ```python
        # Default options
        options = FormatOptions()

        # Custom formatting
        options = FormatOptions(
            max_description_length=80,
            indent="    ",
            tree_chars={
                "vertical": "│",
                "horizontal": "─",
                "branch": "├",
                "corner": "└",
                "top": "╭",
                "bottom": "╰"
            }
        )
        ```

    Notes:
        - If tree_chars is not provided, defaults to standard
          box-drawing characters
        - Description truncation adds "..." when exceeding
          max_description_length
    """

    max_description_length: int = 60
    indent: str = "  "
    tree_chars: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Initialize default tree drawing characters if not provided.

        Example:
            Default initialization:
            ```python
            options = FormatOptions()
            print(options.tree_chars["vertical"])  # Output: │

            # Custom partial override
            options = FormatOptions(tree_chars={"vertical": "|"})
            print(options.tree_chars["vertical"])  # Output: |
            print(options.tree_chars["horizontal"])  # Output: ─
            ```
        """
        default_chars = {
            "vertical": "│",
            "horizontal": "─",
            "branch": "├",
            "corner": "└",
            "top": "╭",
            "bottom": "╰",
        }
        self.tree_chars = self.tree_chars or default_chars
