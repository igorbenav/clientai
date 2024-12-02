from enum import Enum
from typing import TypeVar

MemoryKey = TypeVar("MemoryKey")
MemoryValue = TypeVar("MemoryValue")


class MemoryType(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
