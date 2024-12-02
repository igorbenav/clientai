import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .prompts import DEFAULT_PROMPTS, MemoryPromptTemplate
from .types import MemoryType


@dataclass
class MemoryDecision:
    """
    Represents a decision made by an LLM about
    how to store information in memory.

    This class encapsulates the LLM's decisions about memory storage,
    including which memory types to use, importance ratings,
    and associated metadata.

    Attributes:
        store_in: Set of memory types where the information should be stored.
        key: The key under which to store the information.
        importance: A value between 0.0 and 1.0 indicating the
                    importance of the information.
        reason: The LLM's explanation for its storage decision.
        metadata: Additional metadata to store with the information.

    Examples:
        >>> decision = MemoryDecision(
        ...     store_in={MemoryType.WORKING},
        ...     key="task_context",
        ...     importance=0.8,
        ...     reason="Critical information for current task",
        ...     metadata={"category": "task", "priority": "high"}
        ... )
        >>> print(decision.importance)  # Output: 0.8
    """

    store_in: Set[MemoryType]
    key: str
    importance: float
    reason: str
    metadata: Dict[str, Any]


class LLMMemoryManager:
    """
    Manages memory storage and retrieval decisions using LLM-based reasoning.

    This class coordinates with an LLM to make intelligent decisions about
    how to store and retrieve information across different types of memory
    (working, episodic, semantic). It uses customizable prompts to guide the
    LLM's decision-making process.

    Attributes:
        client: The LLM client used for generating decisions.
        model: The name of the LLM model to use.
        memories: Dictionary mapping memory types to their
                  storage implementations.
        prompts: Templates for LLM prompts.

    Examples:
        >>> manager = LLMMemoryManager(
        ...     client=llm_client,
        ...     model="gpt-4",
        ...     memories={
        ...         MemoryType.WORKING: working_memory,
        ...         MemoryType.SEMANTIC: semantic_memory
        ...     }
        ... )
        >>> decisions = manager.decide_storage(
        ...     "Important task context",
        ...     {"current_task": "analysis"}
        ... )
    """

    def __init__(
        self,
        client: Any,
        model: str,
        memories: Dict[MemoryType, Any],
        prompts: Optional[Dict[str, MemoryPromptTemplate]] = None,
    ):
        self.client = client
        self.model = model
        self.memories = memories
        self.prompts = prompts or DEFAULT_PROMPTS.copy()

    def update_prompt(
        self,
        prompt_type: str,
        template: str,
        variables: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Update a prompt template used for LLM decisions.

        Args:
            prompt_type: The type of prompt to update ("storage", "retrieval").
            template: The new template string.
            variables: Variables used in the template.

        Raises:
            ValueError: If prompt_type is not recognized.

        Examples:
            >>> manager.update_prompt(
            ...     "storage",
            ...     "Consider storing: {content}\nContext: {context}",
            ...     {"content": "Content", "context": "Current context"}
            ... )
        """
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        self.prompts[prompt_type] = MemoryPromptTemplate(
            template=template,
            variables=variables or self.prompts[prompt_type].variables,
        )

    def decide_storage(
        self, content: Any, context: Dict[str, Any]
    ) -> List[MemoryDecision]:
        """
        Use the LLM to decide how to store new information in memory.

        The LLM analyzes the content and context to make decisions
        about storage locations, importance, and metadata. It considers
        different memory types and their appropriateness for the information.

        Args:
            content: The information to be stored.
            context: Current context to inform storage decisions.

        Returns:
            List[MemoryDecision]: List of storage decisions made by the LLM.

        Raises:
            ValueError: If the LLM response cannot be parsed or is invalid.

        Examples:
            >>> decisions = manager.decide_storage(
            ...     "Task completed successfully",
            ...     {"task": "data_analysis", "status": "complete"}
            ... )
            >>> for decision in decisions:
            ...     print(f"Store in {decision.store_in}: {decision.reason}")
        """

        prompt = self.prompts["storage"].format(
            content=content, context=context
        )

        response = self.client.generate_text(prompt, model=self.model)

        try:
            decisions = json.loads(response)["decisions"]
            return [
                MemoryDecision(
                    store_in={MemoryType(d["memory_type"])}
                    if d["store"]
                    else set(),
                    key=d["key"],
                    importance=d["importance"],
                    reason=d["reason"],
                    metadata=d["metadata"],
                )
                for d in decisions
            ]
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid LLM response format: {e}")

    def decide_retrieval(
        self, query: str, task: str, context: Dict[str, Any]
    ) -> Dict[MemoryType, List[str]]:
        """
        Use the LLM to decide what information to retrieve from memory.

        The LLM analyzes the query, current task, and context to determine
        which memories would be most relevant and how to retrieve them.

        Args:
            query: The retrieval query or request.
            task: The current task or operation being performed.
            context: Additional context for retrieval decisions.

        Returns:
            Dict[MemoryType, List[str]]: Retrieved memories organized
                                         by memory type.

        Raises:
            ValueError: If the LLM response cannot be parsed or is invalid.

        Examples:
            >>> memories = manager.decide_retrieval(
            ...     "Find relevant task history",
            ...     "data_analysis",
            ...     {"status": "in_progress"}
            ... )
            >>> for memory_type, items in memories.items():
            ...     print(f"{memory_type}: {len(items)} items found")
        """
        prompt = self.prompts["retrieval"].format(
            query=query, task=task, context=context
        )

        response = self.client.generate_text(prompt, model=self.model)

        try:
            decisions = json.loads(response)["retrieval"]
            return self._execute_retrieval(decisions)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid LLM response format: {e}")

    def _execute_retrieval(
        self, decisions: List[Dict[str, Any]]
    ) -> Dict[MemoryType, List[str]]:
        """
        Execute memory retrieval based on LLM decisions.

        Takes the LLM's retrieval decisions and executes them against the
        appropriate memory stores using the specified retrieval strategies.

        Args:
            decisions: List of retrieval decisions from the LLM.

        Returns:
            Dict[MemoryType, List[str]]: Retrieved memories organized
                                         by memory type.

        Examples:
            >>> decisions = [
            ...     {
            ...         "memory_type": "working",
            ...         "search": True,
            ...         "strategy": "recent",
            ...         "criteria": {"limit": 5}
            ...     }
            ... ]
            >>> results = manager._execute_retrieval(decisions)
        """
        results = {}

        for decision in decisions:
            if not decision["search"]:
                continue

            memory_type = MemoryType(decision["memory_type"])
            memory = self.memories[memory_type]
            strategy = decision["strategy"]
            criteria = decision["criteria"]

            if strategy == "similar":
                results[memory_type] = self._retrieve_similar(memory, criteria)
            elif strategy == "recent":
                results[memory_type] = self._retrieve_recent(memory, criteria)
            else:
                results[memory_type] = self._retrieve_exact(memory, criteria)

        return results

    def _convert_to_str_list(self, result: Any) -> List[str]:
        """
        Convert retrieval results to a list of strings.

        Handles various result types (None, single string, list, tuple) and
        ensures consistent string list output.

        Args:
            result: The retrieval result to convert.

        Returns:
            List[str]: The result converted to a list of strings.

        Examples:
            >>> manager._convert_to_str_list("single result")
            ['single result']
            >>> manager._convert_to_str_list(['item1', 'item2'])
            ['item1', 'item2']
            >>> manager._convert_to_str_list(None)
            []
        """
        if result is None:
            return []
        if isinstance(result, str):
            return [result]
        if isinstance(result, (list, tuple)):  # noqa: UP038
            return [str(item) for item in result]
        return [str(result)]

    def _retrieve_similar(
        self, memory: Any, criteria: Dict[str, Any]
    ) -> List[str]:
        """Retrieve similar memories based on criteria."""
        if hasattr(memory, "find_similar"):
            result = memory.find_similar(**criteria)
            return self._convert_to_str_list(result)
        return []

    def _retrieve_recent(
        self, memory: Any, criteria: Dict[str, Any]
    ) -> List[str]:
        """Retrieve recent memories based on criteria."""
        if hasattr(memory, "get_recent"):
            result = memory.get_recent(**criteria)
            return self._convert_to_str_list(result)
        return []

    def _retrieve_exact(
        self, memory: Any, criteria: Dict[str, Any]
    ) -> List[str]:
        """Retrieve exact matches based on criteria."""
        if hasattr(memory, "get_exact"):
            result = memory.get_exact(**criteria)
            return self._convert_to_str_list(result)
        return []
