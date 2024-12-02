from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MemoryPromptTemplate:
    """Template for memory-related prompts with customizable components."""

    template: str
    variables: Dict[str, str]

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables."""
        return self.template.format(**kwargs)


DEFAULT_PROMPTS = {
    "storage": MemoryPromptTemplate(
        template=(
            """
            Given this new information and context,
            decide how it should be stored in memory.

            Content to store: {content}

            Current Context: {context}

            Available Memory Types:
            - WORKING: Short-term, temporary storage for current task
            - EPISODIC: Long-term storage of experiences and events
            - SEMANTIC: Long-term storage of facts and concepts

            For each memory type, decide if and how this
            information should be stored. Consider:
            1. Is this information relevant for this type of memory?
            2. What key should be used to store it?
            3. How important is this information (0.0 to 1.0)?
            4. What additional metadata should be stored?

            Respond in JSON format:
            {{
                "decisions": [
                    {{
                        "memory_type": "type",
                        "store": true/false,
                        "key": "suggested_key",
                        "importance": 0.0-1.0,
                        "reason": "explanation",
                        "metadata": {{}}
                    }}
                ]
            }}
            """
        ),
        variables={
            "content": "The content to be stored",
            "context": "Current agent context",
        },
    ),
    "retrieval": MemoryPromptTemplate(
        template=(
            """
            Given the current task and context, decide what
            information should be retrieved from memory.

            Query: {query}
            Current Task: {task}
            Context: {context}

            Available Memory Types:
            - WORKING: Short-term, temporary storage for current task
            - EPISODIC: Long-term storage of experiences and events
            - SEMANTIC: Long-term storage of facts and concepts

            For each memory type, decide:
            1. Should this type of memory be searched?
            2. What kind of information would be most relevant?
            3. How should the search be performed (exact match, similarity)?

            Respond in JSON format:
            {{
                "retrieval": [
                    {{
                        "memory_type": "type",
                        "search": true/false,
                        "strategy": "exact/similar/recent",
                        "reason": "explanation",
                        "criteria": {{}}
                    }}
                ]
            }}
            """
        ),
        variables={
            "query": "The retrieval query",
            "task": "Current task name",
            "context": "Current agent context",
        },
    ),
}
