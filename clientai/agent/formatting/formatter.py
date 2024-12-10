from inspect import cleandoc
from typing import Any, List, Optional, get_type_hints

from ..config.models import ModelConfig
from ..steps.base import Step
from ..tools.base import Tool
from .options import FormatOptions


class AgentFormatter:
    """Formats agent information into structured string representations.

    Creates human-readable string representations of
    agent configurations, workflows, tools, and state
    using consistent formatting and tree structures.

    Args:
        options: Optional FormatOptions to control formatting behavior

    Example:
        Basic formatting:
        ```python
        formatter = AgentFormatter()
        formatted_str = formatter.format_agent(my_agent)
        print(formatted_str)
        # Output:
        # ╭─ MyAgent (openai provider)
        # │
        # │ Configuration:
        # │ ├─ Model: gpt-4
        # │ └─ Parameters: temperature=0.7
        # ...
        ```

        Custom formatting options:
        ```python
        options = FormatOptions(
            max_description_length=80,
            tree_chars={"vertical": "|", "horizontal": "-"}
        )
        formatter = AgentFormatter(options)
        print(formatter.format_agent(my_agent))
        ```
    """

    def __init__(self, options: Optional[FormatOptions] = None):
        self.options = options or FormatOptions()

    def _format_type(self, t: Any) -> str:
        """Format type hints into clean string representations.

        Args:
            t: Type annotation to format

        Returns:
            str: Formatted type string
        """
        if hasattr(t, "__name__"):
            return str(t.__name__)
        return str(t).replace("typing.", "")

    def _format_description(self, desc: str) -> str:
        """Format and truncate description text according to options.

        Args:
            desc: Description text to format

        Returns:
            str: Formatted description text
        """
        if not desc:
            return "No description"
        desc = cleandoc(desc).split("\n")[0].strip()
        max_len = self.options.max_description_length
        return f"{desc[:max_len-3]}..." if len(desc) > max_len else desc

    def _format_model_config(self, config: ModelConfig) -> List[str]:
        """Format model configuration parameters into string list.

        Args:
            config: Model configuration to format

        Returns:
            List[str]: Formatted parameter strings
        """
        params = []
        for k, v in config.to_dict().items():
            if k != "name" and v is not None:
                params.append(f"{k}={v}")
        return params

    def _format_step_signature(self, step: Step) -> tuple[str, str]:
        """Format step input/output signature.

        Args:
            step: Step to format signature for

        Returns:
            tuple[str, str]: Input type and return type strings
        """
        hints = get_type_hints(step.func)
        params = list(hints.items())
        input_type = (
            self._format_type(params[1][1]) if len(params) > 1 else "None"
        )
        return_type = self._format_type(hints.get("return", Any))
        return input_type, return_type

    def _format_tools(self, tools: List[Tool]) -> List[str]:
        """Format tool information into tree structure.

        Args:
            tools: List of tools to format

        Returns:
            List[str]: Formatted tool information strings
        """
        if not tools:
            return [f"{self.options.indent}└─ No tools available"]

        tree_chars = self.options.tree_chars or {}
        lines = []
        for i, tool in enumerate(tools, 1):
            is_last = i == len(tools)
            prefix = (
                tree_chars.get("corner", "└")
                if is_last
                else tree_chars.get("branch", "├")
            )
            continuation = "  " if is_last else tree_chars.get("vertical", "│")
            horiz = tree_chars.get("horizontal", "─")

            base_indent = f"{self.options.indent}{continuation}"
            tool_line = f"{self.options.indent}{prefix}{horiz} {tool.name}"
            sig_line = f"{base_indent} ├{horiz} {tool.signature_str}"
            desc = self._format_description(tool.description)
            desc_line = f"{base_indent} └{horiz} {desc}"

            lines.extend([tool_line, sig_line, desc_line])

            if not is_last:
                lines.append(f"{self.options.indent}{continuation}")

        return lines

    def _format_header(self, agent: Any) -> List[str]:
        """Format agent header section.

        Args:
            agent: Agent instance to format header for

        Returns:
            List[str]: Formatted header strings
        """
        provider_name = agent._client.provider.__class__.__module__.split(".")[
            -2
        ]
        tree_chars = self.options.tree_chars or {}
        v = tree_chars.get("vertical", "│")
        top = tree_chars.get("top", "╭")
        agent_name = agent.__class__.__name__
        model_name = agent._default_model.name

        header = f"{top}─ {agent_name} ({provider_name} provider)"

        lines = [
            header,
            v,
            f"{v} Configuration:",
            f"{v} ├─ Model: {model_name}",
        ]

        model_params = self._format_model_config(agent._default_model)
        if model_params:
            param_str = ", ".join(model_params)
            lines.append(f"{v} └─ Parameters: {param_str}")
        else:
            lines.append(f"{v} └─ No additional parameters")

        return lines

    def _format_workflow(self, agent: Any) -> List[str]:
        """Format agent workflow section.

        Args:
            agent: Agent instance to format workflow for

        Returns:
            List[str]: Formatted workflow strings
        """
        steps = agent.workflow_manager.get_steps()
        if not steps:
            return []

        tree_chars = self.options.tree_chars or {}
        v = tree_chars.get("vertical", "│")
        lines = [v, f"{v} Workflow:"]

        for i, (name, step) in enumerate(steps.items(), 1):
            is_last_step = i == len(steps)
            prefix = (
                tree_chars.get("corner", "└")
                if is_last_step
                else tree_chars.get("branch", "├")
            )

            input_type, return_type = self._format_step_signature(step)
            step_type = step.step_type.name.lower()
            step_desc = self._format_description(step.description)

            model_info = (
                ", ".join(self._format_model_config(step.llm_config))
                if step.llm_config
                else f"{agent._default_model.name} (default)"
            )

            base = f"{v} {self.options.indent}"
            step_lines = [
                f"{v} {prefix}─ {i}. {name}",
                f"{base}├─ Type: {step_type}",
                f"{base}├─ I/O: {input_type} → {return_type}",
                f"{base}├─ Model: {model_info}",
                f"{base}├─ Send to LLM: {step.send_to_llm}",
                f"{base}├─ Description: {step_desc}",
                f"{base}└─ Available Tools:",
            ]

            lines.extend(step_lines)

            tool_lines = self._format_tools(
                agent.get_tools(step.step_type.name.lower())
            )
            tool_indented = (
                f"{v} {self.options.indent}{line}" for line in tool_lines
            )
            lines.extend(tool_indented)

            if not is_last_step:
                lines.append(v)

        return lines

    def _format_custom_run(self, agent: Any) -> List[str]:
        """Format custom run method section.

        Args:
            agent: Agent instance to check for custom run method

        Returns:
            List[str]: Formatted custom run information strings
        """
        if not hasattr(agent, "_custom_run") or not agent._custom_run:
            return []

        tree_chars = self.options.tree_chars or {}
        v = tree_chars.get("vertical", "│")
        hints = get_type_hints(agent._custom_run)
        params = list(hints.items())
        input_type = (
            self._format_type(params[1][1]) if len(params) > 1 else "None"
        )
        return_type = self._format_type(hints.get("return", Any))

        return [
            v,
            f"{v} Custom Run Method:",
            f"{v} └─ Signature: {input_type} → {return_type}",
        ]

    def _format_context(self, agent: Any) -> List[str]:
        """Format agent context state section.

        Args:
            agent: Agent instance to format context for

        Returns:
            List[str]: Formatted context information strings
        """
        tree_chars = self.options.tree_chars or {}
        v = tree_chars.get("vertical", "│")
        keys = agent.context.state.keys()
        state_keys = ", ".join(keys) or "empty"

        return [
            v,
            f"{v} Context State:",
            f"{v} ├─ Results: {len(agent.context.last_results)} stored",
            f"{v} ├─ Memory: {len(agent.context.memory)} entries",
            f"{v} └─ State Keys: {state_keys}",
        ]

    def format_agent(self, agent: Any) -> str:
        """Format complete agent information into a string representation.

        Creates a comprehensive view of the agent including its configuration,
        workflow steps, tools, and current state.

        Args:
            agent: The agent instance to format

        Returns:
            str: Formatted string representation of the agent

        Example:
            Format agent information:
            ```python
            formatter = AgentFormatter()
            print(formatter.format_agent(my_agent))
            # Output:
            # ╭─ MyAgent (openai provider)
            # │
            # │ Configuration:
            # │ ├─ Model: gpt-4
            # │ └─ Parameters: temperature=0.7, top_p=0.9
            # │
            # │ Workflow:
            # │ ├─ 1. analyze
            # │ │  ├─ Type: think
            # │ │  ├─ I/O: str → str
            # │ │  └─ Description: Analyzes input data
            # ...
            ```
        """
        tree_chars = self.options.tree_chars or {}
        sections = [
            self._format_header(agent),
            self._format_workflow(agent),
            self._format_custom_run(agent),
            self._format_context(agent),
            [tree_chars.get("bottom", "╰") + "─"],
        ]

        return "\n".join(line for section in sections for line in section)
