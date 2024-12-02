from inspect import cleandoc
from typing import Any, List, Optional, get_type_hints

from ..config.models import ModelConfig
from ..steps.base import Step
from ..tools.base import Tool
from .options import FormatOptions


class AgentFormatter:
    """Handles formatting of Agent string representations."""

    def __init__(self, options: Optional[FormatOptions] = None):
        self.options = options or FormatOptions()

    def _format_type(self, t: Any) -> str:
        """Format type hints cleanly."""
        if hasattr(t, "__name__"):
            return str(t.__name__)
        return str(t).replace("typing.", "")

    def _format_description(self, desc: str) -> str:
        """Format description with proper line handling."""
        if not desc:
            return "No description"
        desc = cleandoc(desc).split("\n")[0].strip()
        max_len = self.options.max_description_length
        return f"{desc[:max_len-3]}..." if len(desc) > max_len else desc

    def _format_model_config(self, config: ModelConfig) -> List[str]:
        """Format model configuration parameters."""
        params = []
        for k, v in config.to_dict().items():
            if k != "name" and v is not None:
                params.append(f"{k}={v}")
        return params

    def _format_step_signature(self, step: Step) -> tuple[str, str]:
        """Format step input/output signature."""
        hints = get_type_hints(step.func)
        params = list(hints.items())
        input_type = (
            self._format_type(params[1][1]) if len(params) > 1 else "None"
        )
        return_type = self._format_type(hints.get("return", Any))
        return input_type, return_type

    def _format_tools(self, tools: List[Tool]) -> List[str]:
        """Format tools with proper tree structure."""
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
        """Format agent header section."""
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
        """Format workflow section."""
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
        """Format custom run method section."""
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
        """Format context state section."""
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
        """Format complete agent string representation."""
        tree_chars = self.options.tree_chars or {}
        sections = [
            self._format_header(agent),
            self._format_workflow(agent),
            self._format_custom_run(agent),
            self._format_context(agent),
            [tree_chars.get("bottom", "╰") + "─"],
        ]

        return "\n".join(line for section in sections for line in section)
