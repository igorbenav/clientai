import json
import time
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from clientai.agent.config.models import ModelConfig
from clientai.agent.config.steps import StepConfig
from clientai.agent.core.execution import StepExecutionEngine
from clientai.agent.steps import Step, StepType
from clientai.agent.tools import Tool, ToolSelectionConfig
from clientai.exceptions import ClientAIError


class TestStepExecutionEngine:
    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.generate_text = Mock(return_value="Generated response")
        return client

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.context = MagicMock()
        agent.context.last_results = {}
        agent.get_tools = Mock(return_value=[])
        return agent

    @pytest.fixture
    def engine(self, mock_client):
        return StepExecutionEngine(
            client=mock_client,
            default_model=ModelConfig(name="test-model"),
            default_kwargs={},
        )

    def test_basic_step_execution(self, engine, mock_client, mock_agent):
        """Test basic step execution without LLM."""

        def test_func(agent: Any, input_data: str) -> str:
            return f"Processed: {input_data}"

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="test_step",
            send_to_llm=False,
        )

        result = engine.execute_step(step, mock_agent, "test_input")
        assert result == "Processed: test_input"
        assert not mock_client.generate_text.called
        assert mock_agent.context.last_results["test_step"] == result

    def test_llm_step_execution(self, engine, mock_client, mock_agent):
        """Test step execution with LLM integration."""

        def test_func(agent: Any, input_data: str) -> str:
            return f"Analyze: {input_data}"

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="llm_step",
            send_to_llm=True,
        )

        result = engine.execute_step(step, mock_agent, "test_input")
        assert result == "Generated response"
        mock_client.generate_text.assert_called_once()
        assert mock_agent.context.last_results["llm_step"] == result

    def test_retry_mechanism(self, engine, mock_client, mock_agent):
        """Test retry mechanism for failed steps."""
        mock_client.generate_text.side_effect = [
            ClientAIError("First attempt failed"),
            ClientAIError("Second attempt failed"),
            "Success on third try",
        ]

        def test_func(agent: Any, input_data: str) -> str:
            return f"Retry test: {input_data}"

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="retry_step",
            send_to_llm=True,
            step_config=StepConfig(retry_count=2),
        )

        result = engine.execute_step(step, mock_agent, "test_input")
        assert result == "Success on third try"
        assert mock_client.generate_text.call_count == 3
        assert mock_agent.context.last_results["retry_step"] == result

    def test_timeout_handling(self, engine, mock_client, mock_agent):
        """Test timeout handling during step execution."""
        import signal
        from contextlib import contextmanager

        @contextmanager
        def timeout_handler(seconds):
            def signal_handler(signum, frame):
                raise TimeoutError("Step execution timed out")

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)

        def slow_func(agent: Any, input_data: str) -> str:
            time.sleep(2)
            return input_data

        step = Step.create(
            func=slow_func,
            step_type=StepType.THINK,
            name="timeout_step",
            send_to_llm=False,
            step_config=StepConfig(timeout=1.0),
        )

        with pytest.raises(TimeoutError):
            with timeout_handler(1):
                engine.execute_step(step, mock_agent, "test_input")

    def test_tool_integration(self, engine, mock_client, mock_agent):
        """Test tool integration with automatic tool selection."""
        call_spy = Mock()

        def tool_callable(x: str) -> str:
            call_spy(x)  # Record the call
            return "Tool execution result"

        mock_tool = Tool.create(
            func=tool_callable,
            name="test_tool",
            description="Test tool description",
        )

        mock_agent.get_tools.return_value = [mock_tool]

        mock_client.generate_text.side_effect = [
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool_name": "test_tool",
                            "arguments": {"x": "test_input"},
                            "confidence": 0.95,
                            "reasoning": "Test reasoning",
                        }
                    ]
                }
            ),
            "Generated response",
        ]

        def tool_func(agent: Any, input_data: str) -> str:
            return f"Use tool for: {input_data}"

        step = Step.create(
            func=tool_func,
            step_type=StepType.ACT,
            name="tool_step",
            send_to_llm=True,
            use_tools=True,
            tool_selection_config=ToolSelectionConfig(
                confidence_threshold=0.8, max_tools_per_step=1
            ),
        )

        result = engine.execute_step(step, mock_agent, "test_input")

        assert mock_client.generate_text.call_count >= 1
        first_call = mock_client.generate_text.call_args_list[0]
        tool_selection_prompt = first_call[0][0]
        assert "test_tool" in tool_selection_prompt

        call_spy.assert_called_once_with("test_input")

        second_call = mock_client.generate_text.call_args_list[1]
        final_prompt = second_call[0][0]
        assert "Tool Execution Results" in final_prompt
        assert "Tool execution result" in final_prompt

        assert result == "Generated response"
        assert mock_agent.context.last_results["tool_step"] == result

    def test_error_recovery(self, engine, mock_client, mock_agent):
        """Test error recovery and fallback behavior."""
        mock_client.generate_text.side_effect = ClientAIError("API Error")

        def test_func(agent: Any, input_data: str) -> str:
            return input_data

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="optional_step",
            send_to_llm=True,
            step_config=StepConfig(required=False),
        )

        result = engine.execute_step(step, mock_agent, "test_input")
        assert result is None

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="required_step",
            send_to_llm=True,
            step_config=StepConfig(required=True),
        )

        with pytest.raises(ClientAIError):
            engine.execute_step(step, mock_agent, "test_input")

    def test_model_configuration(self, engine, mock_client, mock_agent):
        """Test model configuration handling."""

        def test_func(agent: Any, input_data: str) -> str:
            return input_data

        custom_config = ModelConfig(
            name="custom-model", temperature=0.7, max_tokens=100
        )

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="config_step",
            send_to_llm=True,
            llm_config=custom_config,
        )

        engine.execute_step(step, mock_agent, "test_input")

        call_kwargs = mock_client.generate_text.call_args[1]
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100

    def test_prompt_building(self, engine, mock_client, mock_agent):
        """Test prompt construction for LLM steps."""

        def test_func(agent: Any, input_data: str) -> str:
            return f"Complex prompt with {input_data}"

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="prompt_step",
            send_to_llm=True,
        )

        engine.execute_step(step, mock_agent, "test_input")
        prompt = mock_client.generate_text.call_args[0][0]
        assert "Complex prompt with test_input" in prompt
        assert (
            mock_agent.context.last_results["prompt_step"]
            == "Generated response"
        )

    def test_concurrent_execution(self, engine, mock_client):
        """Test handling of concurrent step executions."""
        from concurrent.futures import ThreadPoolExecutor

        def test_func(agent: Any, input_data: str) -> str:
            return input_data

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="concurrent_step",
            send_to_llm=True,
        )

        mock_agent = MagicMock()
        mock_agent.context = MagicMock()
        mock_agent.context.last_results = {}
        mock_agent.get_tools = Mock(return_value=[])

        futures = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            for i in range(3):
                futures.append(
                    executor.submit(
                        engine.execute_step, step, mock_agent, f"input{i}"
                    )
                )

        for future in futures:
            future.result()

        assert mock_client.generate_text.call_count == 3
