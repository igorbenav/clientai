import json
import time
from typing import Any, Iterator, Optional
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
        def mock_generate_text(*args, **kwargs):
            if kwargs.get("stream", False):
                return (chunk for chunk in ["Gen", "erated ", "response"])
            return "Generated response"

        client = Mock()
        client.generate_text = Mock(side_effect=mock_generate_text)
        return client

    @pytest.fixture
    def mock_agent(self):
        context_mock = MagicMock()
        context_mock.last_results = {}
        context_mock.state = {}
        context_mock.current_input = None

        agent = MagicMock()
        agent.context = context_mock
        agent.get_tools = Mock(return_value=[])
        return agent

    @pytest.fixture
    def engine(self, mock_client):
        return StepExecutionEngine(
            client=mock_client,
            default_model=ModelConfig(name="test-model", stream=False),
            default_kwargs={},
            tool_selection_config=ToolSelectionConfig(
                confidence_threshold=0.8, max_tools_per_step=1
            ),
        )

    def test_basic_step_execution(self, engine, mock_client, mock_agent):
        """Test basic step execution without LLM."""

        def test_func(
            agent: Any, input_data: str, stream: Optional[bool] = None
        ) -> str:
            return f"Processed: {input_data}"

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="test_step",
            send_to_llm=False,
            step_config=StepConfig(enabled=True),
        )

        result = engine.execute_step(step, mock_agent, "test_input")
        assert result == "Processed: test_input"
        assert not mock_client.generate_text.called
        assert mock_agent.context.last_results["test_step"] == result

    def test_llm_step_execution(self, engine, mock_client, mock_agent):
        """Test step execution with LLM integration."""

        def test_func(
            agent: Any, input_data: str, stream: Optional[bool] = None
        ) -> str:
            return f"Analyze: {input_data}"

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="llm_step",
            send_to_llm=True,
            step_config=StepConfig(enabled=True),
        )

        result = engine.execute_step(step, mock_agent, "test_input")
        assert result == "Generated response"
        mock_client.generate_text.assert_called_once()
        assert mock_agent.context.last_results["llm_step"] == result

    def test_retry_mechanism(self, engine, mock_client, mock_agent):
        """Test retry mechanism for failed steps."""
        responses = [
            ClientAIError("First attempt failed"),
            ClientAIError("Second attempt failed"),
            "Success on third try",
        ]

        def mock_generate(*args, **kwargs):
            response = responses[mock_client.generate_text.call_count - 1]
            if isinstance(response, Exception):
                raise response
            return response

        mock_client.generate_text = Mock(side_effect=mock_generate)

        def test_func(
            agent: Any, input_data: str, stream: Optional[bool] = None
        ) -> str:
            return f"Retry test: {input_data}"

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="retry_step",
            send_to_llm=True,
            step_config=StepConfig(retry_count=2, enabled=True),
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

        def slow_func(
            agent: Any, input_data: str, stream: Optional[bool] = None
        ) -> str:
            time.sleep(2)
            return input_data

        step = Step.create(
            func=slow_func,
            step_type=StepType.THINK,
            name="timeout_step",
            send_to_llm=False,
            step_config=StepConfig(timeout=1.0, enabled=True),
        )

        with pytest.raises(TimeoutError):
            with timeout_handler(1):
                engine.execute_step(step, mock_agent, "test_input")

    def test_tool_integration(self, engine, mock_client, mock_agent):
        """Test tool integration with automatic tool selection."""
        call_spy = Mock()

        def tool_callable(x: str) -> str:
            call_spy(x)
            return "Tool execution result"

        mock_tool = Tool.create(
            func=tool_callable,
            name="test_tool",
            description="Test tool description",
        )

        mock_agent.get_tools.return_value = [mock_tool]

        def mock_generate_text(*args, **kwargs):
            if mock_client.generate_text.call_count == 1:
                return json.dumps(
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
                )
            return "Generated response"

        mock_client.generate_text = Mock(side_effect=mock_generate_text)

        def tool_func(
            agent: Any, input_data: str, stream: Optional[bool] = None
        ) -> str:
            return f"Use tool for: {input_data}"

        step = Step.create(
            func=tool_func,
            step_type=StepType.ACT,
            name="tool_step",
            send_to_llm=True,
            use_tools=True,
            step_config=StepConfig(enabled=True),
            tool_selection_config=ToolSelectionConfig(
                confidence_threshold=0.8,
                max_tools_per_step=1,
            ),
        )

        result = engine.execute_step(step, mock_agent, "test_input")

        assert mock_client.generate_text.call_count >= 2
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

    def test_streaming_configuration(self, engine, mock_client, mock_agent):
        """Test streaming configuration handling."""

        def test_func(agent: Any, input_data: str) -> str:
            return input_data

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="stream_step",
            send_to_llm=True,
            llm_config=ModelConfig(name="test-model", stream=True),
            step_config=StepConfig(enabled=True),
        )

        engine.execute_step(step, mock_agent, "test_input")
        assert mock_client.generate_text.call_args[1].get("stream") is True

        engine.execute_step(step, mock_agent, "test_input", stream=False)
        assert mock_client.generate_text.call_args[1].get("stream") is False

    def test_streaming_response_handling(
        self, engine, mock_client, mock_agent
    ):
        """Test handling of streaming responses."""

        def test_func(agent: Any, input_data: str) -> str:
            return input_data

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="stream_step",
            send_to_llm=True,
            llm_config=ModelConfig(name="test-model", stream=True),
            step_config=StepConfig(enabled=True),
        )

        result = engine.execute_step(step, mock_agent, "test_input")
        assert isinstance(result, Iterator)
        assert "".join(result) == "Generated response"
        assert "stream_step" not in mock_agent.context.last_results

    def test_model_configuration(self, engine, mock_client, mock_agent):
        """Test model configuration handling."""

        def test_func(agent: Any, input_data: str) -> str:
            return input_data

        custom_config = ModelConfig(
            name="custom-model",
            temperature=0.7,
            max_tokens=100,
        )

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="config_step",
            send_to_llm=True,
            llm_config=custom_config,
            step_config=StepConfig(enabled=True),
        )

        engine.execute_step(step, mock_agent, "test_input")
        call_kwargs = mock_client.generate_text.call_args[1]
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100

    def test_error_handling(self, engine, mock_client, mock_agent):
        """Test error handling and required step behavior."""

        def error_func(agent: Any, input_data: str) -> str:
            raise ValueError("Test error")

        step = Step.create(
            func=error_func,
            step_type=StepType.THINK,
            name="optional_step",
            send_to_llm=False,
            step_config=StepConfig(required=False, enabled=True),
        )

        with pytest.raises(ValueError, match="Test error"):
            engine.execute_step(step, mock_agent, "test_input")

        step = Step.create(
            func=error_func,
            step_type=StepType.THINK,
            name="required_step",
            send_to_llm=False,
            step_config=StepConfig(required=True, enabled=True),
        )

        with pytest.raises(ValueError, match="Test error"):
            engine.execute_step(step, mock_agent, "test_input")

    def test_pass_result_configuration(self, engine, mock_client, mock_agent):
        """Test pass_result configuration behavior."""

        def test_func(agent: Any, input_data: str) -> str:
            return f"Modified: {input_data}"

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="pass_step",
            send_to_llm=False,
            step_config=StepConfig(enabled=True, pass_result=True),
        )

        result = engine.execute_step(step, mock_agent, "test_input")
        assert result == "Modified: test_input"
        assert mock_agent.context.current_input == result

        step = Step.create(
            func=test_func,
            step_type=StepType.THINK,
            name="no_pass_step",
            send_to_llm=False,
            step_config=StepConfig(enabled=True, pass_result=False),
        )

        original_input = mock_agent.context.current_input
        result = engine.execute_step(step, mock_agent, "test_input")
        assert result == "Modified: test_input"
        assert mock_agent.context.current_input == original_input
