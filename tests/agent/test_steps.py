from typing import Optional

import pytest

from clientai.agent.config.models import ModelConfig
from clientai.agent.config.steps import StepConfig
from clientai.agent.steps import Step, StepType
from clientai.agent.steps.base import FunctionMetadata
from clientai.agent.steps.decorators import act, observe, synthesize, think


class MockAgent:
    """Mock agent class for testing steps."""

    def __init__(self):
        self.context = None


def test_step_creation():
    """Test basic step creation."""

    def sample_func(data: str) -> str:
        """Sample function."""
        return f"processed: {data}"

    step = Step(
        func=sample_func,
        step_type=StepType.THINK,
        name="sample_step",
        description="Sample function.",
        send_to_llm=True,
    )

    assert step.name == "sample_step"
    assert step.step_type == StepType.THINK
    assert step.description == "Sample function."
    assert callable(step.func)


def test_step_with_invalid_return_type():
    """Test step creation with invalid return type."""

    def invalid_func(data: str) -> int:
        return 42

    with pytest.raises(ValueError, match="must return str"):
        Step(
            func=invalid_func,
            step_type=StepType.THINK,
            name="invalid_step",
            send_to_llm=True,
        )


def test_step_with_model_config():
    """Test step creation with model configuration."""

    def sample_func(data: str) -> str:
        return data

    model_config = ModelConfig(name="test-model", temperature=0.7)

    step = Step(
        func=sample_func,
        step_type=StepType.THINK,
        name="config_step",
        llm_config=model_config,
        send_to_llm=True,
    )

    assert step.llm_config is not None
    assert step.llm_config.name == "test-model"
    assert step.llm_config.get_parameters()["temperature"] == 0.7


class TestStepDecorators:
    """Test suite for step decorators."""

    def test_think_decorator(self):
        """Test @think decorator."""

        @think(name="test_think", model="test-model")
        def think_step(self, data: str) -> str:
            return f"thought about: {data}"

        step_info = think_step._step_info
        assert step_info.name == "test_think"
        assert step_info.step_type == StepType.THINK
        assert step_info.llm_config.name == "test-model"

    def test_act_decorator(self):
        """Test @act decorator."""

        @act(name="test_act", model="test-model")
        def act_step(self, data: str) -> str:
            return f"acted on: {data}"

        step_info = act_step._step_info
        assert step_info.name == "test_act"
        assert step_info.step_type == StepType.ACT
        assert step_info.llm_config.name == "test-model"

    def test_observe_decorator(self):
        """Test @observe decorator."""

        @observe(name="test_observe", model="test-model")
        def observe_step(self, data: str) -> str:
            return f"observed: {data}"

        step_info = observe_step._step_info
        assert step_info.name == "test_observe"
        assert step_info.step_type == StepType.OBSERVE
        assert step_info.llm_config.name == "test-model"

    def test_synthesize_decorator(self):
        """Test @synthesize decorator."""

        @synthesize(name="test_synthesize", model="test-model")
        def synthesize_step(self, data: str) -> str:
            return f"synthesized: {data}"

        step_info = synthesize_step._step_info
        assert step_info.name == "test_synthesize"
        assert step_info.step_type == StepType.SYNTHESIZE
        assert step_info.llm_config.name == "test-model"

    def test_decorator_without_model(self):
        """Test decorator without model configuration."""

        @think()
        def simple_step(self, data: str) -> str:
            return data

        step_info = simple_step._step_info
        assert step_info.llm_config is None

    def test_decorator_with_description(self):
        """Test decorator with custom description."""

        @think(description="Custom step description")
        def desc_step(self, data: str) -> str:
            return data

        step_info = desc_step._step_info
        assert step_info.description == "Custom step description"


class TestStepCompatibility:
    """Test suite for step compatibility checks."""

    def test_compatible_steps(self):
        """Test compatibility between steps with matching types."""

        def step1(data: str) -> str:
            return data

        def step2(text: str) -> str:
            return text

        step_a = Step(
            func=step1,
            step_type=StepType.THINK,
            name="step_a",
            send_to_llm=True,
            metadata=FunctionMetadata.from_function(step1),
        )
        step_b = Step(
            func=step2,
            step_type=StepType.ACT,
            name="step_b",
            send_to_llm=True,
            metadata=FunctionMetadata.from_function(step2),
        )

        assert step_b.is_compatible_with(step_a)

    def test_incompatible_steps(self):
        """Test incompatibility between steps with mismatched types."""

        def step1(data: str) -> str:
            return data

        def step2(num: int) -> str:
            return str(num)

        step_a = Step(
            func=step1,
            step_type=StepType.THINK,
            name="step_a",
            send_to_llm=True,
            metadata=FunctionMetadata.from_function(step1),
        )
        step_b = Step(
            func=step2,
            step_type=StepType.ACT,
            name="step_b",
            send_to_llm=True,
            metadata=FunctionMetadata.from_function(step2),
        )

        assert not step_b.is_compatible_with(step_a)

    def test_step_execution_validation(self):
        """Test step input validation."""

        def sample_step(data: str) -> str:
            return data

        step = Step(
            func=sample_step,
            step_type=StepType.THINK,
            name="sample",
            send_to_llm=True,
            metadata=FunctionMetadata.from_function(sample_step),
        )

        assert step.can_execute_with("test input")
        assert not step.can_execute_with(42)


class TestStepConfig:
    """Test suite for step configuration."""

    def test_default_config(self):
        """Test default step configuration."""

        def sample_step(data: str) -> str:
            return data

        step = Step(
            func=sample_step,
            step_type=StepType.THINK,
            name="sample",
            send_to_llm=True,
        )

        assert step.config.enabled
        assert step.config.retry_count == 0
        assert step.config.required
        assert step.config.pass_result

    def test_custom_config(self):
        """Test custom step configuration."""

        def sample_step(data: str) -> str:
            return data

        config = StepConfig(
            enabled=False, retry_count=3, required=False, pass_result=False
        )

        step = Step(
            func=sample_step,
            step_type=StepType.THINK,
            name="sample",
            send_to_llm=True,
            config=config,
        )

        assert not step.config.enabled
        assert step.config.retry_count == 3
        assert not step.config.required
        assert not step.config.pass_result


def test_step_metadata():
    """Test step metadata extraction."""

    def sample_func(data: str, optional: Optional[int] = None) -> str:
        """Sample function with documentation."""
        return str(data)

    metadata = FunctionMetadata.from_function(sample_func)

    assert metadata.name == "sample_func"
    print(f"DEBUG - Return type actual value: {metadata.return_type}")
    print(f"DEBUG - Return type type: {type(metadata.return_type)}")
    assert "<class 'str'>" == str(metadata.return_type)
    assert metadata.docstring == "Sample function with documentation."
    assert "data" in metadata.arg_types
    assert metadata.arg_types["data"] == str
    assert "optional" in metadata.arg_types
