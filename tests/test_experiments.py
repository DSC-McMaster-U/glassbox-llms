"""
Tests for the experiment framework: BaseExperiment, ExperimentResult, registry.
"""

import pytest
from typing import Any, Dict


class TestExperimentResult:
    def test_summary(self):
        from glassboxllms.experiments.base import ExperimentResult

        result = ExperimentResult(
            experiment_type="test",
            model_name="gpt2",
            status="success",
            metrics={"accuracy": 0.95, "loss": 0.123},
        )

        s = result.summary()
        assert "TEST" in s
        assert "gpt2" in s
        assert "success" in s

    def test_to_dict(self):
        from glassboxllms.experiments.base import ExperimentResult

        result = ExperimentResult(
            experiment_type="test",
            model_name="gpt2",
            status="success",
            metrics={"acc": 0.9},
        )

        d = result.to_dict()
        assert d["experiment_type"] == "test"
        assert d["metrics"]["acc"] == 0.9
        assert "timestamp" in d


class TestRegistry:
    def test_list_experiments(self):
        from glassboxllms.experiments import list_experiments

        names = list_experiments()
        assert isinstance(names, list)
        assert "logit_lens" in names
        assert "probing" in names
        assert "cot_faithfulness" in names

    def test_get_experiment(self):
        from glassboxllms.experiments import get_experiment
        from glassboxllms.experiments.base import BaseExperiment

        exp = get_experiment("logit_lens")
        assert isinstance(exp, BaseExperiment)
        assert exp.name == "logit_lens"

    def test_get_unknown_raises(self):
        from glassboxllms.experiments import get_experiment

        with pytest.raises(KeyError, match="Unknown experiment"):
            get_experiment("nonexistent_experiment")

    def test_register_custom(self):
        from glassboxllms.experiments import register_experiment, get_experiment
        from glassboxllms.experiments.base import BaseExperiment, ExperimentResult

        class DummyExperiment(BaseExperiment):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def default_config(self) -> Dict[str, Any]:
                return {"foo": "bar"}

            def validate_config(self, config: Dict[str, Any]) -> bool:
                return True

            def run(self, config: Dict[str, Any]) -> ExperimentResult:
                return ExperimentResult(
                    experiment_type="dummy",
                    model_name="test",
                    status="success",
                    metrics={"x": 1},
                )

        register_experiment("dummy", DummyExperiment)
        exp = get_experiment("dummy")
        result = exp.run({})
        assert result.status == "success"


class TestLogitLensExperiment:
    def test_validate_config(self):
        from glassboxllms.experiments.logit_lens import LogitLensExperiment

        exp = LogitLensExperiment()
        assert exp.validate_config({"model_name": "gpt2", "text": "hello"})
        assert exp.validate_config({})  # defaults cover required keys

    def test_default_config(self):
        from glassboxllms.experiments.logit_lens import LogitLensExperiment

        exp = LogitLensExperiment()
        cfg = exp.default_config
        assert "model_name" in cfg
        assert "text" in cfg
        assert "top_k" in cfg
