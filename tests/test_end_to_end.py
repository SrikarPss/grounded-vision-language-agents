"""End-to-end smoke test: ORA + mock backend on the bundled sample tasks."""
from pathlib import Path

from grounded_vla.agents import ORAAgent, ReActAgent
from grounded_vla.backends.mock import MockBackend
from grounded_vla.data.base import JsonlDataset
from grounded_vla.eval import EvalRunner


SAMPLE = Path(__file__).parent.parent / "data" / "samples" / "synthetic_sample.jsonl"


def test_ora_full_pipeline_on_samples(tmp_path):
    dataset = JsonlDataset(SAMPLE, source="synthetic")
    agent = ORAAgent(MockBackend(policy="oracle", supports_vision=True))
    runner = EvalRunner(agent)
    result = runner.evaluate(dataset, save_dir=tmp_path / "run")
    assert result.n_tasks == 3
    # Oracle mock + correct gold hints => 100% completion.
    assert result.task_completion_rate == 1.0
    # All three tasks are single-step; ORA ends as soon as the terminal
    # action fires, so mean_steps should be 1.0.
    assert abs(result.mean_steps - 1.0) < 1e-6
    # Summary file was written.
    assert (tmp_path / "run" / "summary.json").exists()


def test_react_full_pipeline_on_samples(tmp_path):
    dataset = JsonlDataset(SAMPLE, source="synthetic")
    agent = ReActAgent(MockBackend(policy="oracle", supports_vision=False))
    runner = EvalRunner(agent)
    result = runner.evaluate(dataset)
    assert result.n_tasks == 3
    # The oracle mock also works for the text-only baseline here, so this
    # serves as a regression test that ReAct runs without crashing.
    assert result.task_completion_rate > 0.0
