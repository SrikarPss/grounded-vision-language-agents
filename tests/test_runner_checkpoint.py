"""Tests for EvalRunner's checkpoint + resume behavior (added for Kaggle)."""
from pathlib import Path

from grounded_vla.agents import ORAAgent
from grounded_vla.backends.mock import MockBackend
from grounded_vla.data.base import JsonlDataset
from grounded_vla.eval import EvalRunner


SAMPLE = Path(__file__).parent.parent / "data" / "samples" / "synthetic_sample.jsonl"


def test_checkpoint_writes_per_task_files(tmp_path):
    dataset = JsonlDataset(SAMPLE, source="synthetic")
    agent = ORAAgent(MockBackend(policy="oracle", supports_vision=True))
    runner = EvalRunner(agent)
    runner.evaluate(dataset, save_dir=tmp_path / "run", checkpoint_every=1)

    # Each of the three sample tasks should have an on-disk trajectory.
    tdir = tmp_path / "run" / "trajectories"
    assert sorted(p.name for p in tdir.glob("*.json")) == [
        "syn_diagram.json",
        "syn_login.json",
        "syn_submit.json",
    ]
    assert (tmp_path / "run" / "summary.json").exists()


def test_resume_skips_already_done_tasks(tmp_path):
    dataset = JsonlDataset(SAMPLE, source="synthetic")
    agent = ORAAgent(MockBackend(policy="oracle", supports_vision=True))
    runner = EvalRunner(agent)

    # First pass: do all three tasks.
    r1 = runner.evaluate(dataset, save_dir=tmp_path / "run", checkpoint_every=1)
    assert r1.n_tasks == 3

    # Replace the agent with one that would fail (no vision) — if resume
    # works, it should never get called because all trajectories exist.
    runner2 = EvalRunner(ORAAgent(MockBackend(policy="random", supports_vision=True)))
    r2 = runner2.evaluate(
        JsonlDataset(SAMPLE, source="synthetic"),
        save_dir=tmp_path / "run",
        resume=True,
    )
    # Same totals as before — the agent didn't actually run.
    assert r2.n_tasks == 3
    assert r2.task_completion_rate == r1.task_completion_rate
