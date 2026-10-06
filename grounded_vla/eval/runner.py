"""EvalRunner: ties agents, datasets, and metrics together.

Usage::

    runner = EvalRunner(agent, env_factory)
    result = runner.evaluate(dataset, limit=50)

Results are also serialized to disk (trajectories + aggregate numbers) so
the final report can load them directly without re-running anything.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from ..agents.base import Agent
from ..data.base import Dataset
from ..env import Environment, StaticQAEnv, TaskReplayEnv
from ..schemas import RunResult, Task, Trajectory
from ..utils.logging import get_logger
from .error_analysis import categorize_error
from .metrics import score_trajectory, step_efficiency

_log = get_logger(__name__)


def default_env_factory(task: Task) -> Environment:
    """Pick an environment based on the task source.

    ScienceQA -> StaticQAEnv; everything else -> TaskReplayEnv.
    """
    if task.source == "scienceqa":
        return StaticQAEnv()
    return TaskReplayEnv()


class EvalRunner:
    def __init__(
        self,
        agent: Agent,
        env_factory: Callable[[Task], Environment] = default_env_factory,
    ) -> None:
        self.agent = agent
        self.env_factory = env_factory

    def evaluate(
        self,
        dataset: Dataset,
        limit: Optional[int] = None,
        save_dir: Optional[Path | str] = None,
        checkpoint_every: int = 0,
        resume: bool = False,
    ) -> RunResult:
        """Run the agent on every task in ``dataset``.

        Args:
            dataset: streaming dataset of Tasks.
            limit: optionally cap the number of tasks evaluated.
            save_dir: if set, summary + per-task trajectories are written here.
            checkpoint_every: if > 0 and ``save_dir`` is set, flush partial
                results to disk every N tasks. Crucial on Kaggle where 9-hour
                session limits can otherwise destroy a long sweep.
            resume: if True and ``save_dir`` exists with prior trajectories,
                skip tasks whose trajectory file already exists. Lets you
                resume an interrupted run from where it left off.
        """
        trajs: list[Trajectory] = []
        successes = 0
        step_counts: list[int] = []
        errors: Counter[str] = Counter()
        n = 0

        # Prepare on-disk layout up front if checkpointing or resuming.
        save_path = Path(save_dir) if save_dir is not None else None
        already_done: set[str] = set()
        if save_path is not None:
            (save_path / "trajectories").mkdir(parents=True, exist_ok=True)
            if resume:
                for fp in (save_path / "trajectories").glob("*.json"):
                    already_done.add(fp.stem)
                if already_done:
                    _log.info("resuming: %d tasks already on disk", len(already_done))

        for i, task in enumerate(dataset):
            if limit is not None and i >= limit:
                break
            if task.task_id in already_done:
                # Re-load the prior trajectory so totals remain consistent.
                prior_path = save_path / "trajectories" / f"{task.task_id}.json"  # type: ignore[union-attr]
                try:
                    traj = Trajectory.model_validate_json(prior_path.read_text())
                except Exception:
                    _log.warning("could not re-read %s; re-running task", prior_path)
                else:
                    score = score_trajectory(task, traj)
                    if score.success:
                        successes += 1
                    step_counts.append(step_efficiency(traj))
                    errors[categorize_error(task, traj, score.success).value] += 1
                    trajs.append(traj)
                    n += 1
                    continue

            env = self.env_factory(task)
            try:
                traj = self.agent.run(task, env)
            except Exception as e:  # never let one bad task kill the whole eval
                _log.exception("agent crashed on %s", task.task_id)
                traj = Trajectory(task_id=task.task_id)
                traj.truncated = True
                errors["agent_exception"] += 1
            score = score_trajectory(task, traj)
            if score.success:
                successes += 1
            step_counts.append(step_efficiency(traj))
            errors[categorize_error(task, traj, score.success).value] += 1
            trajs.append(traj)
            n += 1

            # Per-task checkpoint write (cheap, idempotent) so resume works
            # even if a session dies between full checkpoint flushes.
            if save_path is not None:
                self._write_trajectory(save_path, traj)

            if (
                checkpoint_every > 0
                and save_path is not None
                and n % checkpoint_every == 0
            ):
                self._write_summary(
                    save_path,
                    self.agent.name,
                    dataset.name,
                    n,
                    successes,
                    step_counts,
                    errors,
                )
                _log.info("checkpoint @ task %d -> %s", n, save_path)

        result = RunResult(
            agent_name=self.agent.name,
            dataset=dataset.name,
            n_tasks=n,
            task_completion_rate=(successes / n) if n else 0.0,
            mean_steps=(sum(step_counts) / len(step_counts)) if step_counts else 0.0,
            error_breakdown=dict(errors),
            trajectories=trajs,
        )

        if save_path is not None:
            self._save(result, save_path)
        return result

    @staticmethod
    def _write_trajectory(save_dir: Path, traj: Trajectory) -> None:
        tdir = save_dir / "trajectories"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{traj.task_id}.json").write_text(traj.model_dump_json(indent=2))

    @staticmethod
    def _write_summary(
        save_dir: Path,
        agent_name: str,
        dataset_name: str,
        n: int,
        successes: int,
        step_counts: list[int],
        errors: Counter[str],
    ) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "summary.json").write_text(
            json.dumps(
                {
                    "agent_name": agent_name,
                    "dataset": dataset_name,
                    "n_tasks": n,
                    "task_completion_rate": (successes / n) if n else 0.0,
                    "mean_steps": (sum(step_counts) / len(step_counts)) if step_counts else 0.0,
                    "error_breakdown": dict(errors),
                },
                indent=2,
            )
        )

    @classmethod
    def _save(cls, result: RunResult, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cls._write_summary(
            save_dir,
            result.agent_name,
            result.dataset,
            result.n_tasks,
            int(round(result.task_completion_rate * result.n_tasks)),
            [],  # mean_steps already computed; pass empty to avoid double-averaging
            Counter(result.error_breakdown),
        )
        # Re-write summary.json with the precomputed mean_steps so we don't
        # lose precision through the round-trip above.
        (save_dir / "summary.json").write_text(
            json.dumps(
                {
                    "agent_name": result.agent_name,
                    "dataset": result.dataset,
                    "n_tasks": result.n_tasks,
                    "task_completion_rate": result.task_completion_rate,
                    "mean_steps": result.mean_steps,
                    "error_breakdown": result.error_breakdown,
                },
                indent=2,
            )
        )
        for t in result.trajectories:
            cls._write_trajectory(save_dir, t)
