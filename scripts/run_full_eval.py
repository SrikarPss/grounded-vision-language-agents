"""End-to-end evaluation sweep: all three agents x all three benchmarks.

This produces the results table that goes into the final report. It is
embarrassingly parallel across (agent, dataset) pairs, but we keep it
sequential so a single GPU can hold model weights in cache.

Usage::

    python scripts/run_full_eval.py --save-root runs/2026-05-02
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from grounded_vla.agents import ORAAgent, ReActAgent, SingleShotVLMAgent
from grounded_vla.backends import make_backend
from grounded_vla.backends.base import GenerationConfig
from grounded_vla.data import make_dataset
from grounded_vla.eval import EvalRunner
from grounded_vla.utils.logging import get_logger

_log = get_logger(__name__)


AGENT_CLASSES = {"react": ReActAgent, "single_shot": SingleShotVLMAgent, "ora": ORAAgent}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_agent(cfg: dict):
    backend = make_backend(cfg["backend"])
    gen = cfg["agent"].get("generation", {})
    gen_cfg = GenerationConfig(**gen) if gen else None
    extra = {k: v for k, v in cfg["agent"].items() if k not in ("kind", "generation")}
    return AGENT_CLASSES[cfg["agent"]["kind"]](backend=backend, gen_config=gen_cfg, **extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--save-root", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    configs_dir = Path(args.configs_dir)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    agents = ["react_mistral", "single_shot_llava", "ora_llava"]
    datasets = ["mind2web", "scienceqa", "synthetic"]

    summary = {}
    for a_name in agents:
        agent_cfg = _load_yaml(configs_dir / f"{a_name}.yaml")
        agent = build_agent(agent_cfg)
        for d_name in datasets:
            ds_cfg = _load_yaml(configs_dir / "datasets" / f"{d_name}.yaml")
            # ReAct has no vision; skip combinations that don't make sense.
            if a_name == "react_mistral" and d_name == "scienceqa":
                _log.info("still running %s on %s (text-only upper bound)", a_name, d_name)
            ds = make_dataset(ds_cfg)
            run_dir = save_root / f"{a_name}__{d_name}"
            runner = EvalRunner(agent)
            result = runner.evaluate(ds, limit=args.limit, save_dir=run_dir)
            summary[f"{a_name}__{d_name}"] = {
                "task_completion_rate": result.task_completion_rate,
                "mean_steps": result.mean_steps,
                "error_breakdown": result.error_breakdown,
                "n_tasks": result.n_tasks,
            }
            _log.info("%s on %s: %.3f (%d tasks)", a_name, d_name,
                      result.task_completion_rate, result.n_tasks)
        agent.backend.close()

    (save_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {save_root / 'summary.json'}")


if __name__ == "__main__":
    main()
