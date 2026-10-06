"""Unified CLI.

Examples::

    # Quick smoke test against the sample synthetic tasks (mock backend, no GPU).
    grounded-vla smoke

    # Run ORA + LLaVA on the first 20 ScienceQA tasks.
    grounded-vla eval --config configs/ora_llava.yaml \
        --dataset-config configs/datasets/scienceqa.yaml --limit 20

    # Generate synthetic candidates (mock backend for testing the pipeline).
    grounded-vla build-synthetic --manifest data/samples/cc_manifest.json \
        --out data/synthetic --backend-kind mock
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import yaml

from . import __version__
from .agents import ORAAgent, ReActAgent, SingleShotVLMAgent
from .backends import make_backend
from .backends.base import GenerationConfig
from .data import make_dataset
from .eval import EvalRunner
from .synthetic import SyntheticBuilder
from .synthetic.builder import ImageRecord
from .utils.logging import get_logger

_log = get_logger(__name__)


AGENT_CLASSES = {
    "react": ReActAgent,
    "single_shot": SingleShotVLMAgent,
    "ora": ORAAgent,
}


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_agent(cfg: dict):
    backend = make_backend(cfg["backend"])
    agent_kind = cfg["agent"]["kind"]
    cls = AGENT_CLASSES[agent_kind]
    gen_cfg_dict = cfg["agent"].get("generation", {})
    gen_cfg = GenerationConfig(**gen_cfg_dict) if gen_cfg_dict else None
    extra = {k: v for k, v in cfg["agent"].items() if k not in ("kind", "generation")}
    return cls(backend=backend, gen_config=gen_cfg, **extra)


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """Grounded Vision-Language Agents for Instruction Following."""


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Agent/backend YAML")
@click.option("--dataset-config", required=True, type=click.Path(exists=True))
@click.option("--limit", type=int, default=None, help="Max tasks to evaluate")
@click.option("--save-dir", type=click.Path(), default=None, help="Where to dump results")
def eval(config: str, dataset_config: str, limit: Optional[int], save_dir: Optional[str]) -> None:
    """Run an agent on a dataset and report aggregate metrics."""
    agent_cfg = _load_yaml(config)
    dataset_cfg = _load_yaml(dataset_config)
    agent = _build_agent(agent_cfg)
    dataset = make_dataset(dataset_cfg)

    runner = EvalRunner(agent)
    result = runner.evaluate(dataset, limit=limit, save_dir=save_dir)
    click.echo(json.dumps(
        {
            "agent": result.agent_name,
            "dataset": result.dataset,
            "n_tasks": result.n_tasks,
            "task_completion_rate": round(result.task_completion_rate, 4),
            "mean_steps": round(result.mean_steps, 2),
            "error_breakdown": result.error_breakdown,
        },
        indent=2,
    ))


@cli.command()
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--backend-kind", default="mock", help="mock | llava")
@click.option("--n", type=int, default=None, help="Cap the number of candidates to generate")
def build_synthetic(manifest: str, out: str, backend_kind: str, n: Optional[int]) -> None:
    """Generate the synthetic dataset from a CC image manifest."""
    records = [ImageRecord(**r) for r in json.loads(Path(manifest).read_text())]
    for r in records:
        r.image_path = Path(r.image_path)
    backend = make_backend({"kind": backend_kind, "supports_vision": True, "policy": "random"}) \
        if backend_kind == "mock" else make_backend({"kind": backend_kind})
    builder = SyntheticBuilder(backend=backend, out_dir=out)
    staged = builder.build(records, n=n)
    click.echo(f"staged {staged} candidates -> {out}/staging")


@cli.command()
@click.option("--out", required=True, type=click.Path(exists=True))
def finalize_synthetic(out: str) -> None:
    """After two-person review, write the approved-subset JSONL."""
    from .backends.mock import MockBackend
    # We only need the queue here; pass a throwaway backend.
    builder = SyntheticBuilder(backend=MockBackend(), out_dir=out)
    path = builder.finalize()
    click.echo(f"wrote {path}")


@cli.command()
def smoke() -> None:
    """End-to-end smoke test using the mock backend and bundled sample tasks.

    This is what CI runs. It exits nonzero if anything throws.
    """
    from .data.base import JsonlDataset

    sample_path = Path(__file__).parent.parent / "data" / "samples" / "synthetic_sample.jsonl"
    if not sample_path.exists():
        raise click.ClickException(f"missing sample fixture: {sample_path}")

    dataset = JsonlDataset(sample_path, source="synthetic")
    backend = make_backend({"kind": "mock", "policy": "oracle"})
    agent = ORAAgent(backend=backend)
    runner = EvalRunner(agent)
    result = runner.evaluate(dataset)
    click.echo(
        f"smoke ok: agent={result.agent_name}, "
        f"n={result.n_tasks}, "
        f"success={result.task_completion_rate:.2f}, "
        f"mean_steps={result.mean_steps:.2f}"
    )


if __name__ == "__main__":
    cli()
