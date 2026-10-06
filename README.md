# Grounded Vision-Language Agents

Do vision and a structured action loop help agents follow multi-step
instructions in visual environments? This repo evaluates three agents on
ScienceQA and a synthetic grounded-UI task set to test whether **visual
grounding** and a **structured Observe → Reason → Act (ORA) loop** improve
instruction following over a text-only ReAct baseline.

| Agent                   | Backbone            | Vision? | Agentic loop? |
|-------------------------|---------------------|---------|---------------|
| `react`                 | Mistral-7B-Instruct | no      | yes (ReAct)   |
| `single_shot`           | LLaVA-1.6-7B        | yes     | no            |
| `ora` (primary method)  | LLaVA-1.6-7B        | yes     | yes (ORA)     |

What sets the ORA loop apart from ReAct-with-images is that **the screenshot
or diagram is re-encoded through the vision tower at every step**, so each
action decision is grounded in the current visual state rather than a stale
text summary. An optional LoRA adapter can be trained on top of the ORA agent.

## Quick start (no GPU required)

```bash
pip install -e .
grounded-vla smoke
```

`smoke` runs the full pipeline (ORA agent + mock backend) against three
bundled sample tasks. It exits nonzero if any piece of the pipeline breaks,
which makes it handy as a CI gate.

## GPU quick start

```bash
pip install -e .[gpu,data]
# one-time preprocessing of the public benchmarks
python scripts/prepare_mind2web.py --split test_task --out-dir data/mind2web --limit 200
python scripts/prepare_scienceqa.py --split test --out-dir data/scienceqa --limit 200

# run the primary method
grounded-vla eval \
  --config configs/ora_llava.yaml \
  --dataset-config configs/datasets/scienceqa.yaml \
  --limit 50 --save-dir runs/ora_scienceqa
```

`runs/ora_scienceqa/summary.json` holds the aggregate metrics; individual
trajectories land in `runs/ora_scienceqa/trajectories/` for error analysis.

Run the full agent × dataset sweep with `python scripts/run_full_eval.py
--save-root runs/$(date +%F)`.

## Results

Task completion rate (%). ScienceQA uses 200 diagram-based test examples.

| Agent                     | ScienceQA (n=200) | Synthetic        |
|---------------------------|-------------------|------------------|
| ReAct-Mistral (text-only) | 23.5              | 13.0 (n=46)      |
| Single-Shot LLaVA         | 46.5              | 37.0 (n=46)      |
| ORA-LLaVA                 | 56.5              | 41.3 (n=46)      |
| ORA-LLaVA + LoRA          | 57.5              | 92.0 (n=200)     |

Notes:

- The three baseline synthetic runs used 46 tasks, while the LoRA run used
  200, so the synthetic column is not a like-for-like comparison.
- The 92.0% is in-domain: the adapter was trained on the same synthetic task
  generator it is evaluated on. Generalization to harder layouts is untested.
- The LoRA adapter (r=16) trains 0.21% of model parameters, in roughly 17
  minutes on a single A100 40 GB.
- Masking the prompt and image-patch tokens out of the loss was necessary
  for the LoRA run to train properly.
- A Mind2Web loader and preprocessing script are included, but Mind2Web is
  not part of the results above.

## Repo tour

```
grounded_vla/
├── schemas.py          # Task, Action, Observation, Trajectory, RunResult
├── action_parser.py    # Robust Thought/Action parser (JSON + NL fallback)
├── env.py              # Environments: StaticQAEnv and TaskReplayEnv
├── backends/
│   ├── base.py         # Backend interface
│   ├── mock.py         # Deterministic stand-in (used by CI + smoke)
│   ├── llava.py        # LLaVA-1.6-7B (lazy-loaded, 4-bit quantized)
│   └── mistral.py      # Mistral-7B-Instruct (text-only baseline)
├── agents/
│   ├── base.py         # Agent base class
│   ├── prompts.py      # ReAct / single-shot / ORA prompt templates
│   ├── react_agent.py  # Text-only baseline
│   ├── single_shot_agent.py   # Vision, single-call baseline
│   └── ora_agent.py    # ORA agent (primary method)
├── data/
│   ├── base.py         # Streaming Dataset + JsonlDataset
│   ├── mind2web.py     # Mind2Web loader (HF + JSONL modes)
│   ├── scienceqa.py    # ScienceQA loader
│   └── synthetic.py    # Synthetic corpus loader
├── synthetic/
│   ├── builder.py      # Generates candidate triples from a CC manifest
│   └── review.py       # Two-person review queue for synthetic samples
├── eval/
│   ├── metrics.py      # Per-benchmark success + step efficiency
│   ├── error_analysis.py  # visual_misgrounding / reasoning / parse / truncated
│   └── runner.py       # Ties it all together; writes runs/ artifacts
├── lora.py             # PEFT-LoRA fine-tuning on the synthetic set
├── cli.py              # `grounded-vla` entry point
└── utils/              # Image loader + Rich-flavored logger
```

## Components → code

| Component                                      | Code                                                        |
|------------------------------------------------|-------------------------------------------------------------|
| Backbone VLM (LLaVA-1.6)                       | `grounded_vla/backends/llava.py`                            |
| Text-only baseline (Mistral-7B + ReAct)        | `grounded_vla/backends/mistral.py`, `agents/react_agent.py` |
| ORA loop                                       | `grounded_vla/agents/ora_agent.py`, `agents/prompts.py`     |
| Mind2Web loader                                | `grounded_vla/data/mind2web.py`                             |
| ScienceQA loader                               | `grounded_vla/data/scienceqa.py`                            |
| Synthetic dataset + two-person review          | `grounded_vla/synthetic/{builder,review}.py`                |
| Agent configs and full sweep                   | `configs/*.yaml`, `scripts/run_full_eval.py`                |
| Metrics (completion, step efficiency, errors)  | `grounded_vla/eval/metrics.py`, `error_analysis.py`         |
| LoRA fine-tuning                               | `grounded_vla/lora.py`                                      |

## Experiments

Three pairwise comparisons isolate one design choice each:

- **Does vision help?** Compare `react` vs `single_shot`. Same reasoning
  budget (one call, no loop); the only difference is the presence of vision.
- **Does the ORA loop help?** Compare `single_shot` vs `ora`. Same backbone
  and vision input; the only difference is the agentic loop with per-step
  re-encoding.
- **Does a small LoRA adapter help?** Compare `ora` vs `ora + LoRA adapter`
  on the synthetic split, using ScienceQA to check for forgetting. Requires
  `grounded_vla.lora.train_lora` and GPU time.

## Tests

```bash
pip install -e .[dev]
pytest
```

Tests cover: action parser (JSON + NL fallback), the ORA loop (re-encoding
changes outputs), the ReAct loop (parsing failures surface correctly),
metrics (answer normalization), and an end-to-end smoke run.
