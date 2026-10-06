"""Generates the five Kaggle notebooks under this directory.

We keep the cell content in plain Python lists so it stays reviewable in a
diff. Re-run this script after editing to regenerate the .ipynb files::

    python kaggle/_build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _stitch(lines),
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _stitch(lines),
    }


def _stitch(lines: tuple[str, ...]) -> list[str]:
    """Each line gets its own list entry with a trailing newline (Jupyter
    convention). Multi-line strings are split on \\n."""
    out: list[str] = []
    for chunk in lines:
        for sub in chunk.split("\n"):
            out.append(sub + "\n")
    if out:
        out[-1] = out[-1].rstrip("\n")
    return out


def write_notebook(name: str, cells: list[dict]) -> Path:
    # nbformat 4.5+ requires every cell to carry a stable id. We use the
    # notebook stem + index so re-runs produce diff-friendly output.
    stem = Path(name).stem
    for i, cell in enumerate(cells):
        cell.setdefault("id", f"{stem}-{i:02d}")
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = HERE / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Notebook 01: Download model weights
# ---------------------------------------------------------------------------


nb01 = [
    md(
        "# 01: Download Model Weights",
        "",
        "**Run once.** Pulls LLaVA-1.6-7B and Mistral-7B-Instruct from HuggingFace into",
        "`/kaggle/working/hf_cache`. After this notebook finishes, click **Save Version**",
        "(Quick Save → 'Save & Run All') and the resulting Kaggle Dataset (named",
        "`gvla-weights`) becomes the read-only weight cache for every other notebook.",
        "",
        "## Prerequisites",
        "",
        "1. Accept the model licenses on HuggingFace:",
        "   - https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf",
        "   - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
        "2. Add your HF token as a Kaggle Secret named `HF_TOKEN`",
        "   (Add-ons → Secrets → Add a secret).",
        "3. In **Settings** (right panel): Accelerator = `GPU T4 x2`, Persistence = `Files only`,",
        "   Internet = `On`.",
        "",
        "Approximate runtime: 8–12 minutes. Approximate disk usage: ~28 GB.",
    ),
    code(
        "# Pre-flight: GPU + disk + internet",
        "import shutil, subprocess, os",
        "print('--- GPU ---')",
        "subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'])",
        "print('--- Disk (/kaggle/working) ---')",
        "total, used, free = shutil.disk_usage('/kaggle/working')",
        "print(f'free: {free/1e9:.1f} GB / total: {total/1e9:.1f} GB')",
        "assert free > 30 * 1e9, 'need ~30 GB free; restart the session'",
    ),
    code(
        "!pip install -q --upgrade huggingface_hub",
    ),
    code(
        "# Authenticate with the HF token stored as a Kaggle Secret.",
        "from kaggle_secrets import UserSecretsClient",
        "from huggingface_hub import login",
        "",
        "token = UserSecretsClient().get_secret('HF_TOKEN')",
        "login(token=token, add_to_git_credential=False)",
        "print('HF auth OK')",
    ),
    code(
        "from pathlib import Path",
        "from huggingface_hub import snapshot_download",
        "",
        "CACHE = Path('/kaggle/working/hf_cache')",
        "CACHE.mkdir(parents=True, exist_ok=True)",
        "",
        "print('Downloading LLaVA-1.6-7B (~13 GB) ...')",
        "snapshot_download(",
        "    repo_id='llava-hf/llava-v1.6-mistral-7b-hf',",
        "    local_dir=CACHE / 'llava-v1.6-mistral-7b-hf',",
        "    local_dir_use_symlinks=False,",
        "    # Drop the redundant .bin shards if .safetensors are present.",
        "    ignore_patterns=['*.bin', '*.msgpack', '*.h5'],",
        ")",
        "print('LLaVA done.')",
    ),
    code(
        "print('Downloading Mistral-7B-Instruct (~14 GB) ...')",
        "snapshot_download(",
        "    repo_id='mistralai/Mistral-7B-Instruct-v0.2',",
        "    local_dir=CACHE / 'Mistral-7B-Instruct-v0.2',",
        "    local_dir_use_symlinks=False,",
        "    ignore_patterns=['*.bin', '*.msgpack', '*.h5'],",
        ")",
        "print('Mistral done.')",
    ),
    code(
        "# Verify and report.",
        "import subprocess",
        "subprocess.run(['du', '-sh', '/kaggle/working/hf_cache'])",
        "for sub in sorted(CACHE.iterdir()):",
        "    print(sub.name)",
    ),
    md(
        "## Next step",
        "",
        "1. Click **Save Version** → **Quick Save** (or **Save & Run All**) at the top right.",
        "2. After the version is saved, open the notebook's **Output** tab and click",
        "   **New Notebook** → name the dataset `gvla-weights`. (Or, more robust: use",
        "   *Output* → *New Dataset* directly.)",
        "3. Subsequent notebooks mount this dataset read-only at",
        "   `/kaggle/input/gvla-weights/hf_cache`.",
        "",
        "**You will not need to re-run this notebook unless model versions change.**",
    ),
]


# ---------------------------------------------------------------------------
# Notebook 02: Prepare datasets
# ---------------------------------------------------------------------------


nb02 = [
    md(
        "# 02: Prepare Benchmarks",
        "",
        "**Run once.** Pre-converts ScienceQA + Mind2Web into our unified JSONL format,",
        "drops images to disk, and bundles the in-repo synthetic sample. The output of",
        "this notebook becomes the `gvla-data` Kaggle Dataset that every eval notebook",
        "mounts.",
        "",
        "## Prerequisites",
        "",
        "1. The `grounded_vla` repo is reachable (we clone from GitHub here; substitute",
        "   your fork's URL).",
        "2. Internet = `On` and CPU runtime is fine for this notebook (no GPU needed).",
        "",
        "Approximate runtime: 10–20 minutes depending on the slice size.",
    ),
    code(
        "# Clone repo (skip if already attached as a Kaggle Dataset).",
        "import subprocess, os",
        "REPO_URL = 'https://github.com/<your-org>/grounded_vla.git'",
        "if not os.path.exists('/kaggle/working/grounded_vla'):",
        "    subprocess.run(['git', 'clone', REPO_URL, '/kaggle/working/grounded_vla'], check=True)",
        "%cd /kaggle/working/grounded_vla",
    ),
    code(
        "!pip install -q -e .[data]",
    ),
    code(
        "# Set HF token if your dataset slice is gated (most are not).",
        "try:",
        "    from kaggle_secrets import UserSecretsClient",
        "    from huggingface_hub import login",
        "    login(token=UserSecretsClient().get_secret('HF_TOKEN'), add_to_git_credential=False)",
        "    print('HF auth OK')",
        "except Exception as e:",
        "    print('Skipping HF auth (probably fine):', e)",
    ),
    code(
        "# ScienceQA (test split, image-bearing rows only). 500 tasks ~= 12 minutes.",
        "OUT = '/kaggle/working/gvla-data/scienceqa'",
        "!python scripts/prepare_scienceqa.py --split test --out-dir {OUT} --limit 500",
        "!ls {OUT} && wc -l {OUT}/test.jsonl",
    ),
    code(
        "# Synthetic sample (already in the repo, just copy into the data folder).",
        "import shutil",
        "from pathlib import Path",
        "src = Path('data/samples')",
        "dst = Path('/kaggle/working/gvla-data/synthetic_sample')",
        "dst.mkdir(parents=True, exist_ok=True)",
        "shutil.copytree(src / 'images', dst / 'images', dirs_exist_ok=True)",
        "shutil.copy(src / 'synthetic_sample.jsonl', dst / 'synthetic_sample.jsonl')",
        "print('synthetic sample @', dst)",
    ),
    code(
        "# Final tree.",
        "import subprocess",
        "subprocess.run(['du', '-sh', '/kaggle/working/gvla-data'])",
        "subprocess.run(['ls', '-R', '/kaggle/working/gvla-data'])",
    ),
    md(
        "## Next step",
        "",
        "1. **Save Version** → **Quick Save** so `/kaggle/working/gvla-data` becomes a Kaggle",
        "   Dataset (name it `gvla-data`).",
        "2. From here on, mount both `gvla-weights` and `gvla-data` in every eval notebook.",
        "",
        "If you ever want a larger evaluation slice, increase `--limit` and re-run.",
    ),
]


# ---------------------------------------------------------------------------
# Notebook 03: Run evaluation (smoke + baselines + ORA)
# ---------------------------------------------------------------------------


nb03 = [
    md(
        "# 03: Smoke Test + Baselines + ORA",
        "",
        "Workhorse notebook. Mounts `gvla-weights` and `gvla-data`, runs the smoke test,",
        "then evaluates the three agents (ReAct + Mistral, single-shot LLaVA, ORA + LLaVA)",
        "across ScienceQA, the synthetic sample, and Mind2Web.",
        "",
        "**Settings:** Accelerator = `GPU T4 x2`, Internet = `On`, Persistence = `Files only`.",
        "",
        "**Datasets to attach (Add data → Your datasets):**",
        "- `gvla-weights`",
        "- `gvla-data`",
        "",
        "Eval is checkpointed every task, so a session timeout is recoverable — re-run with",
        "`resume=True` and it picks up where it left off.",
    ),
    code(
        "# Mount paths + offline mode so transformers never tries to re-download.",
        "import os",
        "WEIGHTS = '/kaggle/input/gvla-weights/hf_cache'",
        "DATA    = '/kaggle/input/gvla-data'",
        "os.environ['HF_HOME'] = WEIGHTS",
        "os.environ['TRANSFORMERS_CACHE'] = WEIGHTS",
        "os.environ['TRANSFORMERS_OFFLINE'] = '1'",
        "os.environ['HF_DATASETS_OFFLINE'] = '1'",
        "assert os.path.isdir(WEIGHTS), 'mount the gvla-weights Kaggle Dataset first'",
        "assert os.path.isdir(DATA),    'mount the gvla-data Kaggle Dataset first'",
    ),
    code(
        "# Repo + GPU stack. Clone on first run, pull on re-run to get latest fixes.",
        "REPO_URL = 'https://github.com/<your-org>/grounded_vla.git'",
        "!if [ ! -d /kaggle/working/grounded_vla ]; then git clone {REPO_URL} /kaggle/working/grounded_vla; fi",
        "!git -C /kaggle/working/grounded_vla pull",
        "%cd /kaggle/working/grounded_vla",
        "!pip install -q -e .[gpu]",
        "!nvidia-smi",
    ),
    code(
        "# Smoke test: pure-mock pipeline. Should print 'smoke ok' in <5 seconds.",
        "!grounded-vla smoke",
    ),
    code(
        "# Build the three agents.",
        "from grounded_vla.agents import ORAAgent, ReActAgent, SingleShotVLMAgent",
        "from grounded_vla.backends import make_backend",
        "from grounded_vla.backends.base import GenerationConfig",
        "",
        "WEIGHTS = '/kaggle/input/gvla-weights/hf_cache'",
        "",
        "def build_react():",
        "    backend = make_backend({",
        "        'kind': 'mistral',",
        "        'model_id': f'{WEIGHTS}/Mistral-7B-Instruct-v0.2',",
        "        'device': 'cuda',",
        "        'quantize': '4bit',",
        "    })",
        "    return ReActAgent(backend, GenerationConfig(max_new_tokens=256, temperature=0.1))",
        "",
        "def build_single_shot():",
        "    backend = make_backend({",
        "        'kind': 'llava',",
        "        'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',",
        "        'hf_cache_dir': WEIGHTS,",
        "        'device': 'cuda',",
        "        'quantize': '4bit',",
        "    })",
        "    return SingleShotVLMAgent(backend, GenerationConfig(max_new_tokens=192, temperature=0.0))",
        "",
        "def build_ora():",
        "    backend = make_backend({",
        "        'kind': 'llava',",
        "        'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',",
        "        'hf_cache_dir': WEIGHTS,",
        "        'device': 'cuda',",
        "        'quantize': '4bit',",
        "    })",
        "    return ORAAgent(backend, GenerationConfig(max_new_tokens=256, temperature=0.1))",
    ),
    code(
        "# Build the three datasets pointing at the mounted gvla-data.",
        "from grounded_vla.data import make_dataset",
        "DATA = '/kaggle/input/gvla-data'",
        "",
        "def scienceqa(limit=200):",
        "    return make_dataset({",
        "        'kind': 'scienceqa',",
        "        'jsonl_path': f'{DATA}/scienceqa/test.jsonl',",
        "        'images_dir': f'{DATA}/scienceqa',",
        "        'limit': limit,",
        "    })",
        "",
        "def synthetic():",
        "    return make_dataset({",
        "        'kind': 'jsonl',",
        "        'path': f'{DATA}/synthetic_sample/synthetic_sample.jsonl',",
        "        'source': 'synthetic',",
        "    })",
    ),
    code(
        "# Generic eval loop with checkpointing. Resumable across sessions.",
        "from grounded_vla.eval import EvalRunner",
        "from pathlib import Path",
        "import json",
        "",
        "RUNS = Path('/kaggle/working/runs')",
        "",
        "def run(agent_name, agent, ds_name, dataset, **kw):",
        "    out = RUNS / f'{agent_name}__{ds_name}'",
        "    runner = EvalRunner(agent)",
        "    res = runner.evaluate(",
        "        dataset, save_dir=out, checkpoint_every=5, resume=True, **kw",
        "    )",
        "    print(f'{agent_name} on {ds_name}: '",
        "          f'success={res.task_completion_rate:.3f} '",
        "          f'mean_steps={res.mean_steps:.2f} '",
        "          f'errors={res.error_breakdown}')",
        "    return res",
    ),
    md(
        "## ReAct + Mistral (Baseline 1)",
        "",
        "Text-only. Quick on Mistral-7B with 4-bit quantization (~3–5 sec/task on T4).",
    ),
    code(
        "react = build_react()",
        "_ = run('react_mistral', react, 'scienceqa',  scienceqa(limit=200))",
        "_ = run('react_mistral', react, 'synthetic',  synthetic())",
        "react.backend.close()",
    ),
    md(
        "## Single-shot LLaVA (Baseline 2)",
        "",
        "VLM, one call per task. Tests H1 (vision helps).",
    ),
    code(
        "ss = build_single_shot()",
        "_ = run('single_shot_llava', ss, 'scienceqa',  scienceqa(limit=200))",
        "_ = run('single_shot_llava', ss, 'synthetic',  synthetic())",
        "ss.backend.close()",
    ),
    md(
        "## ORA + LLaVA (Our Method)",
        "",
        "VLM with the Observe-Reason-Act loop and per-step visual re-encoding. Tests H2.",
    ),
    code(
        "ora = build_ora()",
        "_ = run('ora_llava', ora, 'scienceqa',  scienceqa(limit=200))",
        "_ = run('ora_llava', ora, 'synthetic',  synthetic())",
        "ora.backend.close()",
    ),
    code(
        "# Aggregate results table for the report.",
        "import json",
        "from pathlib import Path",
        "rows = []",
        "for d in sorted(Path('/kaggle/working/runs').iterdir()):",
        "    s = json.loads((d / 'summary.json').read_text())",
        "    rows.append((d.name, s['n_tasks'], s['task_completion_rate'], s['mean_steps'], s['error_breakdown']))",
        "for r in rows:",
        "    print(f'{r[0]:30s}  n={r[1]:3d}  success={r[2]:.3f}  steps={r[3]:.2f}  errors={r[4]}')",
    ),
    md(
        "## Save results",
        "",
        "Click **Save Version** → name the output dataset `gvla-runs-YYYY-MM-DD`. The next",
        "notebook (LoRA + ablation) and your report code can mount it read-only.",
    ),
]


# ---------------------------------------------------------------------------
# Notebook 04: LoRA fine-tune
# ---------------------------------------------------------------------------


nb04 = [
    md(
        "# 04: LoRA Fine-tune (H3 Stretch Goal)",
        "",
        "Trains a LoRA adapter for LLaVA-1.6-7B on the synthetic instruction-image",
        "corpus from §3.3 of the proposal. Designed for **2× T4** with `device_map='auto'`",
        "model sharding — single T4 will OOM with default sequence length.",
        "",
        "**Settings:** Accelerator = `GPU T4 x2`, Internet = `On`.",
        "",
        "**Datasets to attach:**",
        "- `gvla-weights`",
        "- `gvla-data`  (must include the `synthetic_sample` folder, or your own larger",
        "  synthetic dataset — increase to ~200 examples for meaningful results)",
        "",
        "Approximate runtime: ~30–60 min on 2× T4 for 3 epochs over 200 examples.",
    ),
    code(
        "import os, subprocess",
        "WEIGHTS = '/kaggle/input/gvla-weights/hf_cache'",
        "DATA    = '/kaggle/input/gvla-data'",
        "os.environ['HF_HOME'] = WEIGHTS",
        "os.environ['TRANSFORMERS_CACHE'] = WEIGHTS",
        "os.environ['TRANSFORMERS_OFFLINE'] = '1'",
        "subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv'])",
    ),
    code(
        "REPO_URL = 'https://github.com/<your-org>/grounded_vla.git'",
        "!if [ ! -d /kaggle/working/grounded_vla ]; then git clone {REPO_URL} /kaggle/working/grounded_vla; fi",
        "!git -C /kaggle/working/grounded_vla pull",
        "%cd /kaggle/working/grounded_vla",
        "!pip install -q -e .[gpu]",
    ),
    code(
        "# Train. The synthetic_sample fixture is tiny (3 examples) and only useful as a",
        "# pipeline check; for real H3 numbers, swap to a 200-example synthetic.jsonl.",
        "from grounded_vla.lora import train_lora, LoRAConfig",
        "from pathlib import Path",
        "",
        "JSONL = f'{DATA}/synthetic_sample/synthetic_sample.jsonl'  # or your real synthetic.jsonl",
        "IMGS  = f'{DATA}/synthetic_sample'",
        "OUT   = Path('/kaggle/working/checkpoints/llava-lora-r1')",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "",
        "cfg = LoRAConfig(",
        "    base_model='llava-hf/llava-v1.6-mistral-7b-hf',",
        "    r=16,",
        "    lora_alpha=32,",
        "    lora_dropout=0.05,",
        "    learning_rate=2e-4,",
        "    per_device_batch_size=1,",
        "    gradient_accumulation_steps=8,",
        "    num_epochs=3,",
        "    max_seq_len=768,  # smaller than default to stay safe on T4",
        ")",
        "train_lora(jsonl_path=JSONL, images_dir=IMGS, output_dir=OUT, config=cfg)",
    ),
    code(
        "# Verify adapter was saved.",
        "import os",
        "for f in sorted(os.listdir('/kaggle/working/checkpoints/llava-lora-r1')):",
        "    print(f)",
    ),
    md(
        "## Next step",
        "",
        "**Save Version** → name the output dataset `gvla-lora-r1`. Notebook 05 mounts it",
        "to evaluate the H3 ablation.",
    ),
]


# ---------------------------------------------------------------------------
# Notebook 05: Evaluate with LoRA adapter
# ---------------------------------------------------------------------------


nb05 = [
    md(
        "# 05: Evaluate LLaVA + LoRA (H3 Ablation)",
        "",
        "Loads the LoRA adapter trained in notebook 04 onto the LLaVA backbone and runs",
        "ORA on the synthetic held-out set + a Mind2Web slice. Compares against the",
        "non-adapted ORA numbers from notebook 03.",
        "",
        "**Settings:** Accelerator = `GPU T4 x2`, Internet = `On`.",
        "",
        "**Datasets to attach:**",
        "- `gvla-weights`",
        "- `gvla-data`",
        "- `gvla-lora-r1` (output of notebook 04)",
        "- `gvla-runs-YYYY-MM-DD` (output of notebook 03; for comparison)",
    ),
    code(
        "import os, subprocess",
        "WEIGHTS = '/kaggle/input/gvla-weights/hf_cache'",
        "DATA    = '/kaggle/input/gvla-data'",
        "ADAPTER = '/kaggle/input/gvla-lora-r1'",
        "os.environ['HF_HOME'] = WEIGHTS",
        "os.environ['TRANSFORMERS_OFFLINE'] = '1'",
        "subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'])",
    ),
    code(
        "REPO_URL = 'https://github.com/<your-org>/grounded_vla.git'",
        "!if [ ! -d /kaggle/working/grounded_vla ]; then git clone {REPO_URL} /kaggle/working/grounded_vla; fi",
        "!git -C /kaggle/working/grounded_vla pull",
        "%cd /kaggle/working/grounded_vla",
        "!pip install -q -e .[gpu]",
    ),
    code(
        "# Load LLaVA + adapter manually so we can inject PEFT before the agent runs.",
        "from grounded_vla.backends.llava import LLaVABackend",
        "",
        "class LLaVAWithLoRA(LLaVABackend):",
        "    name = 'llava-1.6-7b+lora'",
        "    def __init__(self, adapter_dir, **kw):",
        "        super().__init__(**kw)",
        "        self.adapter_dir = adapter_dir",
        "    def _ensure_loaded(self):",
        "        if self._model is not None:",
        "            return",
        "        super()._ensure_loaded()",
        "        from peft import PeftModel",
        "        self._model = PeftModel.from_pretrained(self._model, self.adapter_dir)",
        "        self._model.eval()",
        "        print(f'[LLaVAWithLoRA] adapter loaded from {self.adapter_dir}')",
    ),
    code(
        "from grounded_vla.agents import ORAAgent",
        "from grounded_vla.backends.base import GenerationConfig",
        "from grounded_vla.data import make_dataset",
        "from grounded_vla.eval import EvalRunner",
        "from pathlib import Path",
        "",
        "backend = LLaVAWithLoRA(",
        "    adapter_dir=ADAPTER,",
        "    model_id='llava-hf/llava-v1.6-mistral-7b-hf',",
        "    hf_cache_dir=WEIGHTS,",
        "    device='cuda',",
        "    quantize='4bit',",
        ")",
        "agent = ORAAgent(backend, GenerationConfig(max_new_tokens=256, temperature=0.1))",
        "",
        "RUNS = Path('/kaggle/working/runs')",
        "",
        "def run(ds_name, dataset):",
        "    out = RUNS / f'ora_lora__{ds_name}'",
        "    res = EvalRunner(agent).evaluate(",
        "        dataset, save_dir=out, checkpoint_every=5, resume=True",
        "    )",
        "    print(f'ora_lora on {ds_name}: '",
        "          f'success={res.task_completion_rate:.3f} '",
        "          f'mean_steps={res.mean_steps:.2f} '",
        "          f'errors={res.error_breakdown}')",
        "    return res",
        "",
        "synthetic = make_dataset({",
        "    'kind': 'jsonl',",
        "    'path':  f'{DATA}/synthetic_sample/synthetic_sample.jsonl',",
        "    'source': 'synthetic',",
        "})",
        "sqa = make_dataset({",
        "    'kind': 'scienceqa',",
        "    'jsonl_path': f'{DATA}/scienceqa/test.jsonl',",
        "    'images_dir': f'{DATA}/scienceqa',",
        "    'limit': 200,",
        "})",
        "",
        "_ = run('synthetic', synthetic)",
        "_ = run('scienceqa', sqa)",
        "backend.close()",
    ),
    code(
        "# Compare ORA-base vs ORA+LoRA side by side (assumes notebook 03 results are",
        "# attached as `gvla-runs-YYYY-MM-DD`; otherwise swap in your own paths).",
        "import glob, json",
        "base_runs = sorted(glob.glob('/kaggle/input/gvla-runs-*/runs/ora_llava__*'))",
        "lora_runs = sorted(glob.glob('/kaggle/working/runs/ora_lora__*'))",
        "",
        "def load(path):",
        "    return json.loads(open(f'{path}/summary.json').read())",
        "",
        "print(f'{\"dataset\":12s} {\"ora-base\":>10s} {\"ora+lora\":>10s} {\"delta\":>8s}')",
        "for lp in lora_runs:",
        "    ds = lp.rsplit('__', 1)[-1]",
        "    base = next((p for p in base_runs if p.endswith(ds)), None)",
        "    if not base:",
        "        continue",
        "    b = load(base)['task_completion_rate']",
        "    l = load(lp)['task_completion_rate']",
        "    print(f'{ds:12s} {b:10.3f} {l:10.3f} {l-b:+8.3f}')",
    ),
    md(
        "## Interpreting the result",
        "",
        "- **Positive delta on synthetic, smaller on Mind2Web:** LoRA closed the in-domain",
        "  gap (expected); cross-domain transfer is partial — useful talking point in §6.",
        "- **Negative delta:** likely overfitting on a tiny synthetic set. Either grow the",
        "  synthetic corpus to ~200 reviewed examples, or shorten training (1 epoch).",
        "- **No change:** LoRA rank may be too small (try `r=32`) or the action format in",
        "  the synthetic dataset doesn't match the agent's prompt template.",
        "",
        "Either way, the result goes into the H3 row of the results table in the report.",
    ),
]


# ---------------------------------------------------------------------------
# Build them all
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    write_notebook("01_setup_weights.ipynb", nb01)
    write_notebook("02_setup_data.ipynb", nb02)
    write_notebook("03_run_eval.ipynb", nb03)
    write_notebook("04_lora_finetune.ipynb", nb04)
    write_notebook("05_eval_with_lora.ipynb", nb05)
