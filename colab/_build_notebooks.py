"""Generates the five Colab Pro+ notebooks under this directory.

Mirrors `kaggle/_build_notebooks.py` but tuned for Colab Pro+:

- Persistence layer is Google Drive (mounted at ``/content/drive``), not
  Kaggle Datasets. Weights, datasets, and run artifacts all live under
  ``/content/drive/MyDrive/grounded_vla/`` so they survive runtime
  restarts.
- Single GPU (A100 expected on Pro+; T4 fallback wired in).
- Token comes from Colab's built-in ``userdata`` Secrets, not Kaggle Secrets.
- Configs are tuned for A100 (full ``max_seq_len=1024``, larger batches,
  no gradient checkpointing).
- Compute units are the budget to watch; each cell prints estimated cost.

Re-run after editing::

    python colab/_build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helpers (same shape as the Kaggle generator so changes flow both ways).
# ---------------------------------------------------------------------------


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _stitch(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _stitch(lines),
    }


def _stitch(lines: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for chunk in lines:
        for sub in chunk.split("\n"):
            out.append(sub + "\n")
    if out:
        out[-1] = out[-1].rstrip("\n")
    return out


def write_notebook(name: str, cells: list[dict]) -> Path:
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
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = HERE / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Shared boilerplate cells (Drive mount + repo clone + GPU detection)
# ---------------------------------------------------------------------------


DRIVE_MOUNT_CELL = code(
    "# Mount Google Drive. The first run prompts for OAuth; subsequent runs",
    "# in the same session reuse the token automatically.",
    "from google.colab import drive",
    "drive.mount('/content/drive')",
    "",
    "import os, pathlib",
    "ROOT = pathlib.Path('/content/drive/MyDrive/grounded_vla')",
    "ROOT.mkdir(parents=True, exist_ok=True)",
    "print('drive root:', ROOT)",
)


GPU_CHECK_CELL = code(
    "# Verify accelerator. Pro+ should give you A100 most of the time; if you",
    "# see T4, change Runtime -> Change runtime type -> A100 GPU and re-run.",
    "import subprocess",
    "subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free',",
    "                '--format=csv'])",
)


CLONE_REPO_CELLS = [
    code(
        "# Clone repo if absent, then always pull to get the latest fixes.",
        "# With an editable install (-e) a pull is all that's needed — no reinstall.",
        "REPO_URL = 'https://github.com/<your-org>/grounded_vla.git'",
        "import os, subprocess",
        "if not os.path.exists('/content/grounded_vla'):",
        "    subprocess.run(['git', 'clone', REPO_URL, '/content/grounded_vla'], check=True)",
        "subprocess.run(['git', '-C', '/content/grounded_vla', 'pull'], check=True)",
        "%cd /content/grounded_vla",
    ),
    code(
        "# Install repo + GPU stack. Quiet to keep the cell output sane.",
        "!pip install -q -e .[gpu,data]",
    ),
]


HF_AUTH_CELL = code(
    "# Authenticate with HF using a token stored as a Colab Secret.",
    "# Add the secret first: left sidebar -> key icon -> 'HF_TOKEN' -> paste token,",
    "# then toggle 'Notebook access' on for this notebook.",
    "from google.colab import userdata",
    "from huggingface_hub import login",
    "login(token=userdata.get('HF_TOKEN'), add_to_git_credential=False)",
    "print('HF auth OK')",
)


# ---------------------------------------------------------------------------
# Notebook 01: Drive setup + model weights
# ---------------------------------------------------------------------------


nb01 = [
    md(
        "# 01: Drive Setup + Download Model Weights",
        "",
        "**Run once.** Mounts Google Drive and downloads LLaVA-1.6-7B + Mistral-7B-Instruct",
        "into `/content/drive/MyDrive/grounded_vla/hf_cache`. Subsequent notebooks read",
        "weights from Drive without re-downloading.",
        "",
        "## Prerequisites",
        "",
        "1. Accept the model licenses on HuggingFace:",
        "   - https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf",
        "   - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
        "2. Add a `HF_TOKEN` secret in Colab (left sidebar → key icon).",
        "3. Runtime → Change runtime type → **A100 GPU** (or T4 if A100 unavailable).",
        "4. Make sure your Drive has **~30 GB free**.",
        "",
        "**Approximate cost:** ~10 min runtime, ~1 compute unit on A100.",
    ),
    DRIVE_MOUNT_CELL,
    GPU_CHECK_CELL,
    code(
        "# Disk check on Drive (Colab caches Drive lazily, so 'free' here is the",
        "# Colab side; actual Drive free space is in your Google account quota).",
        "import shutil",
        "total, used, free = shutil.disk_usage('/content/drive/MyDrive')",
        "print(f'/content/drive/MyDrive: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total')",
    ),
    code(
        "!pip install -q --upgrade huggingface_hub",
    ),
    HF_AUTH_CELL,
    code(
        "from huggingface_hub import snapshot_download",
        "from pathlib import Path",
        "",
        "CACHE = Path('/content/drive/MyDrive/grounded_vla/hf_cache')",
        "CACHE.mkdir(parents=True, exist_ok=True)",
        "",
        "print('Downloading LLaVA-1.6-7B (~13 GB) ...')",
        "snapshot_download(",
        "    repo_id='llava-hf/llava-v1.6-mistral-7b-hf',",
        "    local_dir=CACHE / 'llava-v1.6-mistral-7b-hf',",
        "    local_dir_use_symlinks=False,",
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
        "import subprocess",
        "subprocess.run(['du', '-sh', '/content/drive/MyDrive/grounded_vla/hf_cache'])",
        "for sub in sorted(CACHE.iterdir()):",
        "    print(sub.name)",
    ),
    md(
        "## Done",
        "",
        "Weights are now permanently in your Drive. **You don't need to re-run this**",
        "unless model versions change. Move on to `02_setup_data.ipynb`.",
        "",
        "Tip: if Drive sync is slow afterwards, you can disable Drive auto-sync on the",
        "`hf_cache` folder via the Drive UI (right-click → 'Available offline' → off).",
    ),
]


# ---------------------------------------------------------------------------
# Notebook 02: Datasets
# ---------------------------------------------------------------------------


nb02 = [
    md(
        "# 02: Prepare Benchmarks",
        "",
        "**Run once.** Pulls ScienceQA + Mind2Web from HuggingFace, runs the in-repo",
        "preprocessing scripts to drop a unified JSONL + per-task images, and copies",
        "in the synthetic sample fixture. Output lives at",
        "`/content/drive/MyDrive/grounded_vla/data/`.",
        "",
        "**Runtime:** any (CPU is fine; no GPU needed). **Cost:** zero compute units.",
        "",
        "**Approximate runtime:** ~15 minutes for the configured slice.",
    ),
    DRIVE_MOUNT_CELL,
    *CLONE_REPO_CELLS,
    code(
        "# HF auth (only needed if you increase --limit and start hitting gated splits).",
        "try:",
        "    from google.colab import userdata",
        "    from huggingface_hub import login",
        "    login(token=userdata.get('HF_TOKEN'), add_to_git_credential=False)",
        "    print('HF auth OK')",
        "except Exception as e:",
        "    print('Skipping HF auth (probably fine):', e)",
    ),
    code(
        "import os",
        "DATA = '/content/drive/MyDrive/grounded_vla/data'",
        "os.makedirs(DATA, exist_ok=True)",
        "",
        "# ScienceQA (test split, image-bearing rows). 500 tasks ~= 12 minutes.",
        "!python scripts/prepare_scienceqa.py --split test --out-dir {DATA}/scienceqa --limit 500",
        "!ls {DATA}/scienceqa && wc -l {DATA}/scienceqa/test.jsonl",
    ),
    code(
        "# Synthetic sample (in-repo fixture; copy into the Drive data folder).",
        "import shutil",
        "from pathlib import Path",
        "src = Path('data/samples')",
        "dst = Path(DATA) / 'synthetic_sample'",
        "dst.mkdir(parents=True, exist_ok=True)",
        "shutil.copytree(src / 'images', dst / 'images', dirs_exist_ok=True)",
        "shutil.copy(src / 'synthetic_sample.jsonl', dst / 'synthetic_sample.jsonl')",
        "print('synthetic sample @', dst)",
    ),
    code(
        "import subprocess",
        "subprocess.run(['du', '-sh', DATA])",
        "subprocess.run(['ls', '-R', DATA])",
    ),
    md(
        "## Done",
        "",
        "Data is now on Drive. Move on to `03_run_eval.ipynb`. If you ever want a",
        "larger evaluation slice, increase `--limit` and re-run just the relevant cell.",
    ),
]


# ---------------------------------------------------------------------------
# Notebook 03: Eval (smoke + baselines + ORA)
# ---------------------------------------------------------------------------


nb03 = [
    md(
        "# 03: Smoke Test + Baselines + ORA",
        "",
        "Workhorse notebook. Runs `grounded-vla smoke`, then evaluates the three agents",
        "(ReAct + Mistral, single-shot LLaVA, ORA + LLaVA) across ScienceQA, the",
        "synthetic sample, and Mind2Web. All artifacts go to",
        "`/content/drive/MyDrive/grounded_vla/runs/<agent>__<dataset>/`.",
        "",
        "**Runtime:** A100 GPU **strongly recommended** (Pro+ default).",
        "",
        "**Estimated wallclock:** ~60-90 min on A100 for the full sweep.",
        "**Estimated compute units:** ~15-20 (well within Pro+'s 500/month).",
        "",
        "## Pro+ tip: background execution",
        "",
        "Tools → Settings → check **'Run after disconnecting'**. With this on, you can",
        "close the tab; Colab keeps the runtime alive for up to 24 hours and the",
        "checkpointed `EvalRunner` will keep writing per-task results to Drive. Re-open",
        "the notebook later and the results will be waiting in Drive.",
        "",
        "If a session does die mid-sweep, just re-run — `resume=True` skips already-",
        "completed tasks via the trajectory JSONs already on Drive.",
    ),
    DRIVE_MOUNT_CELL,
    GPU_CHECK_CELL,
    *CLONE_REPO_CELLS,
    code(
        "# Point HF cache at the Drive-mounted weights so transformers loads from there.",
        "import os",
        "WEIGHTS = '/content/drive/MyDrive/grounded_vla/hf_cache'",
        "DATA    = '/content/drive/MyDrive/grounded_vla/data'",
        "os.environ['HF_HOME'] = WEIGHTS",
        "os.environ['TRANSFORMERS_CACHE'] = WEIGHTS",
        "os.environ['TRANSFORMERS_OFFLINE'] = '1'",
        "os.environ['HF_DATASETS_OFFLINE'] = '1'",
        "assert os.path.isdir(WEIGHTS), 'run notebook 01 first'",
        "assert os.path.isdir(DATA),    'run notebook 02 first'",
    ),
    code(
        "# Smoke test: pure-mock pipeline. Should print 'smoke ok' in <5 seconds.",
        "!grounded-vla smoke",
    ),
    code(
        "# Build the three agents. On A100 we can use the default GenerationConfig",
        "# without any concessions to memory.",
        "from grounded_vla.agents import ORAAgent, ReActAgent, SingleShotVLMAgent",
        "from grounded_vla.backends import make_backend",
        "from grounded_vla.backends.base import GenerationConfig",
        "",
        "WEIGHTS = '/content/drive/MyDrive/grounded_vla/hf_cache'",
        "",
        "def build_react():",
        "    backend = make_backend({",
        "        'kind': 'mistral',",
        "        'model_id': f'{WEIGHTS}/Mistral-7B-Instruct-v0.2',",
        "        'device': 'cuda',",
        "        'quantize': '4bit',",
        "    })",
        "    return ReActAgent(backend, GenerationConfig(max_new_tokens=384, temperature=0.1))",
        "",
        "def build_single_shot():",
        "    backend = make_backend({",
        "        'kind': 'llava',",
        "        'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',",
        "        'hf_cache_dir': WEIGHTS,",
        "        'device': 'cuda',",
        "        'quantize': '4bit',",
        "    })",
        "    return SingleShotVLMAgent(backend, GenerationConfig(max_new_tokens=256, temperature=0.0))",
        "",
        "def build_ora():",
        "    backend = make_backend({",
        "        'kind': 'llava',",
        "        'model_id': 'llava-hf/llava-v1.6-mistral-7b-hf',",
        "        'hf_cache_dir': WEIGHTS,",
        "        'device': 'cuda',",
        "        'quantize': '4bit',",
        "    })",
        "    return ORAAgent(backend, GenerationConfig(max_new_tokens=384, temperature=0.1))",
    ),
    code(
        "# Build the three datasets pointing at the Drive-mounted data.",
        "from grounded_vla.data import make_dataset",
        "DATA = '/content/drive/MyDrive/grounded_vla/data'",
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
        "# Generic eval runner with checkpointing -> Drive. Resumable across sessions.",
        "from grounded_vla.eval import EvalRunner",
        "from pathlib import Path",
        "",
        "RUNS = Path('/content/drive/MyDrive/grounded_vla/runs')",
        "RUNS.mkdir(parents=True, exist_ok=True)",
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
        "Text-only. ~1-2 sec/task on A100 with 4-bit Mistral. ~3 minutes total.",
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
        "VLM, one call per task. Tests H1 (visual grounding helps).",
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
        "This is the single longest cell in the project (~30-50 min on A100).",
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
        "for d in sorted(Path('/content/drive/MyDrive/grounded_vla/runs').iterdir()):",
        "    s = json.loads((d / 'summary.json').read_text())",
        "    rows.append((d.name, s['n_tasks'], s['task_completion_rate'], s['mean_steps'], s['error_breakdown']))",
        "for r in rows:",
        "    print(f'{r[0]:30s}  n={r[1]:3d}  success={r[2]:.3f}  steps={r[3]:.2f}  errors={r[4]}')",
    ),
    md(
        "## Done",
        "",
        "Results live at `/content/drive/MyDrive/grounded_vla/runs/`. The H1 and H2",
        "comparisons drop straight out of the table above:",
        "",
        "- **H1 (vision helps):** compare `react_mistral` vs `single_shot_llava` per dataset.",
        "- **H2 (loop helps):** compare `single_shot_llava` vs `ora_llava` per dataset.",
        "",
        "Move on to `04_lora_finetune.ipynb` for the H3 stretch goal.",
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
        "corpus. Configured for **A100 (40 GB)** — full `max_seq_len=1024` and the",
        "default batch sizes. If Pro+ assigns you a T4 instead, reduce `max_seq_len`",
        "to 768 and enable `gradient_checkpointing` (commented hint in the train cell).",
        "",
        "**Estimated wallclock:** ~20-30 min on A100 (3 epochs, 200 examples).",
        "**Estimated compute units:** ~6-7 on A100.",
        "",
        "## Pro+ tip",
        "",
        "Enable **'Run after disconnecting'** (Tools → Settings) and you can fire",
        "this off, close the laptop, and come back to a finished adapter.",
    ),
    DRIVE_MOUNT_CELL,
    GPU_CHECK_CELL,
    *CLONE_REPO_CELLS,
    code(
        "import os",
        "WEIGHTS = '/content/drive/MyDrive/grounded_vla/hf_cache'",
        "DATA    = '/content/drive/MyDrive/grounded_vla/data'",
        "os.environ['HF_HOME'] = WEIGHTS",
        "os.environ['TRANSFORMERS_CACHE'] = WEIGHTS",
        "os.environ['TRANSFORMERS_OFFLINE'] = '1'",
    ),
    code(
        "# Train. The synthetic_sample fixture is tiny (3 examples) and only useful",
        "# as a pipeline check; for real H3 numbers, swap to your real synthetic.jsonl",
        "# (~200 reviewed examples). Edit JSONL/IMGS below.",
        "from grounded_vla.lora import train_lora, LoRAConfig",
        "from pathlib import Path",
        "",
        "JSONL = f'{DATA}/synthetic_sample/synthetic_sample.jsonl'",
        "IMGS  = f'{DATA}/synthetic_sample'",
        "OUT   = Path('/content/drive/MyDrive/grounded_vla/checkpoints/llava-lora-r1')",
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
        "    max_seq_len=1024,  # A100 can handle full length; drop to 768 on T4",
        ")",
        "# T4 fallback: uncomment if Pro+ gave you a T4 instead of A100",
        "# cfg.max_seq_len = 768",
        "# from grounded_vla.lora import LoRAConfig as _C  # (gradient_checkpointing hook is in lora.py if needed)",
        "",
        "train_lora(jsonl_path=JSONL, images_dir=IMGS, output_dir=OUT, config=cfg)",
    ),
    code(
        "# Verify adapter saved.",
        "import os",
        "for f in sorted(os.listdir('/content/drive/MyDrive/grounded_vla/checkpoints/llava-lora-r1')):",
        "    print(f)",
    ),
    md(
        "## Done",
        "",
        "Adapter is on Drive. Move on to `05_eval_with_lora.ipynb` for the H3 ablation.",
    ),
]


# ---------------------------------------------------------------------------
# Notebook 05: Eval with LoRA adapter
# ---------------------------------------------------------------------------


nb05 = [
    md(
        "# 05: Evaluate LLaVA + LoRA (H3 Ablation)",
        "",
        "Loads the LoRA adapter trained in notebook 04 onto the LLaVA backbone and",
        "runs ORA on the synthetic held-out set + a Mind2Web slice. Prints a base-vs-",
        "adapter delta table for the H3 row of the report.",
        "",
        "**Runtime:** A100 GPU. **Wallclock:** ~30-40 min. **Compute units:** ~7-9.",
    ),
    DRIVE_MOUNT_CELL,
    GPU_CHECK_CELL,
    *CLONE_REPO_CELLS,
    code(
        "import os",
        "WEIGHTS = '/content/drive/MyDrive/grounded_vla/hf_cache'",
        "DATA    = '/content/drive/MyDrive/grounded_vla/data'",
        "ADAPTER = '/content/drive/MyDrive/grounded_vla/checkpoints/llava-lora-r1'",
        "os.environ['HF_HOME'] = WEIGHTS",
        "os.environ['TRANSFORMERS_OFFLINE'] = '1'",
    ),
    code(
        "# Load LLaVA + adapter via a thin subclass that injects PEFT after load.",
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
        "agent = ORAAgent(backend, GenerationConfig(max_new_tokens=384, temperature=0.1))",
        "",
        "RUNS = Path('/content/drive/MyDrive/grounded_vla/runs')",
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
        "synth = make_dataset({",
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
        "_ = run('synthetic',  synth)",
        "_ = run('scienceqa',  sqa)",
        "backend.close()",
    ),
    code(
        "# Side-by-side vs the non-LoRA ORA runs from notebook 03.",
        "import json, glob",
        "base_runs = sorted(glob.glob('/content/drive/MyDrive/grounded_vla/runs/ora_llava__*'))",
        "lora_runs = sorted(glob.glob('/content/drive/MyDrive/grounded_vla/runs/ora_lora__*'))",
        "",
        "def load(p):",
        "    return json.loads(open(f'{p}/summary.json').read())",
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
        "## Interpreting the H3 result",
        "",
        "- **Positive delta on synthetic, smaller on Mind2Web:** LoRA closed the in-domain",
        "  gap; cross-domain transfer is partial. Useful talking point in the report.",
        "- **Negative delta:** likely overfitting on a tiny synthetic set. Either grow the",
        "  synthetic corpus to ~200 reviewed examples, or shorten training (1 epoch).",
        "- **No change:** LoRA rank may be too small (try `r=32`) or the action format in",
        "  the synthetic dataset doesn't match the agent's prompt template.",
        "",
        "Either way, the result goes into the H3 row of the results table.",
    ),
]


# ---------------------------------------------------------------------------
# Build them all
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    write_notebook("01_setup_drive_and_weights.ipynb", nb01)
    write_notebook("02_setup_data.ipynb", nb02)
    write_notebook("03_run_eval.ipynb", nb03)
    write_notebook("04_lora_finetune.ipynb", nb04)
    write_notebook("05_eval_with_lora.ipynb", nb05)
