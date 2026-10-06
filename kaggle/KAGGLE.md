# Running grounded_vla on Kaggle

End-to-end recipe for running this project on Kaggle's free GPU tier
(2× T4, 30 hours/week). Five notebooks live in this folder; this guide
explains the order, the one-time prep, and the gotchas.

## TL;DR

```
01_setup_weights.ipynb   →  produces Kaggle Dataset "gvla-weights"   (one-time)
02_setup_data.ipynb      →  produces Kaggle Dataset "gvla-data"      (one-time)
03_run_eval.ipynb        →  produces Kaggle Dataset "gvla-runs-YYYY-MM-DD"
04_lora_finetune.ipynb   →  produces Kaggle Dataset "gvla-lora-r1"    (H3, optional)
05_eval_with_lora.ipynb  →  produces Kaggle Dataset "gvla-runs-lora-YYYY-MM-DD"
```

Each notebook is roughly self-contained; they share state through the
Kaggle Dataset feature, not through shared filesystem or runtime.

## One-time prep (do this before the first notebook)

1. **Push the repo to GitHub** (or anywhere you can `git clone` from Kaggle).
   The notebooks reference the repo as `https://github.com/<your-org>/grounded_vla.git`
   — open each `.ipynb`, `Find & replace` `<your-org>` with your handle.

2. **Accept the model licenses on HuggingFace** (use a personal HF account):
   - https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
   - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

   You'll see "You have been granted access" on each page once approved.

3. **Create a HuggingFace access token**: Account → Settings → Access Tokens
   → `Create new token` → "Read" role. Copy it.

4. **Add the token to Kaggle as a Secret**: open any Kaggle notebook → right
   sidebar → `Add-ons → Secrets → Add a new secret` → label it `HF_TOKEN` →
   paste the value. The token is then accessible in all your notebooks via
   `kaggle_secrets.UserSecretsClient().get_secret('HF_TOKEN')`.

5. **Verify your weekly GPU quota**: Profile → Settings → Account → "GPU
   quota". Free tier gives you 30 hours/week; this project uses ~12–15.

## Running the notebooks

### 1. Upload the notebooks to Kaggle

For each `.ipynb` in this folder: Kaggle → `Create → New Notebook` → File
→ Import notebook → upload. Or use `kaggle kernels push` if you have the CLI.

### 2. `01_setup_weights.ipynb` — download model weights (one-time)

Settings (right panel):
- Accelerator: **GPU T4 x2**
- Persistence: **Files only**
- Internet: **On**

Run all cells. Takes ~10 minutes; downloads ~28 GB to `/kaggle/working/hf_cache`.
When it finishes:

1. Click `Save Version` → `Quick Save` → `Save & Run All`.
2. After the version is saved, open the notebook → `Output` tab →
   `New Dataset`. Name it `gvla-weights`. Visibility = Private.

You will not need to run this notebook again unless model versions change.

### 3. `02_setup_data.ipynb` — preprocess datasets (one-time)

Settings: any CPU-only runtime is fine; Internet on.

This pulls Mind2Web + ScienceQA from HuggingFace, runs the in-repo
`prepare_*.py` scripts, and copies in the synthetic sample fixture. Takes
~15 minutes.

Save Version → New Dataset → name it `gvla-data`.

### 4. `03_run_eval.ipynb` — main evaluation

Attach both datasets created above (right panel → `Add Data → Your
Datasets → gvla-weights, gvla-data`).

Settings:
- Accelerator: **GPU T4 x2**
- Internet: **On** (we don't need HF access here, but `pip install` does)

Run cells top to bottom. The notebook is checkpointed after every task
(see `EvalRunner.evaluate(checkpoint_every=5, resume=True)`), so a session
timeout just means re-running and the agent picks up where it left off.

Estimated runtime for the configured slice (50 ScienceQA + 30 Mind2Web +
3 synthetic, all three agents):
- ReAct + Mistral: ~30–45 min
- Single-shot LLaVA: ~25–35 min
- ORA + LLaVA: ~60–90 min (multi-step is the bottleneck)

Total: roughly 2–3 hours, comfortably inside one Kaggle session.

After the run, Save Version → New Dataset → name it
`gvla-runs-YYYY-MM-DD` so notebook 05 can compare against it.

### 5. `04_lora_finetune.ipynb` — H3 stretch goal

Skip this if H3 is out of scope. Otherwise:

Attach `gvla-weights` + `gvla-data`. Settings: GPU T4 x2, Internet on.

The `LoRAConfig` defaults are tuned for 2× T4 with sharded base (`max_seq_len=768`,
`per_device_batch_size=1`, `gradient_accumulation_steps=8`). For meaningful
H3 numbers, replace the synthetic sample (3 examples, pipeline-test only)
with your real `synthetic.jsonl` (~200 reviewed examples). Edit the cell
that sets `JSONL` and `IMGS`.

Save Version → New Dataset → `gvla-lora-r1`.

### 6. `05_eval_with_lora.ipynb` — H3 ablation

Attach `gvla-weights`, `gvla-data`, `gvla-lora-r1`, and the
`gvla-runs-YYYY-MM-DD` dataset from notebook 03 (for the comparison).

Loads LLaVA + the LoRA adapter, runs ORA on the synthetic held-out set
and a Mind2Web slice, and prints a base-vs-adapter delta table for the
report.

## Kaggle-specific gotchas to know about

- **`/kaggle/working` is ephemeral.** Anything not committed via "Save
  Version" disappears when the session ends. The notebooks all write to
  `/kaggle/working/runs/` and the Save Version step turns that into a
  Dataset.

- **No SSH, no `tmux`.** Long sweeps run inside notebook cells. Don't
  close the browser tab while a cell is running. If your session does
  die, re-run with `resume=True` (already wired up) — the per-task
  trajectory files act as natural resume markers.

- **Pip installs don't persist.** Every notebook re-installs the repo via
  `pip install -e .[gpu]`. ~90 seconds. Live with it.

- **Multi-GPU sharding is automatic.** Our `LLaVABackend` uses
  `device_map="auto"`, which sees both T4s on Kaggle and shards
  accordingly. No code change needed; just make sure the accelerator is
  set to `GPU T4 x2`, not `GPU P100`.

- **Disk pressure.** Kaggle's `/kaggle/working` cap is 20 GB. The notebook
  03 run writes ~50–200 MB of trajectory JSON; well under. Notebook 04's
  LoRA checkpoints are ~200 MB per epoch; also fine. If you ever blow the
  cap, the first culprit is `transformers`'s safetensors cache being
  copied instead of symlinked — keep `local_dir_use_symlinks=False` and
  point `HF_HOME` at the read-only mount, never at `/kaggle/working`.

- **Quota maths.** A full sweep + a LoRA run + an ablation costs roughly
  6 GPU-hours. You get 30 GPU-hours/week free. Plenty of headroom for
  re-runs and experiments.

- **Kaggle Datasets cap at 100 GB**, far above what we use here. Each
  Dataset can hold up to ~10k files; trajectories are well under that.

## Sharing with teammates

A Kaggle Dataset can be shared via its URL with anyone who has a Kaggle
account (Dataset → `Share`). The clean pattern:

- One person owns `gvla-weights` and `gvla-data` (re-running notebooks
  01–02 is wasteful).
- Anyone runs notebook 03 against those shared datasets to produce their
  own `gvla-runs-*` artifacts.
- One person owns `gvla-lora-r1`; everyone else mounts it for ablation.

Notebooks themselves can also be shared (`Share → Add collaborators`),
which is the easiest way to fork a teammate's setup and tweak it.

## Regenerating the notebooks

The `.ipynb` files are generated from `_build_notebooks.py` so cell
content stays diff-friendly. Edit that script, then:

```bash
python kaggle/_build_notebooks.py
```

This rewrites all five notebooks in place.
