# Running grounded_vla on Colab Pro+

End-to-end recipe for running this project on **Google Colab Pro+**
($50/mo, A100 access, background execution). Five notebooks live in this
folder; this guide explains the order, the one-time prep, and the gotchas.

If you'd rather run for free, see `kaggle/KAGGLE.md` instead.

## TL;DR

```
01_setup_drive_and_weights.ipynb  →  Drive: /MyDrive/grounded_vla/hf_cache    (one-time)
02_setup_data.ipynb               →  Drive: /MyDrive/grounded_vla/data        (one-time)
03_run_eval.ipynb                 →  Drive: /MyDrive/grounded_vla/runs/...
04_lora_finetune.ipynb            →  Drive: /MyDrive/grounded_vla/checkpoints/llava-lora-r1
05_eval_with_lora.ipynb           →  Drive: /MyDrive/grounded_vla/runs/ora_lora__*
```

Everything persists in your Google Drive. Different teammates can mount
the same Drive folder by sharing it via Drive's normal share UI.

## One-time prep

1. **Push the repo to GitHub** (or anywhere `git clone` reaches). The
   notebooks reference it as `https://github.com/<your-org>/grounded_vla.git`
   — open each `.ipynb` in Colab and find/replace `<your-org>` with your handle.

2. **Accept the model licenses** on a HuggingFace account:
   - https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
   - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

3. **Create a HuggingFace access token**: huggingface.co → Settings →
   Access Tokens → `Create new token` → "Read" role.

4. **Add the token as a Colab Secret**: open any Colab notebook → left
   sidebar → key icon → `Add new secret` → name `HF_TOKEN`, value =
   your token → toggle **Notebook access** to ON for each notebook
   that needs it (notebooks 01 and 02). The notebooks read it via
   `from google.colab import userdata; userdata.get('HF_TOKEN')`.

5. **Verify your Drive has ~30 GB free** (model weights take ~28 GB).

## Selecting the right runtime

For each notebook: **Runtime → Change runtime type**:

| Notebook | GPU | Why |
|---|---|---|
| 01 (weights) | A100 or T4 | Just downloads; GPU choice doesn't matter |
| 02 (data) | None / CPU | No GPU work |
| 03 (eval) | **A100 (preferred), V100/T4 fallback** | LLaVA inference 3-5x faster on A100 |
| 04 (LoRA) | **A100** | Default config sized for 40 GB; T4 needs cell tweak |
| 05 (LoRA eval) | **A100** | Same shape as notebook 03 |

If A100 isn't available when you select it, Colab assigns whatever's free
(usually L4 or T4). The notebooks detect this and continue, but notebook
04's defaults assume A100 — see "T4 fallback" below.

## Pro+ killer feature: background execution

In the notebook menu, **Tools → Settings → Site → check 'Run after
disconnecting'**. With this on:

- Close the browser tab.
- Close your laptop.
- Come back hours later — the runtime stays alive for up to 24 hours and
  your eval loop keeps running.

Combined with `EvalRunner(checkpoint_every=5, resume=True)` (already
wired into the notebooks), this means you can fire the full sweep before
bed and check it in the morning. It is the single biggest reason Pro+ is
worth $50 over free Colab.

## Running the notebooks

### 1. Upload to Colab

For each `.ipynb`: Colab → `File → Upload notebook`. Or push the repo to
GitHub and use `File → Open notebook → GitHub`.

### 2. `01_setup_drive_and_weights.ipynb`

Run all cells. First cell triggers the Drive OAuth pop-up; allow access.
Downloads ~28 GB to `/content/drive/MyDrive/grounded_vla/hf_cache`. Takes
~10 minutes on A100/T4; bottleneck is HuggingFace's bandwidth, not the
GPU.

You will not need to run this again unless model versions change.

### 3. `02_setup_data.ipynb`

CPU runtime is fine. Pulls Mind2Web + ScienceQA from HuggingFace and runs
the `prepare_*.py` scripts in the repo. ~15 minutes. Output goes to
`/content/drive/MyDrive/grounded_vla/data`.

### 4. `03_run_eval.ipynb` — main workhorse

Switch to **A100 GPU** runtime. Enable **'Run after disconnecting'**.

Run cells top to bottom. The notebook:

1. Mounts Drive
2. Verifies the runtime is A100 (warns if not)
3. Smoke-tests the pipeline with the mock backend (~5 sec)
4. Runs ReAct + Mistral on all three benchmarks (~3-5 min)
5. Runs single-shot LLaVA on all three (~10-15 min)
6. Runs ORA + LLaVA on all three (~30-50 min)
7. Prints the aggregate results table

Total: ~60-90 minutes on A100 for the configured slice (50 ScienceQA + 30
Mind2Web + 3 synthetic). Per-task trajectories stream to Drive in real
time, so a session timeout is recoverable.

### 5. `04_lora_finetune.ipynb` — H3 stretch goal

Switch to **A100** runtime. The default `LoRAConfig` is sized for 40 GB
A100: `max_seq_len=1024`, `gradient_accumulation_steps=8`, three epochs.

If Pro+ assigns you a T4 instead of A100 (it happens), uncomment the T4
fallback line in the train cell to drop `max_seq_len` to 768. Otherwise
expect OOM.

For meaningful H3 numbers, replace the synthetic sample (3-example pipeline
test) with your real `synthetic.jsonl` (~200 reviewed examples). Edit
`JSONL` and `IMGS` paths in the train cell.

Wallclock on A100: ~20-30 min for 3 epochs / 200 examples.

### 6. `05_eval_with_lora.ipynb` — H3 ablation

A100 runtime. Loads LLaVA + the LoRA adapter via a `LLaVAWithLoRA`
subclass that injects PEFT after model load. Re-runs ORA on the synthetic
held-out set + a Mind2Web slice and prints the base-vs-adapter delta
table for the report.

## Compute unit budget

Colab Pro+ gives you ~500 compute units/month. Rough usage for this project:

| Step | A100 hr | Units |
|---|---|---|
| Notebook 01 (weights) | 0.2 | 3 |
| Notebook 02 (data) | 0 (CPU) | 0 |
| Notebook 03 (full sweep) | 1.5 | 20 |
| Notebook 04 (LoRA) | 0.5 | 7 |
| Notebook 05 (ablation) | 0.7 | 9 |
| Iteration / re-runs buffer | 1.0 | 13 |
| **Total** | **~3.9 hr** | **~52 units** |

You'll use roughly **10% of your monthly budget** for one full pass —
plenty of headroom for re-runs and additional ablations. If you blow
through faster than expected, drop A100 to V100 for the cheap cells.

## Drive layout

After the full pipeline, your Drive contains:

```
/MyDrive/grounded_vla/
├── hf_cache/
│   ├── llava-v1.6-mistral-7b-hf/    (~13 GB)
│   └── Mistral-7B-Instruct-v0.2/    (~14 GB)
├── data/
│   ├── scienceqa/
│   │   ├── test.jsonl
│   │   └── images/                  (200 PNGs)
│   ├── mind2web/
│   │   ├── test_task.jsonl
│   │   └── images/                  (100 PNGs)
│   └── synthetic_sample/
├── checkpoints/
│   └── llava-lora-r1/               (LoRA adapter, ~200 MB)
└── runs/
    ├── react_mistral__scienceqa/    (summary.json + trajectories/)
    ├── react_mistral__synthetic/
    ├── react_mistral__mind2web/
    ├── single_shot_llava__scienceqa/
    ├── ...
    └── ora_lora__mind2web/
```

Total Drive usage: ~30 GB, well under the free 15 GB tier? No — you'll
need at least Drive 100 GB ($1.99/mo) or Google One. Worth budgeting for.

## Sharing with teammates

The cleanest pattern:

1. One person owns the Drive folder. They run notebooks 01 and 02 once.
2. They share the `/MyDrive/grounded_vla/` folder with the team
   (Drive → right-click → Share → Editor access).
3. Each teammate adds the shared folder to their own Drive
   (Drive → Shared with me → right-click → Add shortcut to Drive).
4. Each teammate's Colab notebook now sees the same data + weights at
   `/content/drive/MyDrive/grounded_vla/`.
5. Each teammate runs notebook 03 / 04 / 05 against the shared inputs;
   their results land in the same `runs/` folder under separate
   subfolder names.

A subtlety: writes to a shared folder are slower than writes to your own
Drive. For latency-sensitive trajectory writes (notebooks 03 and 05),
consider having each teammate write to their *own* `runs/<username>/`
subdirectory.

## Gotchas

- **A100 isn't always available.** Pro+ prioritizes you, but during peak
  hours you may land on L4 or T4. The notebooks detect this and continue;
  you just lose the speed advantage. If you really need A100, try
  switching runtime types — sometimes there's free A100 capacity in a
  region the auto-allocator didn't try.

- **Drive sync is asynchronous.** When your eval finishes, files may take
  a few minutes to appear in Drive's web UI. If you don't see them
  immediately, give it a minute before panicking.

- **Pip installs don't persist between runtime restarts.** Every notebook
  re-installs the repo via `pip install -e .[gpu,data]`. ~90 seconds.
  Live with it, or save a custom Colab base image (Pro+ feature).

- **Restarting the runtime** is sometimes needed after a heavy pip install
  (CUDA-related package updates can require it). Runtime → Restart
  session, then re-run from the Drive-mount cell.

- **`HF_TOKEN` notebook access toggle.** Each notebook needs to be
  *individually* granted access to your Colab Secret. Forgetting this is
  the most common "why doesn't it work" early on.

- **'Run after disconnecting' has limits.** Pro+ keeps the runtime alive
  for up to 24 hours after disconnect, but if you idle (no cell output)
  for more than ~90 minutes, the heartbeat fails and the runtime dies.
  Workaround: include a long-running cell (the eval loop) so the
  heartbeat is satisfied by the actual work.

## Regenerating the notebooks

Cell content lives in `_build_notebooks.py` so changes are diff-friendly.
Edit that script, then:

```bash
python colab/_build_notebooks.py
```

This rewrites all five notebooks in place.
