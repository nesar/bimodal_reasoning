# Runbook: LoRA Fine-Tuning of GPT-OSS on Galaxy Spectra (LCRC)

Practical guide for running everything without Claude Code.
All commands use absolute paths and can be copy-pasted directly.

---

## Quick Reference

```bash
# Set these ONCE per session (copy-paste this block)
export TMPDIR=/tmp
export HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/lcrc/project/solitons/nramachandra/lm_eval_pkg:${PYTHONPATH:-}
PYTHON=/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python
BASE=/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/bimodal_reasoning
cd $BASE
```

---

## 1. Environment

### 1.1 Two Python environments

| Env | Python path | Torch | Use for |
|-----|-------------|-------|---------|
| **bimodal** | `/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python` | 2.6.0+cu124 | Training, eval, plotting |
| **autoresearch** | `/lcrc/project/solitons/nramachandra/envs/autoresearch/bin/python` | 2.9.1+cu128 | Karpathy's autoresearch (from-scratch GPT) |

The bimodal env also has: transformers 5.5, peft 0.18.1, deepspeed 0.18.9, matplotlib, scipy, h5py, scikit-learn.

**lm-eval 0.4.11** is installed separately at `/lcrc/project/solitons/nramachandra/lm_eval_pkg/`. Access via `PYTHONPATH`:
```bash
export PYTHONPATH=/lcrc/project/solitons/nramachandra/lm_eval_pkg:${PYTHONPATH:-}
```

### 1.2 Filesystem constraints

| Fileset | Quota | Typical usage | Notes |
|---------|-------|---------------|-------|
| Home (`~`) | 100 GB | ~90 GB | Never pip install here |
| cosmo_ai | 75 TB | ~66 TB | Main project space; fills up fast with model weights |
| solitons | 1 TB | ~50 GB | Overflow for envs/packages |

**Always set `TMPDIR=/tmp`** — otherwise temp files go to cosmo_ai and fill the quota.

Check quota: `mmlsquota --block-size auto -j cosmo_ai fs0`

### 1.3 GPU inventory

8x NVIDIA A100-SXM4-40GB on gpu1.lcrc.anl.gov (1 TB RAM).
```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
```

---

## 2. Datasets

### 2.1 Two dataset formats

| Dataset | Path | Output format | Valid at eval? |
|---------|------|---------------|----------------|
| **Original** | `data/datasets/structured_verbalization/text2text.json` | `Redshift: z = 0.3510\nStellar mass: ...` | 11% (22/200) |
| **Compact** (recommended) | `data/datasets/structured_verbalization_compact/text2text.json` | `[z=0.3510\|mass=11.18\|age=10.4\|Z=0.461]` | **100% (200/200)** |

**Always use the compact dataset.** The original format has spectra too long for 512-token training window, causing the model to generate garbage at inference.

### 2.2 Regenerate the compact dataset

If you need to regenerate (e.g., changed tokenization):
```bash
$PYTHON tokenization/regen_compact.py \
    --data-path /lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/data/Tokyo/Data/sdss_galaxy_spec.hdf5 \
    --output-dir data/datasets/structured_verbalization_compact \
    --block-size 512 \
    --tokenizer-name openai/gpt-oss-20b
```

This trims spectra so `input + output` always fits in 512 tokens.

### 2.3 Tokenization strategies

Six strategies in `tokenization/spec_tokenizer.py`: `digit_base10`, `digit_base16`, `log_scaled`, `patch_mean`, `wavelength_value`, `structured_verbalization`. The compact dataset uses `digit_base10` inside `structured_verbalization` with spectrum trimming.

---

## 3. Training

### 3.1 GPT-OSS-20B (~50 min on 2 GPUs)

```bash
$PYTHON training/finetune_hf.py \
    --model_name_or_path openai/gpt-oss-20b \
    --dataset_path data/datasets/structured_verbalization_compact/text2text.json \
    --output_dir output_models/gpt-oss-20b_compact \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --lora_r 8 \
    --logging_steps 20 \
    --save_steps 500
```

Results: loss 1.69 → 1.25, adapter 16 MB. Uses ~7 GB on 2 GPUs.

### 3.2 GPT-OSS-120B (~1.8 hrs on 8 GPUs)

```bash
$PYTHON training/finetune_hf.py \
    --model_name_or_path openai/gpt-oss-120b \
    --dataset_path data/datasets/structured_verbalization_compact/text2text.json \
    --output_dir output_models/gpt-oss-120b_compact \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --lora_r 8 \
    --logging_steps 10 \
    --save_steps 500
```

Results: loss 1.69 → 1.09, adapter 23 MB. Uses 21-38 GB across 8 GPUs.

**Critical:** Load from `openai/gpt-oss-120b` directly. Do NOT use any pre-converted bf16 checkpoint (MoE weight naming bug).

### 3.3 Short experiment run (~10 min, for hyperparameter search)

```bash
$PYTHON experiments/run_experiment.py \
    --output_dir /tmp/exp_test \
    --lr 1e-4 \
    --lora_r 8 \
    --max_steps 100 \
    --eval_samples 50
```

Prints `mae:`, `train_loss:`, `total_seconds:` at the end. Used by the autoresearch loop.

---

## 4. Evaluation

### 4.1 Redshift MAE (Criteria-1: scientific accuracy)

```bash
$PYTHON eval/redshift_eval_peft.py \
    --base_model openai/gpt-oss-20b \
    --adapter_path output_models/gpt-oss-20b_compact \
    --dataset data/datasets/structured_verbalization_compact/text2text.json \
    --output_dir plots/eval_20b_compact \
    --num_test 200 \
    --label "GPT-OSS-20B compact"
```

Outputs:
- `plots/eval_20b_compact/metrics.json` — MAE, median AE, outlier fraction
- `plots/eval_20b_compact/redshift_scatter.png` — true vs predicted plot
- `plots/eval_20b_compact/raw_predictions.jsonl` — per-instance predictions

**For 120B**, same command but use `openai/gpt-oss-120b` and the 120B adapter path.

**For base model (no adapter)**, omit `--adapter_path`:
```bash
$PYTHON eval/redshift_eval_peft.py \
    --base_model openai/gpt-oss-20b \
    --dataset data/datasets/structured_verbalization_compact/text2text.json \
    --output_dir plots/eval_20b_base \
    --num_test 100
```

**Key results (2026-04-08):**

| Model | MAE | Valid | Outliers (>0.1) |
|-------|-----|-------|-----------------|
| 20B compact (full 2-epoch) | **0.071** | 200/200 | 22% |
| 20B original format | 0.286 | 139/200 | 89% |

### 4.2 LM Eval Harness (Criteria-2: benchmark retention)

**Base model:**
```bash
$PYTHON -m lm_eval \
    --model hf \
    --model_args "pretrained=openai/gpt-oss-20b,trust_remote_code=True,dtype=bfloat16,device_map=auto,attn_implementation=eager" \
    --tasks mmlu_college_physics,mmlu_high_school_physics,mmlu_astronomy,leaderboard_gpqa,leaderboard_bbh \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path eval_results/base_20b
```

**Fine-tuned model (with PEFT adapter):**
```bash
$PYTHON -m lm_eval \
    --model hf \
    --model_args "pretrained=openai/gpt-oss-20b,peft=output_models/gpt-oss-20b_compact,trust_remote_code=True,dtype=bfloat16,device_map=auto,attn_implementation=eager" \
    --tasks mmlu_college_physics,mmlu_high_school_physics,mmlu_astronomy,leaderboard_gpqa,leaderboard_bbh \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path eval_results/ft_20b_compact
```

**Notes:**
- Uses `--num_fewshot 0` (some tasks have <5 few-shot examples available)
- `astro_mlab_araa_mcq_gemini15` (AstroBench) is NOT in standard lm-eval; skip for now
- Results appear in `eval_results/*/results_*.json`

---

## 5. Autoresearch Loop (Hyperparameter Search)

This runs ~10 min experiments varying fine-tuning hyperparameters,
tracking MAE and generating a Karpathy-style money plot.

### 5.1 Run the full experiment loop

```bash
nohup bash experiments/run_loop.sh > experiments/autoresearch_runs/loop.log 2>&1 &
```

Runs experiments 2-12 sequentially (~2 hours total). Each varies one hyperparameter
from the baseline. Results in `experiments/autoresearch_runs/results.tsv`.

### 5.2 Results (2026-04-08, 13 experiments)

| # | Change | MAE | Status |
|---|--------|-----|--------|
| 0 | baseline: lr=1e-4 r=8 100steps | 0.111 | keep |
| 1 | lr → 5e-5 | 0.255 | discard (underfit) |
| 2 | lr → 5e-4 | 0.111 | keep (marginal) |
| 3 | r=16, α=32 | 0.088 | keep |
| 4 | r=4, α=8 | 0.133 | discard |
| 5 | grad_accum → 16 | 0.122 | discard |
| 6 | grad_accum → 4 | 0.083 | keep |
| 7 | **200 steps** | **0.066** | keep |
| 8 | warmup → 0.15 | 0.093 | discard |
| 9 | dropout → 0.1 | 0.093 | discard |
| 10 | lr → 2e-4 | 0.095 | discard |
| 11 | **300 steps** | **0.056** | **keep (best)** |
| 12 | r=16 + 200 steps | 0.068 | discard |

**Key insight:** More training steps is the strongest lever (0.111 → 0.056).
Higher LoRA rank and smaller batch help moderately.

### 5.3 Generate the money plot

```bash
$PYTHON analysis/plot_autoresearch_money.py \
    --results experiments/autoresearch_runs/results.tsv \
    --output plots/autoresearch_money.png
```

### 5.4 Run a single custom experiment

```bash
$PYTHON experiments/run_experiment.py \
    --output_dir /tmp/exp_custom \
    --lr 1e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --max_steps 300 \
    --grad_accum 4 \
    --eval_samples 50
```

---

## 6. Plots

### 6.1 Plot scripts

| Script | What it generates | Command |
|--------|-------------------|---------|
| `analysis/plot_training_run.py` | Loss curves, LR schedule, grad norms | `$PYTHON analysis/plot_training_run.py --run_dir output_models/<name>` |
| `analysis/plot_fig1.py` | Paper Figure 1: redshift scatter + benchmarks | See below |
| `analysis/plot_autoresearch_money.py` | Karpathy-style experiment progress | `$PYTHON analysis/plot_autoresearch_money.py` |
| `analysis/plot_money.py` | Multi-panel summary (loss + scatter + table) | `$PYTHON analysis/plot_money.py` |

### 6.2 Figure 1 (paper style)

```bash
$PYTHON analysis/plot_fig1.py \
    --eval_dirs \
        "20B (original)=plots/eval_20b_original" \
        "20B (compact)=plots/eval_20b_compact" \
    --benchmark_json overnight_results/latest/benchmark_comparison.json \
    --output plots/fig1.png
```

Top row: redshift scatter (true vs predicted) with KDE contours and MAE.
Bottom row: benchmark retention bar chart (percent change after fine-tuning).

---

## 7. Overnight Job

Run everything unattended:
```bash
cd $BASE
nohup bash run_overnight.sh > overnight.log 2>&1 &
# Then optionally chain benchmarks:
nohup bash run_benchmarks.sh $! > benchmark.log 2>&1 &
```

Monitor: `tail -f overnight_results/latest/STATUS`

Phases: train 20B compact → eval compact → eval original → plots → autoresearch loop.

---

## 8. What Does NOT Work

| What | Why | Workaround |
|------|-----|------------|
| DeepSpeed ZeRO-3 with 120B | MXFP4 dequant OOM across 8 ranks; torch 2.6 pins triton==3.2 | `device_map="auto"` (single process) |
| Pre-converted bf16 checkpoint | MoE expert weights get `_blocks` suffix, layers randomly init | Always load from `openai/gpt-oss-120b` directly |
| QLoRA via bitsandbytes | MXFP4 can't be overridden | Model auto-dequants to bf16 |
| `sdpa` / `flash_attention_2` | Not supported by gpt-oss | Use `attn_implementation="eager"` |
| Original output format at eval | Spectrum >500 tokens, no room for answer in 512 window | Use compact dataset |
| `lm_eval` CLI directly | Not installed in bimodal env PATH | `$PYTHON -m lm_eval` |
| `num_fewshot=5` | Some tasks have <5 examples | Use `--num_fewshot 0` |
| `pip install` on cosmo_ai | Fileset near quota | `TMPDIR=/tmp HOME=/tmp pip install --no-cache-dir --target=/lcrc/project/solitons/...` |

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No space left on device` | `export TMPDIR=/tmp`; check quota with `mmlsquota`; clean `Projects/tmp/` |
| `torch_dtype is deprecated` | Use `dtype=` instead (transformers 5.x) |
| Wrong python/deepspeed picked up | Use full paths: `/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python` |
| `Python.h not found` (autoresearch) | `export C_INCLUDE_PATH=/home/nramachandra/anaconda3/envs/eval-harness/include/python3.10` |
| GPU OOM in autoresearch | Set `DEVICE_BATCH_SIZE=64` in `train.py` |
| Autoresearch crash loop | Script has max 3 consecutive failures breaker |

---

## 10. File Map

```
bimodal_reasoning/
├── RUNBOOK.md                          # This file
├── CLAUDE.md                           # Claude Code project instructions
├── TODO.md                             # Design roadmap
│
├── data/datasets/
│   ├── structured_verbalization/       # Original format (don't use for eval)
│   └── structured_verbalization_compact/ # Compact format (USE THIS)
│
├── tokenization/
│   ├── spec_tokenizer.py              # All tokenization strategies
│   ├── verbalize.py                   # Output format (verbose + compact)
│   └── regen_compact.py               # Regenerate compact dataset
│
├── training/
│   ├── finetune_hf.py                 # Main training script (20B + 120B)
│   └── convert_to_bf16.py            # DO NOT USE (bf16 conversion has bugs)
│
├── eval/
│   ├── redshift_eval_peft.py          # Redshift MAE eval (USE THIS)
│   ├── redshift_eval.py               # Old eval (no PEFT support)
│   └── lm_harness_eval.sh            # Benchmark eval wrapper
│
├── experiments/
│   ├── run_experiment.py              # Single experiment (train + eval, ~10 min)
│   ├── run_loop.sh                    # Autoresearch hyperparameter loop
│   ├── autoresearch_runs/results.tsv  # Experiment log
│   └── autoresearch_stub.py           # Original objective function design
│
├── analysis/
│   ├── plots.py                       # Style + reusable plot functions
│   ├── plot_training_run.py           # Loss/LR/gradient plots from trainer state
│   ├── plot_fig1.py                   # Paper Figure 1 (scatter + benchmarks)
│   ├── plot_autoresearch_money.py     # Karpathy-style money plot
│   └── plot_money.py                  # Multi-panel summary
│
├── output_models/
│   ├── gpt-oss-20b_structured/        # 20B adapter (original format)
│   ├── gpt-oss-20b_compact/           # 20B adapter (compact format)
│   └── gpt-oss-120b_structured/       # 120B adapter (original format)
│
├── plots/                              # All generated plots
├── run_overnight.sh                   # Unattended overnight job
├── run_benchmarks.sh                  # LM eval harness benchmarks
└── configs/ds_config_zero3.json       # DeepSpeed config (doesn't work, kept for reference)
```
