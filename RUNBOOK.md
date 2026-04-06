# Runbook: LoRA Fine-Tuning of GPT-OSS on Galaxy Spectra (LCRC Cluster)

This document captures all decisions, failure modes, and reproducible commands
for training GPT-OSS models (20B and 120B) on SDSS galaxy spectra using LoRA
on the LCRC gpu1 node (8x NVIDIA A100-SXM4-40GB, 1 TB RAM).

---

## 1. Environment

### 1.1 Conda environment

| Component        | Version           | Path / Notes                                           |
|------------------|-------------------|--------------------------------------------------------|
| Python           | 3.11              | `/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/`   |
| PyTorch          | 2.6.0+cu124       | Compiled with CUDA 12.4                                |
| Transformers     | 5.5.0             | Note: `torch_dtype` kwarg deprecated, use `dtype`     |
| PEFT             | 0.18.1            |                                                        |
| DeepSpeed        | 0.18.9            | Installed but not usable for 120B (see Section 5)      |
| Triton           | 3.2.0             | Too old for native MXFP4; torch 2.6 pins `triton==3.2`|
| Flash Attention  | 2.8.3             | gpt-oss models only support `eager` or `flash_attention_4` |
| System CUDA      | 13.0 (driver 580) | Mismatch with torch's 12.4; set `DS_SKIP_CUDA_CHECK=1`|
| HF cache         | `/lcrc/project/cosmo_ai/nramachandra/hf_cache` |                      |

### 1.2 Filesystem quotas and workarounds

The LCRC cluster has two key constraints:

1. **Home directory** (`~`): 100 GB quota, often >90% full.
2. **cosmo_ai fileset**: 75 TB GPFS quota, ~65.7 TB used. Check with:
   ```bash
   mmlsquota --block-size auto -j cosmo_ai fs0
   ```

**Required environment variables for every command:**
```bash
export TMPDIR=/tmp                    # Avoid writing temp files to project space
export HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache
```

For pip installs, also set `HOME=/tmp` and use `--no-cache-dir`.
If cosmo_ai is full, use the solitons fileset:
`/lcrc/project/solitons/nramachandra/`.

### 1.3 GPU inventory

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
```
Expected output: 8x NVIDIA A100-SXM4-40GB (40960 MiB each).

---

## 2. Data

### 2.1 Source data
- SDSS galaxy spectra in HDF5 format
- ~9800 samples, 4556 wavelength channels
- Physical properties: redshift, stellar mass, age, metallicity

### 2.2 Text2Text dataset
- **Location:** `data/datasets/structured_verbalization/text2text.json`
- **Format:** JSON with `{"instances": [{"input": "...", "output": "..."}, ...]}`
- **Size:** 2939 instances (after filtering)
- **Content:** Each instance encodes a galaxy's physical properties + tokenized
  spectrum as the input, with the redshift value as the output.

### 2.3 Tokenization strategies
Six strategies are available in `tokenization/spec_tokenizer.py`:
- `digit_base10`, `digit_base16`, `log_scaled`, `patch_mean`,
  `wavelength_value`, `structured_verbalization`

The current dataset uses `structured_verbalization`, which includes
galaxy type, stellar mass, age, metallicity, SFR estimate, and SNR
alongside the compact spectrum tokens.

Generate a new dataset:
```bash
cd tokenization/
python spec_tokenizer.py \
    --data-path /path/to/sdss_galaxy_spec.hdf5 \
    --output-dir ../data/datasets/<strategy_name> \
    --strategy <strategy_name>
```

---

## 3. Model Details

### 3.1 GPT-OSS-20B (`openai/gpt-oss-20b`)

| Property              | Value                                |
|-----------------------|--------------------------------------|
| Parameters            | ~20B                                 |
| Native quantization   | MXFP4 (dequants to bf16 without triton>=3.4) |
| Dequantized size      | ~40 GB (bf16)                        |
| Attention             | `eager` only (no sdpa / flash_attn_2)|
| Min GPUs (40 GB A100) | 2                                    |

### 3.2 GPT-OSS-120B (`openai/gpt-oss-120b`)

| Property              | Value                                |
|-----------------------|--------------------------------------|
| Parameters            | 116.8B (120.4B total incl. embeddings)|
| Architecture          | MoE (Mixture of Experts)             |
| Native quantization   | MXFP4 (~74 GB on disk, 15 shard files)|
| Dequantized size      | ~240 GB (bf16)                       |
| Attention             | `eager` only                         |
| Min GPUs (40 GB A100) | 8 (with `device_map="auto"`)         |
| GPU memory usage      | 21-38 GB per GPU (uneven due to MoE) |
| RAM during loading    | ~240 GB peak (single process dequant)|

### 3.3 MXFP4 quantization behavior

Both gpt-oss models ship with MXFP4 (Microscaling FP4) quantization.
Without `triton>=3.4`, the models **automatically dequantize to bf16** at
load time. This is transparent but has memory implications:

- **Single process (device_map="auto"):** loads and dequants on CPU, then
  distributes shards to GPUs. Peak RAM = full model in bf16.
- **Multi-process (DeepSpeed):** each rank loads and dequants independently,
  multiplying RAM usage by the number of ranks. See Section 5.

---

## 4. Training

### 4.1 LoRA configuration

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,              # Rank (8 for both 20B and 120B)
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)
```

Trainable parameters: ~6M / 116.8B (0.005%) for the 120B model.

### 4.2 Training hyperparameters

| Hyperparameter              | 20B run         | 120B run         |
|-----------------------------|-----------------|------------------|
| Learning rate               | 1e-4            | 5e-5             |
| LR schedule                 | cosine          | cosine           |
| Warmup                      | 5% of steps     | 5% of steps      |
| Batch size (per device)     | 1               | 1                |
| Gradient accumulation       | 8               | 8                |
| Effective batch size        | 8               | 8                |
| Epochs                      | 2               | 2                |
| Steps per epoch             | ~368            | ~368             |
| Total steps                 | ~736            | ~736             |
| bf16                        | yes             | yes              |
| Gradient checkpointing      | yes             | yes              |
| Block size (max seq length) | 512             | 512              |

**Why lr=5e-5 for 120B:** Larger models are more sensitive to learning rate.
The 20B model used 1e-4 successfully. We halved it for 120B as a conservative
starting point.

### 4.3 Training command: GPT-OSS-20B

```bash
TMPDIR=/tmp \
HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache \
python training/finetune_hf.py \
    --model_name_or_path openai/gpt-oss-20b \
    --dataset_path data/datasets/structured_verbalization/text2text.json \
    --output_dir output_models/gpt-oss-20b_structured \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --lora_r 8 \
    --logging_steps 20 \
    --save_steps 500
```

- **Runtime:** ~48 min on 2x A100
- **Loss trajectory:** 2.14 -> 1.08
- **Output:** `output_models/gpt-oss-20b_structured/` (16 MB adapter)

### 4.4 Training command: GPT-OSS-120B

```bash
TMPDIR=/tmp \
HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python training/finetune_hf.py \
    --model_name_or_path openai/gpt-oss-120b \
    --dataset_path data/datasets/structured_verbalization/text2text.json \
    --output_dir output_models/gpt-oss-120b_structured \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --lora_r 8 \
    --logging_steps 10 \
    --save_steps 500
```

- **Runtime:** ~1.8 hours on 8x A100 (~9 s/step)
- **Loss trajectory:** 1.69 -> 1.42 (in 50-step test; full run in progress)
- **Key:** must load from `openai/gpt-oss-120b` directly (not a pre-converted
  bf16 checkpoint). The MXFP4 dequantization handles MoE expert weight naming
  correctly; a naive `save_pretrained` after dequant produces weights with
  `_blocks` suffix that the architecture doesn't expect.

### 4.5 Generating training plots

```bash
python analysis/plot_training_run.py \
    --run_dir output_models/gpt-oss-20b_structured \
    --output_dir plots/gpt-oss-20b_structured
```

Generates three plots:
- `training_curves.png` — loss, learning rate, and gradient norm vs step
- `loss_by_epoch.png` — loss colored by epoch
- `loss_curve_simple.png` — plain loss curve

---

## 5. What Does NOT Work (and Why)

### 5.1 DeepSpeed ZeRO-3 with GPT-OSS-120B

**Status:** Does not work with the current environment (torch 2.6, triton 3.2).

We attempted six different configurations. All failed:

| Attempt                        | Failure mode                                    |
|--------------------------------|-------------------------------------------------|
| Naive `from_pretrained` + DS   | CPU OOM: 8 ranks x 240 GB = 1.9 TB > 1 TB RAM  |
| `HfDeepSpeedConfig` + `"auto"` batch | `TypeError: '>' not supported (str vs int)` |
| `HfDeepSpeedConfig` + explicit batch  | GPU OOM at allgather (38/40 GB used)      |
| `deepspeed.zero.Init`          | `"auto"` batch size assertion in DS config      |
| `init_distributed` + `zero.Init` | `IndexError` on empty partitioned tensors     |
| Reduced `stage3_max_live_parameters` | Still GPU OOM at allgather                |

**Root causes:**
1. **MXFP4 dequant + multi-rank loading:** Each DeepSpeed rank independently
   loads and dequantizes the model. 8 x 240 GB = 1.9 TB exceeds 1 TB RAM.
2. **Triton version lock:** `torch==2.6` pins `triton==3.2`, but MXFP4 native
   support requires `triton>=3.4`. Cannot keep weights quantized on GPU.
3. **GPU memory budget:** 120B / 8 GPUs = 15B params per GPU = 30 GB in bf16.
   With LoRA adapters and allgather buffers, this exceeds the 40 GB A100 limit.

**To make DeepSpeed work in the future:** Upgrade to `torch>=2.7` (which ships
with `triton>=3.4`), enabling native MXFP4 on GPU. This would keep the model
at ~7.5 GB/GPU instead of 30 GB/GPU.

### 5.2 Pre-converted bf16 checkpoint

**Status:** Do not use.

We saved a bf16 version at `/lcrc/project/cosmo_ai/nramachandra/hf_cache/gpt-oss-120b-bf16`.
It has a critical bug: MoE expert weight names retain the `_blocks` suffix
from the MXFP4 format (`gate_up_proj_blocks` instead of `gate_up_proj`).
When loaded, these are marked UNEXPECTED and the actual parameters are
randomly initialized. Result: loss starts at 5.78 instead of 1.69.

### 5.3 QLoRA via bitsandbytes

The model's native MXFP4 quantization cannot be overridden by a
`BitsAndBytesConfig`. The model always dequantizes to bf16 first.

### 5.4 Attention implementations

Only `eager` and `flash_attention_4` are supported. Using `sdpa` or
`flash_attention_2` raises an error.

---

## 6. Results Summary

### 6.1 GPT-OSS-20B (completed 2026-04-04)

```
Training:  2 epochs, 736 steps, 48 min
Loss:      2.14 -> 1.08 (min 1.078 at step 620)
Grad norm: 0.05 -> 0.20 (healthy, no instability)
FLOPs:     3.67e17
Adapter:   16 MB (output_models/gpt-oss-20b_structured/)
```

### 6.2 GPT-OSS-120B (50-step validation, 2026-04-05)

```
Training:  50 steps, 7.7 min
Loss:      1.69 -> 1.42
Grad norm: 0.18 -> 0.085 (very stable)
Speed:     9.0 s/step
GPU mem:   21-38 GB per GPU (8 GPUs)
RAM:       32 GB (after model loaded to GPUs)
```

The 120B model starts at a lower loss than the 20B's initial loss (1.69 vs 2.14),
demonstrating the larger model's advantage on this domain task.

### 6.3 GPT-OSS-120B (full 2-epoch run, 2026-04-05, in progress)

Same command as Section 4.4. Expected ~1.8 hours.

---

## 7. AutoResearch Integration

### 7.1 What is autoresearch

Karpathy's framework where an AI agent autonomously iterates on a `train.py`
file, running 5-minute training experiments in a loop, keeping improvements.
It trains small GPT models **from scratch** (not fine-tuning a pretrained model).

### 7.2 Environment (ready)

| Component | Value |
|-----------|-------|
| Repo      | `/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/autoresearch/` |
| Venv      | `/lcrc/project/solitons/nramachandra/envs/autoresearch` |
| PyTorch   | 2.9.1+cu128 |
| Data      | `~/.cache/autoresearch/data/` (9 shards, FineWeb-Edu) |
| Tokenizer | `~/.cache/autoresearch/tokenizer/` |

### 7.3 Running the baseline

```bash
source /lcrc/project/solitons/nramachandra/envs/autoresearch/bin/activate
cd /lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/autoresearch
python train.py
```

This trains a small GPT (~50M params, depth=8) for 5 minutes on FineWeb-Edu
data, reporting `val_bpb` at the end.

### 7.4 Adapting for spectral domain

The autoresearch approach could be adapted to:
1. Replace FineWeb-Edu data with our spectral text2text dataset
2. Modify the objective from `val_bpb` to redshift MAE
3. Let the agent explore architecture and hyperparameter changes

This adaptation requires modifying `prepare.py` (data loading) and the
objective function, which are currently read-only in the autoresearch framework.

---

## 8. Plotting

All plots use the publication style defined in `analysis/plots.py:setup_style()`.

### 8.1 Available plot functions

| Function                     | Description                               |
|------------------------------|-------------------------------------------|
| `plot_data_overview`         | 4-panel histogram of physical properties  |
| `plot_sample_spectra`        | Example spectra offset by redshift        |
| `plot_tokenization_comparison`| Token count + MAE per strategy           |
| `plot_loss_curves`           | Training/validation loss vs step          |
| `plot_redshift_scatter`      | True vs predicted z with KDE contours     |
| `plot_mae_vs_hyperparam`     | MAE vs one hyperparameter                 |
| `plot_benchmark_comparison`  | Base vs fine-tuned accuracy bar chart      |
| `plot_experiment_summary`    | 4-panel MAE vs each hyperparameter        |

### 8.2 Training run plots

```bash
python analysis/plot_training_run.py \
    --run_dir output_models/<run_name> \
    --output_dir plots/<run_name>
```

---

## 9. Evaluation (post-training)

### 9.1 Redshift prediction

```bash
python eval/redshift_eval.py \
    --model_path output_models/<run_name> \
    --test_data data/datasets/structured_verbalization/text2text.json
```

### 9.2 LM evaluation harness (benchmark retention)

```bash
bash eval/lm_harness_eval.sh \
    --model_path output_models/<run_name> \
    --tasks mmlu_physics,gpqa,bbh,astromlAB
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No space left on device` during pip install | cosmo_ai fileset quota | `TMPDIR=/tmp HOME=/tmp pip install --no-cache-dir` |
| `No space left on device` during training | TMPDIR defaulting to project space | `export TMPDIR=/tmp` |
| `DS_SKIP_CUDA_CHECK` error | CUDA 13.0 vs 12.4 mismatch | `export DS_SKIP_CUDA_CHECK=1` |
| `torch_dtype is deprecated` | transformers 5.x API change | Use `dtype=` instead |
| `warmup_ratio is deprecated` | transformers 5.x API change | Use `warmup_steps=` instead |
| `deepspeed` resolves to wrong env | PATH picks up env_jax_2024 | Use full path: `/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/deepspeed` |
| MoE weights UNEXPECTED/MISSING | Used pre-converted bf16 model | Load from `openai/gpt-oss-120b` directly |
| GPU OOM with DeepSpeed | 120B bf16 too large for ZeRO-3 on A100-40GB | Use `device_map="auto"` instead |

---

## 11. Reproduction Checklist

To reproduce the full pipeline from scratch:

```bash
# 0. Activate environment
export TMPDIR=/tmp
export HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python

# 1. Verify GPUs
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 2. Verify dataset
wc -c data/datasets/structured_verbalization/text2text.json
# Expected: ~3.5 MB, 2939 instances

# 3. Train GPT-OSS-20B (2x A100, ~48 min)
$PYTHON training/finetune_hf.py \
    --model_name_or_path openai/gpt-oss-20b \
    --dataset_path data/datasets/structured_verbalization/text2text.json \
    --output_dir output_models/gpt-oss-20b_structured \
    --learning_rate 1e-4 --num_train_epochs 2 --lora_r 8

# 4. Train GPT-OSS-120B (8x A100, ~1.8 hrs)
$PYTHON training/finetune_hf.py \
    --model_name_or_path openai/gpt-oss-120b \
    --dataset_path data/datasets/structured_verbalization/text2text.json \
    --output_dir output_models/gpt-oss-120b_structured \
    --learning_rate 5e-5 --num_train_epochs 2 --lora_r 8

# 5. Generate plots
$PYTHON analysis/plot_training_run.py \
    --run_dir output_models/gpt-oss-20b_structured
$PYTHON analysis/plot_training_run.py \
    --run_dir output_models/gpt-oss-120b_structured

# 6. Evaluate (after training completes)
$PYTHON eval/redshift_eval.py \
    --model_path output_models/gpt-oss-120b_structured \
    --test_data data/datasets/structured_verbalization/text2text.json
```
