# bimodal_reasoning

Fine-tuning and evaluation framework for bimodal (galaxy spectra + text) reasoning with large language models.

## Overview

This project fine-tunes LLMs to predict galaxy properties (redshift, age, metallicity, stellar mass) from tokenized
SDSS galaxy spectra, then evaluates whether fine-tuning preserves general reasoning ability.

**Current model:** `gpt-oss-120b`
**Data:** SDSS galaxy spectra (HDF5), ~9800 samples, 4556 wavelength channels
**Task:** Text-to-text — encode spectrum as digit sequence → predict redshift token

## Directory Layout

```
bimodal_reasoning/
├── TODO.md                        # Design roadmap and future work
├── README.md                      # This file
├── config.yaml                    # Central experiment configuration
│
├── data/
│   └── read_data.py               # Load and preprocess HDF5 spectral data
│
├── tokenization/
│   └── spec_tokenizer.py          # Convert spectra → text2text JSON dataset
│                                  # Supports multiple tokenization strategies
│
├── training/
│   └── finetune_lora.sh           # LoRA fine-tuning with DeepSpeed/FSDP
│
├── eval/
│   ├── redshift_eval.py           # Domain eval: redshift prediction MAE
│   └── lm_harness_eval.sh         # LM eval harness benchmarks (vLLM template)
│
├── experiments/
│   ├── generate_experiments.py    # Generate experiment configs from config.yaml
│   └── run_experiment_template.sh # Single experiment orchestration
│
├── run_suite.sh                   # High-level suite runner
├── run_benchmarks.sh              # End-to-end lm-eval harness (20B + 120B, base + FT)
│
└── analysis/
    ├── collect_results.py         # Aggregate results → tables + LaTeX
    └── benchmark_extraction.py    # Parse LM harness JSON → metrics.json
```

## Quickstart

### 1. Prepare dataset
```bash
cd tokenization/
python spec_tokenizer.py \
    --data-path /path/to/sdss_galaxy_spec.hdf5 \
    --output-dir ../data/datasets/spec_text2text \
    --strategy digit_base10
```

### 2. Generate experiments
```bash
python experiments/generate_experiments.py --config config.yaml
```

### 3. Run a single experiment
```bash
bash experiments/<experiment_id>/run_experiment.sh
```

### 4. Run all experiments
```bash
bash run_suite.sh --run-all
```

### 5. Collect results
```bash
bash run_suite.sh --collect-results
```

### 6. Run LM eval harness benchmarks (MMLU, GPQA, BBH, …)
Models run in **pure bf16** (MXFP4 weights are dequantized at load time — no
quantized inference, no weight changes).

**One-shot terminal command** (single task, base or adapter, 80GB A100s):
```bash
python experiments/benchmark_adapter.py \
    --base-model openai/gpt-oss-20b \
    --adapter   output_models/gpt-oss-20b_compact \
    --mode      fast \
    --output-dir /tmp/bench_run
```
Omit `--adapter` to evaluate the base model. `--mode full` runs the full task
suite; `--tasks "a,b,c"` overrides with an explicit list. The script sets
`PYTHONPATH`, `LD_LIBRARY_PATH` (for `libcusparseLt.so.0` inside the torch
nvidia wheels), and the correct `parallelize=True,max_memory_per_gpu=50GiB,
attn_implementation=eager` model args — the only combination that works for
`GptOssForCausalLM` in pure bf16.

**Full paired suite** (base vs FT, 20B + 120B, ~few hours on 8×A100-80GB):
```bash
nohup bash run_benchmarks.sh 0 > benchmark.log 2>&1 &
tail -f benchmark.log
```

Raw lm-eval invocation (same args benchmark_adapter.py would use):
```bash
python -m lm_eval \
    --model hf \
    --model_args "pretrained=openai/gpt-oss-20b,trust_remote_code=True,dtype=bfloat16,parallelize=True,max_memory_per_gpu=50GiB,attn_implementation=eager" \
    --tasks mmlu_college_physics \
    --batch_size 1 --num_fewshot 0 \
    --output_path ./eval_out
```
Add `peft=path/to/adapter` inside the model_args to evaluate a LoRA checkpoint.
Swap `gpt-oss-20b` → `gpt-oss-120b` for the big model.

Results land in `overnight_results/latest/benchmark_{base,ft}_{20b,120b}*/` as
standard lm-eval JSON, plus a `benchmark_comparison.json` and PNG plots.

### 7. AutoResearch with benchmark retention
The autoresearch loop optionally evaluates each trial against the lm-eval
harness and scores it with the dual objective in
`experiments/autoresearch_stub.py` (minimize redshift MAE **and** preserve base-
model MMLU/GPQA/BBH scores). One-time baseline setup (reads from the last full
base run under `overnight_results/latest`):

```bash
python experiments/benchmark_adapter.py \
    --base-model openai/gpt-oss-20b \
    --mode fast \
    --output-dir experiments/autoresearch_runs/base_fast
# → writes experiments/autoresearch_runs/base_benchmarks.json
```

Then run the loop with benchmark-aware selection:
```bash
WITH_BENCHMARKS=1 bash experiments/run_loop.sh
```
Each trial still trains + measures redshift MAE (fast), then also runs
`TASKS_FAST` from `benchmark_adapter.py` (~2 extra min) and logs
`bench_sci_reasoning`, `bench_general_qa`, and the compound `score`. Default
(`WITH_BENCHMARKS` unset) keeps the MAE-only selection used so far.

### 8. Multi-objective Pareto sweep (overnight)
A randomized search that tracks **both** objectives independently and traces
the Pareto front:
  * **MAE** (minimize) — redshift prediction
  * **MCQ score** (maximize) — `(sci_reasoning + general_qa) / 2`

```bash
nohup bash experiments/run_pareto_overnight.sh 50 > pareto.log 2>&1 &
tail -f pareto.log
```

Each trial samples `experiments/pareto_loop.py:SEARCH_SPACE` (lr, lora_r,
lora_alpha, lora_dropout, grad_accum, max_steps, warmup_ratio), trains,
benchmarks, and appends one JSON line to
`experiments/autoresearch_runs/pareto/results.jsonl`. The Pareto front is
recomputed and `plots/autoresearch_pareto.png` is regenerated after every
trial — progress is visible mid-run. The loop is resumable: re-running picks
up after the highest `trial` index already in the JSONL.

The plot (see `analysis/plot_pareto_money.py`) has two panels: a scatter of
(MAE, MCQ) coloured by trial order with the Pareto front highlighted, and a
twin-axes convergence panel showing best-MAE-so-far and best-MCQ-so-far vs
trial #.

## Key Design Choices (see TODO.md for details)
1. **Model:** GPT-OSS-120B (switched from Llama-3-8B)
2. **Tokenization:** Pluggable strategies (digit, hex, quantized vocab, ...)
3. **AutoResearch:** Planned — replace grid search with automated exploration
4. **Interpretability:** Planned — logit lens, weight diffs, probing classifiers
