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
`run_benchmarks.sh` runs lm-eval on the base and fine-tuned checkpoints for both
gpt-oss-20b and gpt-oss-120b. Models run in **pure bf16** (MXFP4 weights are
dequantized at load time — no quantized inference).

Required environment (handled by the script):
- `PYTHONPATH` → custom lm-eval at `/lcrc/project/solitons/nramachandra/lm_eval_pkg`
- `LD_LIBRARY_PATH` → pip-installed NVIDIA libs in the `bimodal` env
  (torch ships `libcusparseLt.so.0` inside its nvidia wheels; the cluster's
  default search path does not include them)
- `max_memory_per_gpu=50GiB`, `parallelize=True`, `attn_implementation=eager`
  (GptOssForCausalLM only supports eager)

Run the full suite (20B + 120B, ~few hours on 8×A100-80GB):
```bash
nohup bash run_benchmarks.sh 0 > benchmark.log 2>&1 &
tail -f benchmark.log
```

Run a single model/task combination directly:
```bash
# Source the env bits from run_benchmarks.sh, then:
python -m lm_eval \
    --model hf \
    --model_args "pretrained=openai/gpt-oss-20b,trust_remote_code=True,dtype=bfloat16,parallelize=True,max_memory_per_gpu=50GiB,attn_implementation=eager" \
    --tasks mmlu_college_physics \
    --batch_size 1 --num_fewshot 0 \
    --output_path ./eval_out
```

Add `peft=path/to/adapter` to the `--model_args` to evaluate a LoRA-fine-tuned
checkpoint. For the 120B model, change `pretrained=openai/gpt-oss-120b`; the
same settings work (ensure all 8 GPUs are visible).

Results land in `overnight_results/latest/benchmark_{base,ft}_{20b,120b}*/` as
standard lm-eval JSON, plus a `benchmark_comparison.json` and PNG plots.

## Key Design Choices (see TODO.md for details)
1. **Model:** GPT-OSS-120B (switched from Llama-3-8B)
2. **Tokenization:** Pluggable strategies (digit, hex, quantized vocab, ...)
3. **AutoResearch:** Planned — replace grid search with automated exploration
4. **Interpretability:** Planned — logit lens, weight diffs, probing classifiers
