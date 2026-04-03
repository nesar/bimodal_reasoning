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
│   └── lm_harness_eval.sh         # LM eval harness benchmarks
│
├── experiments/
│   ├── generate_experiments.py    # Generate experiment configs from config.yaml
│   └── run_experiment_template.sh # Single experiment orchestration
│
├── run_suite.sh                   # High-level suite runner
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

## Key Design Choices (see TODO.md for details)
1. **Model:** GPT-OSS-120B (switched from Llama-3-8B)
2. **Tokenization:** Pluggable strategies (digit, hex, quantized vocab, ...)
3. **AutoResearch:** Planned — replace grid search with automated exploration
4. **Interpretability:** Planned — logit lens, weight diffs, probing classifiers
