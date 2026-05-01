# bimodal_reasoning — TODO & Design Roadmap

This file tracks design decisions and future work for the bimodal (spectra + text) reasoning project.

---

## Active Design Decisions

### [1] Model: Switch to GPT-OSS-120B
- **Status:** DONE — 120B trained (LoRA r=8, structured verbalization adapter at
  `output_models/gpt-oss-120b_structured/`) and benchmarked end-to-end on 8×A100-80GB
- **120B vs 20B benchmark headline** (mean Δ accuracy after fine-tuning, FT − base):
  - 20B: −3.17 pp average (12 tasks degraded, worst `bbh_temporal_sequences` −72.8)
  - 120B: +0.04 pp average — preserved general reasoning, gained on GPQA (+3.4)
  - See `overnight_results/latest/benchmark_diff_20b_vs_120b.png`
- **Production-ready settings** (no quantization):
  - bf16 dequantized at load, `parallelize=True`, `max_memory_per_gpu=35GiB`,
    `attn_implementation=eager`, `max_length=4096` (caps the 18 GiB eager-attention
    activation that would otherwise OOM GPU 0)
  - All in `run_benchmarks.sh` and `experiments/benchmark_adapter.py`

### [2] Tokenization Iteration
- **Status:** Implemented — `tokenization/spec_tokenizer.py` + `tokenization/verbalize.py`
- **Strategies available:**
  - `digit_base10` (original): serialize each flux value as comma-separated digits
  - `digit_base16`: hex encoding — fewer tokens per value
  - `log_scaled`: log-transform spectra before serialization
  - `patch_mean`: encode fixed-length patch means (downsampling)
  - `wavelength_value`: `(wavelength_index, flux)` pairs for interpretability
  - `structured_verbalization` (**NEW**): rich structured text block + compact spectrum tokens
    - Verbalization includes: object pseudo-ID, galaxy type + confidence, redshift [PREDICT],
      stellar mass, age, metallicity, estimated SFR, survey + SNR
    - Physical properties via `read_with_physical()` in `data/read_data.py`
    - Galaxy type classifier: QG / SFG / Starburst / Green-valley based on mass + age
    - SFR estimate: M_* / (age × 10⁹)  [M_sun/yr]
    - SNR estimate: percentile-90 / std proxy from normalized spectrum
- **Evaluation criterion:** MAE on redshift prediction AND token count per sample
- **Tracked via:** `config.yaml` → `tokenization_strategy` field
- **Next:** compare redshift MAE across strategies; try masking different fields at training time

### [3] AutoResearch
- **Status:** WORKING — hand-rolled greedy axis-aligned search has produced 12 trials
  on 20B; dual-objective scoring is wired but not yet exercised end-to-end
- **What's done:**
  - `experiments/run_loop.sh` — sequential trial loop, `keep`/`discard` vs current best
  - `experiments/run_experiment.py` — single-trial training + redshift MAE eval (~10 min)
  - `experiments/autoresearch_runs/results.tsv` — 12 trials logged, best MAE **0.0561**
    (300 training steps, lr=1e-4, r=8); see `analysis/plot_autoresearch_money.py`
  - `experiments/benchmark_adapter.py` — fast lm-eval helper (~2 min/trial via `TASKS_FAST`)
  - `experiments/autoresearch_stub.py:compute_objective` — dual objective penalizing
    >5% drop on sci_reasoning or general_qa
  - `experiments/autoresearch_runs/base_benchmarks.json` — 20B baseline scores
  - `WITH_BENCHMARKS=1 bash experiments/run_loop.sh` — opt-in dual-objective mode
- **Multi-objective sweep (NEW):**
  - `experiments/pareto_loop.py` — random sampling, dual-objective scoring,
    incremental Pareto front, JSONL log, plot refresh after every trial
  - `experiments/run_pareto_overnight.sh` — overnight launcher (`nohup … &`)
  - `analysis/plot_pareto_money.py` — Pareto-aware money plot (scatter + Pareto
    front + twin-axes convergence)
  - **Run it:** `nohup bash experiments/run_pareto_overnight.sh 50 > pareto.log 2>&1 &`
- **Still to do:**
  1. Run the overnight sweep on 20B and analyze the Pareto front (~10 hours, 50 trials)
  2. Replace random sampling with a smarter proposer. Options: Karpathy's
     https://github.com/karpathy/autoresearch, Optuna's NSGA-II sampler, Ax —
     all slot in by replacing `sample_config()` in `pareto_loop.py`
  3. Expand the search space to include tokenization strategies (currently fixed at
     `structured_verbalization_compact`), block_size, and dataset size
  4. Decide whether to also sweep 120B once the 20B Pareto front is mapped

### [4] Interpretability: Where Do Spectra Live in the Weights?
- **Status:** TODO — design study
- **Goal:** Understand how fine-tuning on spectra changes the model weights
- **Questions to answer:**
  1. Which layers change the most after fine-tuning? (weight diff norms per layer)
  2. Where is redshift information stored? (logit lens / representation probing)
  3. Do spectra tokens activate the same circuits as numbers in pre-training? (logit lens)
  4. What is the token-level contribution to the redshift prediction? (attention attribution)
- **References:**
  - Logit Lens: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
  - Logit Prisms: https://neuralblog.github.io/logit-prisms/
  - Andrej Karpathy weight visualization lecture: https://www.youtube.com/watch?v=UGO_Ehywuxc
- **Implementation plan:**
  1. Script: `analysis/logit_lens.py` — extract intermediate representations at each layer for spectrum tokens
  2. Script: `analysis/weight_diff.py` — compute per-layer weight change norms (pre vs post fine-tuning)
  3. Script: `analysis/probing.py` — train linear probes on layer activations to predict redshift
  4. Compare spectrum token embeddings to numeric token embeddings from pre-training data
- **Saved checkpoints needed:** base model + fine-tuned model (LoRA merged)

---

## Backlog

- [ ] Add support for multi-target fine-tuning (redshift + age + metallicity + stellar mass simultaneously)
- [ ] Try `text-to-spectrum` direction: can the model generate a plausible spectrum given a redshift?
- [ ] Explore longer context: current block_size=512 covers ~256 flux bins. Try 1024 or 2048.
- [ ] Add data augmentation: noise injection, random wavelength shifts
- [ ] Evaluate on OOD data (DESI spectra vs SDSS training)
- [ ] Add SLURM job submission wrappers for the HPC environment

---

## Done

- [x] Initial LoRA fine-tuning pipeline on Llama-3-8B-Instruct
- [x] Digit-base-10 tokenization of SDSS galaxy spectra
- [x] LM eval harness integration (BBH, GPQA, MMLU physics, AstroMLAB)
- [x] Experiment suite with grid search (lr, lora_r, epochs, training_size)
- [x] Results collection into LaTeX tables
- [x] Switch to GPT-OSS-20B + 120B (config.yaml, training, eval scripts)
- [x] Structured verbalization tokenization strategy (rich text + compact spectrum)
- [x] 20B + 120B fine-tuned LoRA adapters trained
- [x] lm-eval harness benchmarks for both 20B and 120B (base + fine-tuned),
      pure bf16 with `parallelize=True` + `max_length=4096` for 120B
- [x] 4-model comparison plots (20B vs 120B base/FT delta — see
      `analysis/plot_benchmark_diff.py`, `overnight_results/latest/benchmark_diff_20b_vs_120b.png`)
- [x] AutoResearch dual-objective scaffolding (helper + scoring + baseline JSON)
