# bimodal_reasoning — TODO & Design Roadmap

This file tracks design decisions and future work for the bimodal (spectra + text) reasoning project.

---

## Active Design Decisions

### [1] Model: Switch to GPT-OSS-120B
- **Status:** In progress — config.yaml updated, training/eval scripts point to `gpt-oss-120b`
- **Notes:**
  - 120B requires FSDP or DeepSpeed Zero-3 (scripts already updated for this)
  - Use `bfloat16` instead of `fp16`
  - LoRA rank may need tuning (start with r=8 or r=16, large models often need lower rank)
  - Verify the model ID on HuggingFace / internal registry before running

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

### [3] AutoResearch Integration
- **Status:** TODO — placeholder stub at `experiments/autoresearch_stub.py`
- **Goal:** Use Karpathy's autoresearch framework to auto-discover best fine-tuning configurations
  - Repo: https://github.com/karpathy/autoresearch
  - Replace manual grid search in `config.yaml` with autoresearch-driven exploration
  - Define a reward/objective function (redshift MAE + benchmark retention)
  - The criteria for "best model" needs to be decided (see below)
- **Candidate objective:**
  - Primary: minimize redshift MAE (domain task)
  - Secondary: preserve ≥ X% of base-model benchmark scores (MMLU physics, GPQA, BBH)
  - Penalty: catastrophic forgetting (if scientific reasoning drops > 5% absolute)
- **Steps:**
  1. Install autoresearch: `pip install autoresearch` (or clone from GitHub)
  2. Define `autoresearch_config.yaml` with search space and objective
  3. Replace `generate_experiments.py` loop with autoresearch-guided trials
  4. Store autoresearch logs in `experiments/autoresearch_runs/`

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
