#!/bin/bash
# run_overnight.sh — Retrain with compact dataset + eval + autoresearch
#
# Phases:
#   1: Retrain 20B with compact dataset (~50 min)
#   2: Eval retrained 20B on redshift prediction (~15 min)
#   3: Eval old 20B model for comparison (~15 min)
#   4: Plots (~2 min)
#   5: Autoresearch loop (fills remaining time)
#   6: Final autoresearch plot
#
# Usage:
#   nohup bash run_overnight.sh > overnight.log 2>&1 &
#   tail -f overnight_results/latest/STATUS

set -uo pipefail

# ── Environment ──────────────────────────────────────────────────────────

export TMPDIR=/tmp
export HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache
export HUGGINGFACE_HUB_CACHE=/lcrc/project/cosmo_ai/nramachandra/hf_cache/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export C_INCLUDE_PATH=/home/nramachandra/anaconda3/envs/eval-harness/include/python3.10
export CPATH=/home/nramachandra/anaconda3/envs/eval-harness/include/python3.10
export PYTHONPATH=/lcrc/project/solitons/nramachandra/lm_eval_pkg:${PYTHONPATH:-}

BASE_DIR="/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/bimodal_reasoning"
BIMODAL_PYTHON="/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python"
AR_PYTHON="/lcrc/project/solitons/nramachandra/envs/autoresearch/bin/python"
AR_DIR="/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/autoresearch"

DATASET_COMPACT="$BASE_DIR/data/datasets/structured_verbalization_compact/text2text.json"
DATASET_OLD="$BASE_DIR/data/datasets/structured_verbalization/text2text.json"
ADAPTER_20B_OLD="$BASE_DIR/output_models/gpt-oss-20b_structured"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$BASE_DIR/overnight_results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"
ln -sfn "$RESULTS_DIR" "$BASE_DIR/overnight_results/latest"

cd "$BASE_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESULTS_DIR/STATUS"; }

log "=== OVERNIGHT JOB STARTED ==="
log "Results dir: $RESULTS_DIR"

# Validate
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader > "$RESULTS_DIR/gpu_info.txt" 2>&1
log "GPUs: $(wc -l < "$RESULTS_DIR/gpu_info.txt") available"
[[ -f "$DATASET_COMPACT" ]] || { log "FATAL: compact dataset not found"; exit 1; }
log "Phase 0: Validation passed"

# ── Phase 1: Retrain 20B with compact dataset ───────────────────────────

ADAPTER_20B_COMPACT="$BASE_DIR/output_models/gpt-oss-20b_compact"

log "Phase 1: START — Retrain 20B with compact dataset"
$BIMODAL_PYTHON training/finetune_hf.py \
    --model_name_or_path openai/gpt-oss-20b \
    --dataset_path "$DATASET_COMPACT" \
    --output_dir "$ADAPTER_20B_COMPACT" \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --lora_r 8 \
    --logging_steps 20 \
    --save_steps 500 \
    > "$RESULTS_DIR/phase1_train.log" 2>&1 && {
    FINAL_LOSS=$(grep "'train_loss'" "$RESULTS_DIR/phase1_train.log" | tail -1 | grep -oP "'train_loss': '[\d.]+'|'train_loss': [\d.]+" | grep -oP "[\d.]+$")
    log "Phase 1: DONE — train_loss=${FINAL_LOSS}"
} || {
    log "Phase 1: FAILED — see $RESULTS_DIR/phase1_train.log"
}

# ── Phase 2: Eval retrained 20B (compact) ───────────────────────────────

log "Phase 2: START — Eval 20B compact on redshift prediction"
PHASE2_DIR="$RESULTS_DIR/phase2_eval_20b_compact"
$BIMODAL_PYTHON eval/redshift_eval_peft.py \
    --base_model openai/gpt-oss-20b \
    --adapter_path "$ADAPTER_20B_COMPACT" \
    --dataset "$DATASET_COMPACT" \
    --output_dir "$PHASE2_DIR" \
    --num_test 200 \
    --label "GPT-OSS-20B (compact)" \
    > "$PHASE2_DIR.log" 2>&1 && {
    log "Phase 2: DONE — $(grep 'Metrics:' "$PHASE2_DIR.log" | tail -1)"
} || {
    log "Phase 2: FAILED — see $PHASE2_DIR.log"
}

# ── Phase 3: Eval old 20B (original dataset) for comparison ─────────────

log "Phase 3: START — Eval 20B old model (original dataset)"
PHASE3_DIR="$RESULTS_DIR/phase3_eval_20b_old"
$BIMODAL_PYTHON eval/redshift_eval_peft.py \
    --base_model openai/gpt-oss-20b \
    --adapter_path "$ADAPTER_20B_OLD" \
    --dataset "$DATASET_OLD" \
    --output_dir "$PHASE3_DIR" \
    --num_test 200 \
    --label "GPT-OSS-20B (original)" \
    > "$PHASE3_DIR.log" 2>&1 && {
    log "Phase 3: DONE — $(grep 'Metrics:' "$PHASE3_DIR.log" | tail -1)"
} || {
    log "Phase 3: FAILED — see $PHASE3_DIR.log"
}

# ── Phase 4: Plots ──────────────────────────────────────────────────────

log "Phase 4: START — Generating plots"

# Training plots for the compact run
$BIMODAL_PYTHON analysis/plot_training_run.py \
    --run_dir "$ADAPTER_20B_COMPACT" \
    --output_dir plots/gpt-oss-20b_compact \
    > "$RESULTS_DIR/phase4_train_plots.log" 2>&1

# Money plot
$BIMODAL_PYTHON analysis/plot_money.py \
    --output_dir plots/summary \
    --runs "GPT-OSS-20B (original)=output_models/gpt-oss-20b_structured" \
           "GPT-OSS-20B (compact)=$ADAPTER_20B_COMPACT" \
           "GPT-OSS-120B=output_models/gpt-oss-120b_structured" \
    --evals "GPT-OSS-20B (compact)=$PHASE2_DIR" \
            "GPT-OSS-20B (original)=$PHASE3_DIR" \
    > "$RESULTS_DIR/phase4_money_plot.log" 2>&1 && {
    log "Phase 4: DONE — plots in plots/summary/"
} || {
    log "Phase 4: FAILED — see $RESULTS_DIR/phase4_money_plot.log"
}
cp -r plots/summary "$RESULTS_DIR/plots_summary" 2>/dev/null || true

# ── Phase 5: Autoresearch loop ──────────────────────────────────────────

log "Phase 5: START — Autoresearch loop"

AR_RESULTS="$RESULTS_DIR/autoresearch"
mkdir -p "$AR_RESULTS"

JOB_START_EPOCH=$(date +%s)
MAX_DURATION_SEC=$((8 * 3600))

run_count=0
consecutive_fails=0
MAX_CONSECUTIVE_FAILS=3
cd "$AR_DIR"

printf "run\tval_bpb\tpeak_vram_gb\tstatus\tdescription\n" > "$AR_RESULTS/results.tsv"

while true; do
    ELAPSED=$(( $(date +%s) - JOB_START_EPOCH ))
    REMAINING=$(( MAX_DURATION_SEC - ELAPSED ))

    if [[ $REMAINING -lt 600 ]]; then
        log "Phase 5: Time limit reached. Stopping."
        break
    fi

    run_count=$((run_count + 1))
    RUN_LOG="$AR_RESULTS/run_${run_count}.log"

    log "Phase 5: Autoresearch run #${run_count} (${REMAINING}s remaining)"

    timeout 600 $AR_PYTHON train.py > "$RUN_LOG" 2>&1
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        VAL_BPB=$(grep "^val_bpb:" "$RUN_LOG" | awk '{print $2}')
        PEAK_VRAM=$(grep "^peak_vram_mb:" "$RUN_LOG" | awk '{print $2}')
        PEAK_GB=$(echo "scale=1; ${PEAK_VRAM:-0} / 1024" | bc 2>/dev/null || echo "0.0")
        log "Phase 5: Run #${run_count} — val_bpb=${VAL_BPB}, vram=${PEAK_GB}GB"
        printf "${run_count}\t${VAL_BPB}\t${PEAK_GB}\tkeep\tbaseline run #${run_count}\n" >> "$AR_RESULTS/results.tsv"
        consecutive_fails=0
    elif [[ $EXIT_CODE -eq 124 ]]; then
        log "Phase 5: Run #${run_count} — TIMEOUT"
        printf "${run_count}\t0.0\t0.0\tcrash\ttimeout\n" >> "$AR_RESULTS/results.tsv"
        consecutive_fails=$((consecutive_fails + 1))
    else
        log "Phase 5: Run #${run_count} — FAILED (exit $EXIT_CODE)"
        printf "${run_count}\t0.0\t0.0\tcrash\texit code ${EXIT_CODE}\n" >> "$AR_RESULTS/results.tsv"
        consecutive_fails=$((consecutive_fails + 1))
    fi

    if [[ $consecutive_fails -ge $MAX_CONSECUTIVE_FAILS ]]; then
        log "Phase 5: ${MAX_CONSECUTIVE_FAILS} consecutive failures. Stopping."
        break
    fi
done

cd "$BASE_DIR"

# ── Phase 6: Autoresearch progress plot ──────────────────────────────────

log "Phase 6: START — Autoresearch summary plot"
$BIMODAL_PYTHON -c "
import sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from analysis.plots import setup_style, COLORS
setup_style()

runs, bpbs = [], []
with open('$AR_RESULTS/results.tsv') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4 and parts[3] == 'keep':
            runs.append(int(parts[0]))
            bpbs.append(float(parts[1]))

if not runs:
    print('No successful autoresearch runs to plot')
    sys.exit(0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(runs, bpbs, 'o-', color=COLORS['primary'], ms=6, lw=1, alpha=0.7)

best = [min(bpbs[:i+1]) for i in range(len(bpbs))]
ax.plot(runs, best, '-', color=COLORS['accent'], lw=2, label=f'Best: {min(bpbs):.6f}')

ax.set_xlabel('Run #')
ax.set_ylabel('val_bpb')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, lw=0.6)
ax.set_title('AutoResearch: val_bpb Over Experiments')
fig.tight_layout()
fig.savefig('$RESULTS_DIR/autoresearch_progress.png')
fig.savefig('$BASE_DIR/plots/autoresearch_progress.png')
print('Saved autoresearch progress plot')
" > "$RESULTS_DIR/phase6_plot.log" 2>&1 && {
    log "Phase 6: DONE"
} || {
    log "Phase 6: FAILED"
}

# ── Summary ──────────────────────────────────────────────────────────────

log "=== OVERNIGHT JOB COMPLETE ==="
log "Total time: $(( ($(date +%s) - JOB_START_EPOCH) / 60 )) minutes"
log ""
log "Results:"
for f in "$RESULTS_DIR"/phase*_eval_*/metrics.json; do
    [[ -f "$f" ]] && log "  $(python3 -c "import json; d=json.load(open('$f')); print(d.get('label','?'), '— MAE:', d.get('mae','?'), 'Valid:', d.get('n_valid','?'))" 2>/dev/null || echo "  $f")"
done
log "Autoresearch runs: $run_count"
if [[ -f "$AR_RESULTS/results.tsv" ]]; then
    BEST_BPB=$(awk -F'\t' 'NR>1 && $2>0 {if(!best || $2<best) best=$2} END{print best+0}' "$AR_RESULTS/results.tsv")
    log "Autoresearch best val_bpb: $BEST_BPB"
fi
