#!/bin/bash
# run_overnight.sh — Comprehensive evaluation + autoresearch overnight job
#
# Phases:
#   0: Setup and validation
#   1: Redshift eval — 120B fine-tuned (~30 min)
#   2: Redshift eval — 20B fine-tuned (~15 min)
#   3: Redshift eval — 120B base model (~30 min, for comparison)
#   4: Comprehensive plots (~2 min)
#   5: Autoresearch loop (fills remaining time, ~5-7 hours)
#
# Usage:
#   nohup bash run_overnight.sh > overnight.log 2>&1 &
#   tail -f overnight.log   # monitor from another terminal
#
# To check status:
#   cat overnight_results/latest/STATUS
#
# Estimated total: 8-10 hours

set -uo pipefail  # no -e: we handle errors per-phase

# ── Phase 0: Environment ─────────────────────────────────────────────────

export TMPDIR=/tmp
export HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache
export HUGGINGFACE_HUB_CACHE=/lcrc/project/cosmo_ai/nramachandra/hf_cache/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export C_INCLUDE_PATH=/home/nramachandra/anaconda3/envs/eval-harness/include/python3.10
export CPATH=/home/nramachandra/anaconda3/envs/eval-harness/include/python3.10

BASE_DIR="/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/bimodal_reasoning"
BIMODAL_PYTHON="/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python"
AR_PYTHON="/lcrc/project/solitons/nramachandra/envs/autoresearch/bin/python"
AR_DIR="/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/autoresearch"

DATASET="$BASE_DIR/data/datasets/structured_verbalization/text2text.json"
ADAPTER_120B="$BASE_DIR/output_models/gpt-oss-120b_structured"
ADAPTER_20B="$BASE_DIR/output_models/gpt-oss-20b_structured"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$BASE_DIR/overnight_results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# Symlink for easy access
ln -sfn "$RESULTS_DIR" "$BASE_DIR/overnight_results/latest"

cd "$BASE_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESULTS_DIR/STATUS"; }

log "=== OVERNIGHT JOB STARTED ==="
log "Results dir: $RESULTS_DIR"
log "Timestamp: $TIMESTAMP"

# Validate
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader > "$RESULTS_DIR/gpu_info.txt" 2>&1
log "GPUs: $(wc -l < "$RESULTS_DIR/gpu_info.txt") available"

for f in "$DATASET" "$ADAPTER_120B/adapter_config.json" "$ADAPTER_20B/adapter_config.json"; do
    if [[ ! -f "$f" ]]; then
        log "FATAL: Missing file $f"
        exit 1
    fi
done
log "Phase 0: Validation passed"

# ── Phase 1: Redshift eval — 120B fine-tuned ─────────────────────────────

log "Phase 1: START — 120B fine-tuned redshift eval"
PHASE1_DIR="$RESULTS_DIR/phase1_eval_120b_ft"
$BIMODAL_PYTHON eval/redshift_eval_peft.py \
    --base_model openai/gpt-oss-120b \
    --adapter_path "$ADAPTER_120B" \
    --dataset "$DATASET" \
    --output_dir "$PHASE1_DIR" \
    --num_test 200 \
    --label "GPT-OSS-120B (fine-tuned)" \
    > "$PHASE1_DIR.log" 2>&1 && {
    log "Phase 1: DONE — $(grep 'MAE=' "$PHASE1_DIR.log" | tail -1)"
} || {
    log "Phase 1: FAILED — see $PHASE1_DIR.log"
}

# ── Phase 2: Redshift eval — 20B fine-tuned ──────────────────────────────

log "Phase 2: START — 20B fine-tuned redshift eval"
PHASE2_DIR="$RESULTS_DIR/phase2_eval_20b_ft"
$BIMODAL_PYTHON eval/redshift_eval_peft.py \
    --base_model openai/gpt-oss-20b \
    --adapter_path "$ADAPTER_20B" \
    --dataset "$DATASET" \
    --output_dir "$PHASE2_DIR" \
    --num_test 200 \
    --label "GPT-OSS-20B (fine-tuned)" \
    > "$PHASE2_DIR.log" 2>&1 && {
    log "Phase 2: DONE — $(grep 'MAE=' "$PHASE2_DIR.log" | tail -1)"
} || {
    log "Phase 2: FAILED — see $PHASE2_DIR.log"
}

# ── Phase 3: Redshift eval — 120B base (no adapter) ─────────────────────

log "Phase 3: START — 120B base model redshift eval (no adapter)"
PHASE3_DIR="$RESULTS_DIR/phase3_eval_120b_base"
$BIMODAL_PYTHON eval/redshift_eval_peft.py \
    --base_model openai/gpt-oss-120b \
    --dataset "$DATASET" \
    --output_dir "$PHASE3_DIR" \
    --num_test 100 \
    --label "GPT-OSS-120B (base)" \
    > "$PHASE3_DIR.log" 2>&1 && {
    log "Phase 3: DONE — $(grep 'MAE=' "$PHASE3_DIR.log" | tail -1)"
} || {
    log "Phase 3: FAILED — see $PHASE3_DIR.log"
}

# ── Phase 4: Comprehensive plots ─────────────────────────────────────────

log "Phase 4: START — Generating comprehensive plots"
PLOT_DIR="$BASE_DIR/plots/summary"
$BIMODAL_PYTHON analysis/plot_money.py \
    --output_dir "$PLOT_DIR" \
    --runs "GPT-OSS-20B=output_models/gpt-oss-20b_structured" \
           "GPT-OSS-120B=output_models/gpt-oss-120b_structured" \
    --evals "GPT-OSS-20B (FT)=$PHASE2_DIR" \
            "GPT-OSS-120B (FT)=$PHASE1_DIR" \
    > "$RESULTS_DIR/phase4_plots.log" 2>&1 && {
    log "Phase 4: DONE — plots in $PLOT_DIR"
} || {
    log "Phase 4: FAILED — see $RESULTS_DIR/phase4_plots.log"
}

# Copy plots to results dir for self-contained archive
cp -r "$PLOT_DIR" "$RESULTS_DIR/plots_summary" 2>/dev/null || true

# ── Phase 5: Autoresearch loop ───────────────────────────────────────────

log "Phase 5: START — Autoresearch loop (filling remaining time)"

AR_RESULTS="$RESULTS_DIR/autoresearch"
mkdir -p "$AR_RESULTS"

# How much time remains? Target 8 hours total from start.
JOB_START_EPOCH=$(date -d "$TIMESTAMP" +%s 2>/dev/null || date +%s)
MAX_DURATION_SEC=$((8 * 3600))  # 8 hours

run_count=0
cd "$AR_DIR"

# Create results.tsv header if it doesn't exist
if [[ ! -f results.tsv ]]; then
    printf "run\tval_bpb\tpeak_vram_gb\tstatus\tdescription\n" > results.tsv
fi

while true; do
    ELAPSED=$(( $(date +%s) - JOB_START_EPOCH ))
    REMAINING=$(( MAX_DURATION_SEC - ELAPSED ))

    # Stop if less than 10 minutes remaining
    if [[ $REMAINING -lt 600 ]]; then
        log "Phase 5: Time limit reached ($ELAPSED s elapsed). Stopping."
        break
    fi

    run_count=$((run_count + 1))
    RUN_LOG="$AR_RESULTS/run_${run_count}.log"

    log "Phase 5: Autoresearch run #${run_count} (${REMAINING}s remaining)"

    # Run train.py with 5-minute budget
    timeout 600 $AR_PYTHON train.py > "$RUN_LOG" 2>&1
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        VAL_BPB=$(grep "^val_bpb:" "$RUN_LOG" | awk '{print $2}')
        PEAK_VRAM=$(grep "^peak_vram_mb:" "$RUN_LOG" | awk '{print $2}')
        PEAK_GB=$(echo "scale=1; ${PEAK_VRAM:-0} / 1024" | bc 2>/dev/null || echo "0.0")
        log "Phase 5: Run #${run_count} — val_bpb=${VAL_BPB}, vram=${PEAK_GB}GB"
        printf "${run_count}\t${VAL_BPB}\t${PEAK_GB}\tkeep\tbaseline run #${run_count}\n" >> results.tsv
    elif [[ $EXIT_CODE -eq 124 ]]; then
        log "Phase 5: Run #${run_count} — TIMEOUT (killed after 600s)"
        printf "${run_count}\t0.0\t0.0\tcrash\ttimeout\n" >> results.tsv
    else
        log "Phase 5: Run #${run_count} — FAILED (exit $EXIT_CODE, see $RUN_LOG)"
        printf "${run_count}\t0.0\t0.0\tcrash\texit code ${EXIT_CODE}\n" >> results.tsv
    fi

    # Copy results.tsv to overnight results
    cp results.tsv "$AR_RESULTS/results.tsv" 2>/dev/null || true
done

cd "$BASE_DIR"

# ── Phase 6: Final autoresearch summary plot ──────────────────────────────

log "Phase 6: START — Autoresearch summary plot"
$BIMODAL_PYTHON -c "
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from analysis.plots import setup_style, COLORS

setup_style()

# Read results.tsv
runs, bpbs, statuses = [], [], []
try:
    with open('$AR_RESULTS/results.tsv') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                runs.append(int(parts[0]))
                bpbs.append(float(parts[1]))
                statuses.append(parts[3])
except FileNotFoundError:
    print('No autoresearch results found')
    sys.exit(0)

if not runs:
    print('No autoresearch runs to plot')
    sys.exit(0)

fig, ax = plt.subplots(figsize=(10, 5))

# Color by status
for i, (r, b, s) in enumerate(zip(runs, bpbs, statuses)):
    if s == 'keep' and b > 0:
        ax.plot(r, b, 'o', color=COLORS['primary'], ms=8, zorder=5)
    elif s == 'crash' or b == 0:
        ax.plot(r, b if b > 0 else None, 'x', color=COLORS['base'], ms=8, zorder=5)

# Connect valid points
valid = [(r, b) for r, b, s in zip(runs, bpbs, statuses) if s == 'keep' and b > 0]
if valid:
    vr, vb = zip(*valid)
    ax.plot(vr, vb, '-', color=COLORS['primary'], lw=1, alpha=0.5)

    # Best line
    best_bpb = min(vb)
    ax.axhline(best_bpb, color=COLORS['accent'], ls='--', lw=1, alpha=0.7,
               label=f'Best: {best_bpb:.6f}')

    # Running best
    running_best = []
    current_best = float('inf')
    for r, b in zip(vr, vb):
        current_best = min(current_best, b)
        running_best.append(current_best)
    ax.plot(vr, running_best, '-', color=COLORS['accent'], lw=2, alpha=0.8,
            label='Running best')

ax.set_xlabel('Run #')
ax.set_ylabel('val_bpb')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, lw=0.6)
ax.set_title('AutoResearch: val_bpb Over Experiments', fontsize=12)
fig.tight_layout()
fig.savefig('$RESULTS_DIR/autoresearch_progress.png')
fig.savefig('$BASE_DIR/plots/autoresearch_progress.png')
print('Autoresearch progress plot saved')
" > "$RESULTS_DIR/phase6_ar_plot.log" 2>&1 && {
    log "Phase 6: DONE — autoresearch progress plot saved"
} || {
    log "Phase 6: FAILED — see $RESULTS_DIR/phase6_ar_plot.log"
}

# ── Summary ──────────────────────────────────────────────────────────────

log "=== OVERNIGHT JOB COMPLETE ==="
log "Total time: $(( ($(date +%s) - JOB_START_EPOCH) / 60 )) minutes"
log "Results: $RESULTS_DIR/"
log "Plots: $BASE_DIR/plots/"
log ""
log "Phase results:"
for f in "$RESULTS_DIR"/phase*_eval_*/metrics.json; do
    if [[ -f "$f" ]]; then
        label=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('label','?'), '— MAE:', d.get('mae','?'))" 2>/dev/null || echo "$f")
        log "  $label"
    fi
done
log "Autoresearch runs: $run_count"
if [[ -f "$AR_RESULTS/results.tsv" ]]; then
    BEST_BPB=$(awk -F'\t' 'NR>1 && $2>0 {if(!best || $2<best) best=$2} END{print best}' "$AR_RESULTS/results.tsv")
    log "Autoresearch best val_bpb: $BEST_BPB"
fi
