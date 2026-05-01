#!/bin/bash
# run_pareto_overnight.sh — Multi-objective autoresearch sweep (20B + dual objective).
#
# Runs experiments/pareto_loop.py with environment set up for the lm-eval harness
# (PYTHONPATH for the solitons install, LD_LIBRARY_PATH for the pip-installed
# nvidia libs). Each trial trains a 20B LoRA, measures redshift MAE, runs the
# fast lm-eval task subset, and logs both objectives. The Pareto front and the
# money plot update after every trial, so progress is visible mid-run.
#
# Usage:
#   nohup bash experiments/run_pareto_overnight.sh [n_trials] > pareto.log 2>&1 &
#   tail -f pareto.log
#
# Default n_trials = 50 (~10 hours at ~12 min/trial on 8×A100-80GB).

set -uo pipefail

N_TRIALS="${1:-50}"

export TMPDIR=/tmp
export HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/lcrc/project/solitons/nramachandra/lm_eval_pkg:${PYTHONPATH:-}

NVIDIA_LIB_ROOT="/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/lib/python3.11/site-packages/nvidia"
for d in "$NVIDIA_LIB_ROOT"/*/lib; do
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

BASE_DIR="/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/bimodal_reasoning"
PYTHON="/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python"
OUT_DIR="$BASE_DIR/experiments/autoresearch_runs/pareto"
PLOT="$BASE_DIR/plots/autoresearch_pareto.png"

cd "$BASE_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pareto sweep: n_trials=$N_TRIALS, out=$OUT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $(nvidia-smi -L 2>/dev/null | wc -l) GPUs detected"

$PYTHON experiments/pareto_loop.py \
    --n-trials   "$N_TRIALS" \
    --output-dir "$OUT_DIR" \
    --plot       "$PLOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pareto sweep done."
echo "Results: $OUT_DIR/results.jsonl"
echo "Plot:    $PLOT"
