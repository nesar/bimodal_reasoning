#!/bin/bash
# run_benchmarks.sh — Run lm-eval harness benchmarks (base vs fine-tuned)
#
# Waits for the overnight job to finish, then runs benchmarks.
# Usage:
#   nohup bash run_benchmarks.sh <overnight_pid> > benchmark.log 2>&1 &

set -uo pipefail

OVERNIGHT_PID="${1:-0}"

export TMPDIR=/tmp
export HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/lcrc/project/solitons/nramachandra/lm_eval_pkg:${PYTHONPATH:-}

BASE_DIR="/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/bimodal_reasoning"
LM_EVAL="/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python -m lm_eval"
RESULTS_DIR="$BASE_DIR/overnight_results/latest"

cd "$BASE_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESULTS_DIR/STATUS"; }

# Wait for overnight job if PID given
if [[ $OVERNIGHT_PID -gt 0 ]]; then
    echo "Waiting for overnight job (PID $OVERNIGHT_PID) to finish..."
    while kill -0 "$OVERNIGHT_PID" 2>/dev/null; do
        sleep 30
    done
    echo "Overnight job finished. Starting benchmarks."
fi

TASKS="mmlu_college_physics,mmlu_high_school_physics,mmlu_astronomy,leaderboard_gpqa,leaderboard_bbh"
ADAPTER_COMPACT="$BASE_DIR/output_models/gpt-oss-20b_compact"

# ── Benchmark 1: Base model ──────────────────────────────────────────────

log "Benchmark: START — Base gpt-oss-20b"
mkdir -p "$RESULTS_DIR/benchmark_base_20b"

$LM_EVAL \
    --model hf \
    --model_args "pretrained=openai/gpt-oss-20b,trust_remote_code=True,dtype=bfloat16,device_map=auto,attn_implementation=eager" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path "$RESULTS_DIR/benchmark_base_20b" \
    > "$RESULTS_DIR/benchmark_base_20b.log" 2>&1 && {
    log "Benchmark: DONE — Base model results in benchmark_base_20b/"
} || {
    log "Benchmark: FAILED — Base model, see benchmark_base_20b.log"
}

# ── Benchmark 2: Fine-tuned model (compact) ──────────────────────────────

if [[ -d "$ADAPTER_COMPACT" ]]; then
    log "Benchmark: START — Fine-tuned gpt-oss-20b (compact)"
    mkdir -p "$RESULTS_DIR/benchmark_ft_20b_compact"

    $LM_EVAL \
        --model hf \
        --model_args "pretrained=openai/gpt-oss-20b,peft=$ADAPTER_COMPACT,trust_remote_code=True,dtype=bfloat16,device_map=auto,attn_implementation=eager" \
        --tasks "$TASKS" \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path "$RESULTS_DIR/benchmark_ft_20b_compact" \
        > "$RESULTS_DIR/benchmark_ft_20b_compact.log" 2>&1 && {
        log "Benchmark: DONE — Fine-tuned model results in benchmark_ft_20b_compact/"
    } || {
        log "Benchmark: FAILED — Fine-tuned model, see benchmark_ft_20b_compact.log"
    }
else
    log "Benchmark: SKIPPED — No compact adapter found at $ADAPTER_COMPACT"
fi

# ── Summary: extract and compare scores ──────────────────────────────────

log "Benchmark: Generating comparison"
/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python -c "
import json, glob, os, sys
sys.path.insert(0, '.')
from analysis.plots import setup_style, plot_benchmark_comparison
import matplotlib; matplotlib.use('Agg')

results_dir = '$RESULTS_DIR'

def extract_scores(benchmark_dir):
    scores = {}
    for f in glob.glob(os.path.join(benchmark_dir, '**', 'results_*.json'), recursive=True):
        with open(f) as fh:
            data = json.load(fh)
        for task_name, task_data in data.get('results', {}).items():
            acc = task_data.get('acc,none', task_data.get('acc_norm,none', task_data.get('exact_match,none')))
            if acc is not None:
                short_name = task_name.replace('mmlu_', '').replace('leaderboard_', '')
                scores[short_name] = round(float(acc) * 100, 1)
    return scores

base_scores = extract_scores(os.path.join(results_dir, 'benchmark_base_20b'))
ft_scores = extract_scores(os.path.join(results_dir, 'benchmark_ft_20b_compact'))

if base_scores and ft_scores:
    tasks = sorted(set(base_scores.keys()) & set(ft_scores.keys()))
    base_vals = [base_scores[t] for t in tasks]
    ft_vals = [ft_scores[t] for t in tasks]

    # Save comparison JSON
    comparison = {
        'tasks': {t: {'base': b, 'finetuned': f} for t, b, f in zip(tasks, base_vals, ft_vals)}
    }
    with open(os.path.join(results_dir, 'benchmark_comparison.json'), 'w') as fh:
        json.dump(comparison, fh, indent=2)

    # Plot
    fig = plot_benchmark_comparison(tasks, base_vals, ft_vals,
                                     save_path=os.path.join(results_dir, 'benchmark_comparison.png'))
    fig.savefig('plots/summary/benchmark_comparison.png')
    print(f'Benchmark comparison saved. Tasks: {len(tasks)}')
    for t, b, f in zip(tasks, base_vals, ft_vals):
        delta = f - b
        print(f'  {t:30s}  base={b:.1f}%  ft={f:.1f}%  delta={delta:+.1f}%')
else:
    print(f'Could not extract scores. base={len(base_scores)} ft={len(ft_scores)}')
" > "$RESULTS_DIR/benchmark_summary.log" 2>&1

cat "$RESULTS_DIR/benchmark_summary.log" | tee -a "$RESULTS_DIR/STATUS"

log "Benchmark: ALL DONE"
