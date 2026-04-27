#!/bin/bash
# run_benchmarks_120b_only.sh — Run lm-eval on 120B base + FT only.
#
# 20B benchmarks are already in overnight_results/latest/. This script just
# adds the 120B half (which previously OOM'd with max_memory_per_gpu=50GiB —
# 120B needs a lower cap so weights spread across all 8 GPUs and leave
# headroom for the 18-19 GiB eager-attention activation per layer).
#
# Usage: nohup bash run_benchmarks_120b_only.sh > benchmark_120b.log 2>&1 &

set -uo pipefail

export TMPDIR=/tmp
export HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/lcrc/project/solitons/nramachandra/lm_eval_pkg:${PYTHONPATH:-}

NVIDIA_LIB_ROOT="/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/lib/python3.11/site-packages/nvidia"
for d in "$NVIDIA_LIB_ROOT"/*/lib; do
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done

BASE_DIR="/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/bimodal_reasoning"
LM_EVAL="/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python -m lm_eval"
RESULTS_DIR="$BASE_DIR/overnight_results/latest"
ADAPTER_120B="$BASE_DIR/output_models/gpt-oss-120b_structured"
TASKS="mmlu_college_physics,mmlu_high_school_physics,mmlu_astronomy,leaderboard_gpqa,leaderboard_bbh"

cd "$BASE_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESULTS_DIR/STATUS"; }

log "120B benchmarks: detected $(nvidia-smi -L 2>/dev/null | wc -l) GPUs, max_memory_per_gpu=35GiB"

# Clear stale (empty) result dirs so lm_eval writes fresh files.
rm -rf "$RESULTS_DIR/benchmark_base_120b" "$RESULTS_DIR/benchmark_ft_120b_structured"

# ── Benchmark: Base 120B ─────────────────────────────────────────────────
log "Benchmark: START — Base gpt-oss-120b"
mkdir -p "$RESULTS_DIR/benchmark_base_120b"

$LM_EVAL \
    --model hf \
    --model_args "pretrained=openai/gpt-oss-120b,trust_remote_code=True,dtype=bfloat16,parallelize=True,max_memory_per_gpu=35GiB,attn_implementation=eager" \
    --tasks "$TASKS" \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path "$RESULTS_DIR/benchmark_base_120b" \
    > "$RESULTS_DIR/benchmark_base_120b.log" 2>&1 && {
    log "Benchmark: DONE — Base 120b"
} || {
    log "Benchmark: FAILED — Base 120b, see benchmark_base_120b.log"
}

# ── Benchmark: FT 120B (structured adapter) ──────────────────────────────
if [[ -d "$ADAPTER_120B" ]]; then
    log "Benchmark: START — Fine-tuned gpt-oss-120b (structured)"
    mkdir -p "$RESULTS_DIR/benchmark_ft_120b_structured"

    $LM_EVAL \
        --model hf \
        --model_args "pretrained=openai/gpt-oss-120b,peft=$ADAPTER_120B,trust_remote_code=True,dtype=bfloat16,parallelize=True,max_memory_per_gpu=35GiB,attn_implementation=eager" \
        --tasks "$TASKS" \
        --batch_size 1 \
        --num_fewshot 0 \
        --output_path "$RESULTS_DIR/benchmark_ft_120b_structured" \
        > "$RESULTS_DIR/benchmark_ft_120b_structured.log" 2>&1 && {
        log "Benchmark: DONE — Fine-tuned 120b"
    } || {
        log "Benchmark: FAILED — Fine-tuned 120b, see benchmark_ft_120b_structured.log"
    }
else
    log "Benchmark: SKIPPED — No structured adapter found at $ADAPTER_120B"
fi

# ── Comparison + plots (4 models: base/ft × 20b/120b) ────────────────────
log "Benchmark: Generating 4-model comparison"
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

model_runs = [
    ('base_20b',   'benchmark_base_20b'),
    ('ft_20b',     'benchmark_ft_20b_compact'),
    ('base_120b',  'benchmark_base_120b'),
    ('ft_120b',    'benchmark_ft_120b_structured'),
]

all_scores = {}
for label, subdir in model_runs:
    d = os.path.join(results_dir, subdir)
    if os.path.isdir(d):
        s = extract_scores(d)
        if s:
            all_scores[label] = s

if len(all_scores) >= 2:
    all_tasks = sorted(set().union(*all_scores.values()))
    comparison = {t: {label: scores.get(t) for label, scores in all_scores.items()}
                  for t in all_tasks}
    with open(os.path.join(results_dir, 'benchmark_comparison.json'), 'w') as fh:
        json.dump({'tasks': comparison}, fh, indent=2)

    header = f\"{'task':30s}\" + ''.join(f'  {label:>10s}' for label in all_scores)
    print(header)
    print('-' * len(header))
    for t in all_tasks:
        row = f'{t:30s}'
        for label in all_scores:
            v = all_scores[label].get(t)
            row += f'  {v:10.1f}' if v is not None else f'  {\"N/A\":>10s}'
        print(row)

    if 'base_20b' in all_scores and 'ft_20b' in all_scores:
        tasks_20b = sorted(set(all_scores['base_20b'].keys()) & set(all_scores['ft_20b'].keys()))
        plot_benchmark_comparison(tasks_20b,
            [all_scores['base_20b'][t] for t in tasks_20b],
            [all_scores['ft_20b'][t]   for t in tasks_20b],
            save_path=os.path.join(results_dir, 'benchmark_comparison_20b.png'))

    if 'base_120b' in all_scores and 'ft_120b' in all_scores:
        tasks_120b = sorted(set(all_scores['base_120b'].keys()) & set(all_scores['ft_120b'].keys()))
        plot_benchmark_comparison(tasks_120b,
            [all_scores['base_120b'][t] for t in tasks_120b],
            [all_scores['ft_120b'][t]   for t in tasks_120b],
            save_path=os.path.join(results_dir, 'benchmark_comparison_120b.png'))
else:
    print(f'Not enough results for comparison. Found: {list(all_scores.keys())}')
" > "$RESULTS_DIR/benchmark_summary_120b.log" 2>&1

cat "$RESULTS_DIR/benchmark_summary_120b.log" | tee -a "$RESULTS_DIR/STATUS"
log "120B benchmarks: ALL DONE"
