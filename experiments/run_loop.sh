#!/bin/bash
# run_loop.sh — Autoresearch experiment loop for fine-tuning hyperparameter search
# Runs experiments sequentially, logs to results.tsv, generates money plot after each.

set -uo pipefail
export TMPDIR=/tmp
export HF_HOME=/lcrc/project/cosmo_ai/nramachandra/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_DIR="/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation/bimodal_reasoning"
PYTHON="/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python"
RESULTS="$BASE_DIR/experiments/autoresearch_runs/results.tsv"

cd "$BASE_DIR"

run_exp() {
    local exp_num="$1"
    local desc="$2"
    shift 2
    local args="$@"

    echo "[$(date '+%H:%M:%S')] Experiment #${exp_num}: ${desc}"
    local log="/tmp/exp_$(printf '%03d' $exp_num).log"

    $PYTHON experiments/run_experiment.py --output_dir "/tmp/exp_$(printf '%03d' $exp_num)" $args > "$log" 2>&1
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        local mae=$(grep "^mae:" "$log" | awk '{print $2}')
        local loss=$(grep "^train_loss:" "$log" | awk '{print $2}')
        echo "  Result: MAE=${mae}, loss=${loss}"
        echo "$mae"
    else
        echo "  CRASHED (exit $exit_code)"
        echo "crash"
    fi
}

# Get current best MAE from results.tsv
best_mae() {
    awk -F'\t' 'NR>1 && $4=="keep" {if(!best || $2<best) best=$2} END{print best+0}' "$RESULTS"
}

log_result() {
    local num="$1" mae="$2" loss="$3" status="$4" desc="$5"
    printf "%s\t%s\t%s\t%s\t%s\n" "$num" "$mae" "$loss" "$status" "$desc" >> "$RESULTS"
}

update_plot() {
    $PYTHON analysis/plot_autoresearch_money.py \
        --results "$RESULTS" \
        --output plots/autoresearch_money.png 2>/dev/null
}

# ── Experiments ──────────────────────────────────────────────────────────

CURRENT_BEST=$(best_mae)
echo "Starting from best MAE: $CURRENT_BEST"

# Exp 2: higher LR
mae=$(run_exp 2 "lr 1e-4 -> 5e-4" --lr 5e-4 --lora_r 8)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 2 "$mae" "$(grep '^train_loss:' /tmp/exp_002.log | awk '{print $2}')" "keep" "lr 1e-4 -> 5e-4"
        CURRENT_BEST="$mae"
    else
        log_result 2 "$mae" "$(grep '^train_loss:' /tmp/exp_002.log | awk '{print $2}')" "discard" "lr 1e-4 -> 5e-4"
    fi
fi
update_plot

# Exp 3: higher LoRA rank
mae=$(run_exp 3 "lora_r 8 -> 16, alpha 16 -> 32" --lr 1e-4 --lora_r 16 --lora_alpha 32)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 3 "$mae" "$(grep '^train_loss:' /tmp/exp_003.log | awk '{print $2}')" "keep" "lora_r 8 -> 16, alpha 16 -> 32"
        CURRENT_BEST="$mae"
    else
        log_result 3 "$mae" "$(grep '^train_loss:' /tmp/exp_003.log | awk '{print $2}')" "discard" "lora_r 8 -> 16, alpha 16 -> 32"
    fi
fi
update_plot

# Exp 4: lower LoRA rank
mae=$(run_exp 4 "lora_r 8 -> 4, alpha 16 -> 8" --lr 1e-4 --lora_r 4 --lora_alpha 8)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 4 "$mae" "$(grep '^train_loss:' /tmp/exp_004.log | awk '{print $2}')" "keep" "lora_r 8 -> 4, alpha 16 -> 8"
        CURRENT_BEST="$mae"
    else
        log_result 4 "$mae" "$(grep '^train_loss:' /tmp/exp_004.log | awk '{print $2}')" "discard" "lora_r 8 -> 4, alpha 16 -> 8"
    fi
fi
update_plot

# Exp 5: larger effective batch (grad_accum 8 -> 16)
mae=$(run_exp 5 "grad_accum 8 -> 16" --lr 1e-4 --lora_r 8 --grad_accum 16)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 5 "$mae" "$(grep '^train_loss:' /tmp/exp_005.log | awk '{print $2}')" "keep" "grad_accum 8 -> 16"
        CURRENT_BEST="$mae"
    else
        log_result 5 "$mae" "$(grep '^train_loss:' /tmp/exp_005.log | awk '{print $2}')" "discard" "grad_accum 8 -> 16"
    fi
fi
update_plot

# Exp 6: smaller effective batch (grad_accum 8 -> 4)
mae=$(run_exp 6 "grad_accum 8 -> 4" --lr 1e-4 --lora_r 8 --grad_accum 4)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 6 "$mae" "$(grep '^train_loss:' /tmp/exp_006.log | awk '{print $2}')" "keep" "grad_accum 8 -> 4"
        CURRENT_BEST="$mae"
    else
        log_result 6 "$mae" "$(grep '^train_loss:' /tmp/exp_006.log | awk '{print $2}')" "discard" "grad_accum 8 -> 4"
    fi
fi
update_plot

# Exp 7: longer training (200 steps)
mae=$(run_exp 7 "max_steps 100 -> 200" --lr 1e-4 --lora_r 8 --max_steps 200)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 7 "$mae" "$(grep '^train_loss:' /tmp/exp_007.log | awk '{print $2}')" "keep" "max_steps 100 -> 200"
        CURRENT_BEST="$mae"
    else
        log_result 7 "$mae" "$(grep '^train_loss:' /tmp/exp_007.log | awk '{print $2}')" "discard" "max_steps 100 -> 200"
    fi
fi
update_plot

# Exp 8: more warmup
mae=$(run_exp 8 "warmup_ratio 0.05 -> 0.15" --lr 1e-4 --lora_r 8 --warmup_ratio 0.15)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 8 "$mae" "$(grep '^train_loss:' /tmp/exp_008.log | awk '{print $2}')" "keep" "warmup_ratio 0.05 -> 0.15"
        CURRENT_BEST="$mae"
    else
        log_result 8 "$mae" "$(grep '^train_loss:' /tmp/exp_008.log | awk '{print $2}')" "discard" "warmup_ratio 0.05 -> 0.15"
    fi
fi
update_plot

# Exp 9: more dropout
mae=$(run_exp 9 "lora_dropout 0.05 -> 0.1" --lr 1e-4 --lora_r 8 --lora_dropout 0.1)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 9 "$mae" "$(grep '^train_loss:' /tmp/exp_009.log | awk '{print $2}')" "keep" "lora_dropout 0.05 -> 0.1"
        CURRENT_BEST="$mae"
    else
        log_result 9 "$mae" "$(grep '^train_loss:' /tmp/exp_009.log | awk '{print $2}')" "discard" "lora_dropout 0.05 -> 0.1"
    fi
fi
update_plot

# Exp 10: lr=2e-4 (between baseline and 5e-4)
mae=$(run_exp 10 "lr 1e-4 -> 2e-4" --lr 2e-4 --lora_r 8)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 10 "$mae" "$(grep '^train_loss:' /tmp/exp_010.log | awk '{print $2}')" "keep" "lr 1e-4 -> 2e-4"
        CURRENT_BEST="$mae"
    else
        log_result 10 "$mae" "$(grep '^train_loss:' /tmp/exp_010.log | awk '{print $2}')" "discard" "lr 1e-4 -> 2e-4"
    fi
fi
update_plot

# Exp 11: 300 steps (even longer)
mae=$(run_exp 11 "max_steps 100 -> 300" --lr 1e-4 --lora_r 8 --max_steps 300)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 11 "$mae" "$(grep '^train_loss:' /tmp/exp_011.log | awk '{print $2}')" "keep" "max_steps 100 -> 300"
        CURRENT_BEST="$mae"
    else
        log_result 11 "$mae" "$(grep '^train_loss:' /tmp/exp_011.log | awk '{print $2}')" "discard" "max_steps 100 -> 300"
    fi
fi
update_plot

# Exp 12: r=16 with 200 steps (combine rank + steps if both helped)
mae=$(run_exp 12 "r=16 + 200 steps" --lr 1e-4 --lora_r 16 --lora_alpha 32 --max_steps 200)
if [[ "$mae" != "crash" ]]; then
    cmp=$(echo "$mae < $CURRENT_BEST" | bc -l 2>/dev/null || echo 0)
    if [[ "$cmp" == "1" ]]; then
        log_result 12 "$mae" "$(grep '^train_loss:' /tmp/exp_012.log | awk '{print $2}')" "keep" "r=16 + 200 steps"
        CURRENT_BEST="$mae"
    else
        log_result 12 "$mae" "$(grep '^train_loss:' /tmp/exp_012.log | awk '{print $2}')" "discard" "r=16 + 200 steps"
    fi
fi
update_plot

echo ""
echo "=== LOOP COMPLETE ==="
echo "Best MAE: $CURRENT_BEST"
cat "$RESULTS"
