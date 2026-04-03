#!/bin/bash
# run_experiment_template.sh — Single experiment orchestration
#
# This template is instantiated by generate_experiments.py.
# Placeholders {{CONFIG_PATH}} and {{BASE_DIR}} are replaced at generation time.

trap 'echo "ERROR: Command failed with exit code $? at line $LINENO"; exit 1' ERR
set -o pipefail
exec > >(tee "${EXPERIMENT_DIR:-/tmp}/experiment_log_$(date +%Y%m%d_%H%M%S).log") 2>&1
echo "Script started at $(date)"

# --- Paths ---
CONFIG_PATH="${CONFIG_PATH:-{{CONFIG_PATH}}}"
BASE_DIR="${BASE_DIR:-{{BASE_DIR}}}"

[[ -f "$CONFIG_PATH" ]] || { echo "ERROR: Config not found: $CONFIG_PATH"; exit 1; }
[[ -d "$BASE_DIR" ]]    || { echo "ERROR: Base dir not found: $BASE_DIR"; exit 1; }

EXPERIMENT_DIR=$(dirname "$CONFIG_PATH")
echo "Experiment directory: $EXPERIMENT_DIR"

mkdir -p "$EXPERIMENT_DIR/models" "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/plots" "$EXPERIMENT_DIR/results"

command -v jq &>/dev/null || { echo "ERROR: jq not installed"; exit 1; }

# --- Extract parameters ---
extract_param() {
    local param=$1 default=$2
    local value
    value=$(jq -r ".$param // \"$default\"" "$CONFIG_PATH")
    [[ "$value" == "null" && -z "$default" ]] && { echo "ERROR: Required param '$param' missing"; exit 1; }
    echo "$value"
}

PRETRAINED_MODEL=$(extract_param "pretrained_model")
DATASET_PATH=$(extract_param "dataset_path")
OUTPUT_MODEL_PATH="$EXPERIMENT_DIR/models/finetuned_model"
LEARNING_RATE=$(extract_param "learning_rate" "1e-4")
LORA_R=$(extract_param "lora_r" "8")
NUM_TRAIN_EPOCHS=$(extract_param "num_train_epochs" "2")
TRAINING_SAMPLES=$(extract_param "training_samples" "10000")
BENCHMARK_TASKS=$(extract_param "benchmark_tasks" "mmlu_college_physics")
BATCH_SIZE=$(extract_param "batch_size" "1")
NUM_FEWSHOT=$(extract_param "num_fewshot" "2")
TOKENIZATION_STRATEGY=$(extract_param "tokenization_strategy" "digit_base10")

echo "===================================================================="
echo "EXPERIMENT CONFIG"
echo "  Model:               $PRETRAINED_MODEL"
echo "  Dataset:             $DATASET_PATH"
echo "  Output:              $OUTPUT_MODEL_PATH"
echo "  Learning Rate:       $LEARNING_RATE"
echo "  LoRA Rank:           $LORA_R"
echo "  Epochs:              $NUM_TRAIN_EPOCHS"
echo "  Training Samples:    $TRAINING_SAMPLES"
echo "  Tokenization:        $TOKENIZATION_STRATEGY"
echo "  Benchmark Tasks:     $BENCHMARK_TASKS"
echo "===================================================================="

# --- Environment ---
export HF_DATASETS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMEXPR_MAX_THREADS=128
export OMP_NUM_THREADS=8

# --- Dataset preparation ---
if [ "$TRAINING_SAMPLES" != "10000" ]; then
    TEMP_DATASET_PATH="$EXPERIMENT_DIR/dataset_${TRAINING_SAMPLES}"
    mkdir -p "$TEMP_DATASET_PATH"
    echo "Creating limited dataset ($TRAINING_SAMPLES samples)..."

    python - <<PYEOF
import json, os, sys
src = '$DATASET_PATH/text2text.json'
dst = '$TEMP_DATASET_PATH/text2text.json'
if not os.path.exists(src):
    print(f"ERROR: Dataset not found at {src}"); sys.exit(1)
with open(src) as f:
    data = json.load(f)
n = min(int('$TRAINING_SAMPLES'), len(data['instances']))
data['instances'] = data['instances'][:n]
with open(dst, 'w') as f:
    json.dump(data, f)
print(f"Created dataset with {n} samples at {dst}")
PYEOF

    DATASET_PATH="$TEMP_DATASET_PATH"
fi

# --- GPU helpers ---
clear_gpu() {
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# =========================================================
# Phase 1: Fine-tuning
# =========================================================
echo "### Phase 1: Fine-tuning ###"

bash "$BASE_DIR/training/finetune_lora.sh" \
  --model_name_or_path "$PRETRAINED_MODEL" \
  --dataset_path "$DATASET_PATH" \
  --output_model_path "$OUTPUT_MODEL_PATH" \
  --learning_rate "$LEARNING_RATE" \
  --lora_r "$LORA_R" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS"

clear_gpu

# =========================================================
# Phase 2: Domain evaluation (redshift prediction)
# =========================================================
echo "### Phase 2: Domain Evaluation ###"
EVAL_SCRIPT="$BASE_DIR/eval/redshift_eval.py"
PLOTS_DIR="$EXPERIMENT_DIR/plots"
mkdir -p "$PLOTS_DIR"

if [[ -f "$EVAL_SCRIPT" ]]; then
    echo "Pre-FT evaluation (base model)..."
    python "$EVAL_SCRIPT" false --output-dir "$PLOTS_DIR"
    clear_gpu

    if [[ -d "$OUTPUT_MODEL_PATH" ]]; then
        echo "Post-FT evaluation (fine-tuned model)..."
        python "$EVAL_SCRIPT" "$OUTPUT_MODEL_PATH" --output-dir "$PLOTS_DIR"
        clear_gpu
    else
        echo "WARNING: Fine-tuned model not found, skipping post-FT eval."
    fi
else
    echo "WARNING: Eval script not found at $EVAL_SCRIPT"
fi

# =========================================================
# Phase 3: LM eval harness benchmarks
# =========================================================
echo "### Phase 3: LM Eval Harness ###"

echo "Benchmarking base model..."
bash "$BASE_DIR/eval/lm_harness_eval.sh" \
  --model "$PRETRAINED_MODEL" \
  --output-dir "$PLOTS_DIR/eval_hf" \
  --tasks "$BENCHMARK_TASKS" \
  --batch-size "$BATCH_SIZE" \
  --num-fewshot "$NUM_FEWSHOT" || echo "WARNING: Base model benchmark failed"

clear_gpu

if [[ -d "$OUTPUT_MODEL_PATH" ]]; then
    echo "Benchmarking fine-tuned model..."
    bash "$BASE_DIR/eval/lm_harness_eval.sh" \
      --model "$OUTPUT_MODEL_PATH" \
      --output-dir "$PLOTS_DIR/eval_local" \
      --tasks "$BENCHMARK_TASKS" \
      --batch-size "$BATCH_SIZE" \
      --num-fewshot "$NUM_FEWSHOT" || echo "WARNING: Fine-tuned model benchmark failed"
    clear_gpu
fi

# =========================================================
# Phase 4: Collect metrics
# =========================================================
echo "### Phase 4: Collecting Metrics ###"

python "$BASE_DIR/analysis/benchmark_extraction.py" "$EXPERIMENT_DIR" || echo "WARNING: Metric extraction failed"

python - <<PYEOF
import os, json, re, glob, numpy as np
from datetime import datetime

experiment_dir = "$EXPERIMENT_DIR"
plots_dir = "$PLOTS_DIR"
results_dir = os.path.join(experiment_dir, "results")
os.makedirs(results_dir, exist_ok=True)

params = {
    "learning_rate": float("$LEARNING_RATE"),
    "lora_r": int("$LORA_R"),
    "num_train_epochs": float("$NUM_TRAIN_EPOCHS"),
    "training_samples": int("$TRAINING_SAMPLES"),
    "tokenization_strategy": "$TOKENIZATION_STRATEGY",
    "pretrained_model": "$PRETRAINED_MODEL",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Load benchmark extraction results if available
metrics_file = os.path.join(results_dir, "metrics.json")
if os.path.exists(metrics_file):
    with open(metrics_file) as f:
        existing = json.load(f)
    existing['parameters'] = params
else:
    existing = {"parameters": params, "metrics": {}, "benchmarks": {}}

with open(metrics_file, 'w') as f:
    json.dump(existing, f, indent=2)

print(f"Metrics saved to {metrics_file}")
PYEOF

echo "===================================================================="
echo "EXPERIMENT COMPLETED at $(date)"
echo "Results: $EXPERIMENT_DIR"
echo "===================================================================="
