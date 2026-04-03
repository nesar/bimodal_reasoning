#!/bin/bash
# lm_harness_eval.sh — Run LM eval harness benchmarks on base or fine-tuned model.
#
# Requires: lm-evaluation-harness (pip install lm-eval)
#
# Usage:
#   bash lm_harness_eval.sh \
#       --model gpt-oss-120b \
#       --output-dir /path/to/eval_results \
#       [--tasks mmlu_college_physics,leaderboard_gpqa] \
#       [--batch-size 1] \
#       [--num-fewshot 2]

MODEL="gpt-oss-120b"
OUTPUT_DIR="./eval_results"
TASKS="mmlu_college_physics,mmlu_high_school_physics,leaderboard_bbh,leaderboard_gpqa,leaderboard_math_hard,astro_mlab_araa_mcq_gemini15"
BATCH_SIZE=1
NUM_FEWSHOT=2
TENSOR_PARALLEL=4

while [[ $# -ge 1 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift ;;
    --tasks) TASKS="$2"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift ;;
    --num-fewshot) NUM_FEWSHOT="$2"; shift ;;
    --tensor-parallel) TENSOR_PARALLEL="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "${OUTPUT_DIR}"

echo "=============================="
echo "LM Eval Harness"
echo "  Model:      ${MODEL}"
echo "  Tasks:      ${TASKS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Few-shot:   ${NUM_FEWSHOT}"
echo "=============================="

export HF_DATASETS_TRUST_REMOTE_CODE=1
RANDOM_PORT=$((10000 + RANDOM % 50000))

# Use vLLM backend for large models (much faster than HF pipeline for inference)
lm_eval \
  --model vllm \
  --model_args "pretrained=${MODEL},trust_remote_code=True,tensor_parallel_size=${TENSOR_PARALLEL},gpu_memory_utilization=0.8,dtype=bfloat16" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_fewshot "${NUM_FEWSHOT}" \
  --log_samples \
  --output_path "${OUTPUT_DIR}"

echo "Eval complete. Results in ${OUTPUT_DIR}"
