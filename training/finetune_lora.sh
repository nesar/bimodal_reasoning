#!/bin/bash
# finetune_lora.sh — LoRA fine-tuning for large models (gpt-oss-120b scale)
#
# For 120B-scale models, use DeepSpeed Zero-3 or FSDP.
# Default here: DeepSpeed Zero-3 with gradient checkpointing.
#
# Usage:
#   bash finetune_lora.sh \
#       --model_name_or_path gpt-oss-120b \
#       --dataset_path /path/to/spec_text2text \
#       --output_model_path /path/to/output \
#       [--learning_rate 1e-4] \
#       [--lora_r 8] \
#       [--num_train_epochs 2] \
#       [--block_size 512]

# --- Defaults ---
model_name_or_path="gpt-oss-120b"
dataset_path="data/datasets/spec_text2text"
output_dir="output_models/finetuned_lora"
learning_rate="1e-4"
lora_r=8
num_train_epochs=2
block_size=512
batch_size=1
deepspeed_args="--master_port=11002"

# --- Argument parsing ---
while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"; shift ;;
    -d|--dataset_path)
      dataset_path="$2"; shift ;;
    -o|--output_model_path)
      output_dir="$2"; shift ;;
    --learning_rate)
      learning_rate="$2"; shift ;;
    --lora_r)
      lora_r="$2"; shift ;;
    --num_train_epochs)
      num_train_epochs="$2"; shift ;;
    --block_size)
      block_size="$2"; shift ;;
    --deepspeed_args)
      deepspeed_args="$2"; shift ;;
    *)
      echo "Unknown option: ${key}" 1>&2; exit 1 ;;
  esac
  shift
done

# --- Paths ---
script_dir=$(cd "$(dirname "$0")"; pwd)
project_dir=$(dirname "$script_dir")
log_dir="${project_dir}/logs/finetune_lora"
mkdir -p "${output_dir}" "${log_dir}"

echo "=============================="
echo "Fine-tuning with LoRA"
echo "  Model:        ${model_name_or_path}"
echo "  Dataset:      ${dataset_path}"
echo "  Output:       ${output_dir}"
echo "  LR:           ${learning_rate}"
echo "  LoRA rank:    ${lora_r}"
echo "  Epochs:       ${num_train_epochs}"
echo "  Block size:   ${block_size}"
echo "=============================="

# Locate the LMFlow finetune script (or use a direct HF trainer script).
# If using LMFlow, point to examples/finetune.py from the LMFlow directory.
# If using HF Trainer directly, swap deepspeed for accelerate below.
LMFLOW_DIR="${project_dir}/../LMFlow"

if [[ -d "${LMFLOW_DIR}" ]]; then
    echo "Using LMFlow at ${LMFLOW_DIR}"
    cd "${LMFLOW_DIR}"

    deepspeed ${deepspeed_args} \
      examples/finetune.py \
        --model_name_or_path "${model_name_or_path}" \
        --dataset_path "${dataset_path}" \
        --output_dir "${output_dir}" --overwrite_output_dir \
        --num_train_epochs "${num_train_epochs}" \
        --learning_rate "${learning_rate}" \
        --block_size "${block_size}" \
        --per_device_train_batch_size "${batch_size}" \
        --use_lora 1 \
        --lora_r "${lora_r}" \
        --save_aggregated_lora 1 \
        --deepspeed configs/ds_config_zero3.json \
        --bf16 \
        --gradient_checkpointing True \
        --run_name finetune_lora \
        --validation_split_percentage 0 \
        --logging_steps 20 \
        --do_train \
        --ddp_timeout 72000 \
        --save_steps 5000 \
        --dataloader_num_workers 1 \
        | tee "${log_dir}/train.log" \
        2> "${log_dir}/train.err"
else
    echo "LMFlow not found at ${LMFLOW_DIR}."
    echo "Falling back to direct HuggingFace Trainer + FSDP via accelerate."
    echo "Please ensure accelerate is configured (accelerate config)."

    accelerate launch \
        --config_file "${project_dir}/configs/accelerate_fsdp.yaml" \
        --main_process_port 11002 \
        "${project_dir}/training/finetune_hf.py" \
          --model_name_or_path "${model_name_or_path}" \
          --dataset_path "${dataset_path}" \
          --output_dir "${output_dir}" \
          --num_train_epochs "${num_train_epochs}" \
          --learning_rate "${learning_rate}" \
          --block_size "${block_size}" \
          --per_device_train_batch_size "${batch_size}" \
          --lora_r "${lora_r}" \
          --bf16 \
          --gradient_checkpointing \
          --logging_steps 20 \
          --save_steps 5000 \
        | tee "${log_dir}/train.log" \
        2> "${log_dir}/train.err"
fi

echo "Fine-tuning done. Output at: ${output_dir}"
