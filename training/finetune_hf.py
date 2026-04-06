#!/usr/bin/env python3
"""
finetune_hf.py — LoRA fine-tuning using HuggingFace Trainer + PEFT.

Usage (device_map="auto", for models that fit in combined GPU memory):
    python training/finetune_hf.py \
        --model_name_or_path openai/gpt-oss-20b \
        --dataset_path data/datasets/structured_verbalization/text2text.json \
        --output_dir output_models/gpt-oss-20b_structured

Usage (DeepSpeed ZeRO-3, for 120B+ models across 8 GPUs):
    deepspeed --num_gpus 8 training/finetune_hf.py \
        --model_name_or_path openai/gpt-oss-120b \
        --dataset_path data/datasets/structured_verbalization/text2text.json \
        --output_dir output_models/gpt-oss-120b_structured \
        --deepspeed configs/ds_config_zero3.json
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import deepspeed
from transformers.integrations.deepspeed import HfDeepSpeedConfig


def load_text2text(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["instances"]


def tokenize_instances(instances, tokenizer, block_size):
    """Concatenate input + output into a single sequence with labels masked on the input portion."""
    input_ids_all = []
    labels_all = []

    for inst in instances:
        prompt = inst["input"] + "\n"
        completion = inst["output"]
        full_text = prompt + completion

        encoded = tokenizer(full_text, truncation=True, max_length=block_size)
        prompt_encoded = tokenizer(prompt, truncation=True, max_length=block_size)

        input_ids = encoded["input_ids"]
        labels = list(input_ids)
        # Mask the prompt portion so loss is only on the completion
        prompt_len = len(prompt_encoded["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        input_ids_all.append(input_ids)
        labels_all.append(labels)

    return Dataset.from_dict({"input_ids": input_ids_all, "labels": labels_all})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="openai/gpt-oss-20b")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="output_models/finetuned_lora")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Override num_train_epochs with a fixed step count (for short test runs)")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON (e.g. configs/ds_config_zero3.json)")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    use_deepspeed = args.deepspeed is not None

    if use_deepspeed:
        # Initialize distributed early so HfDeepSpeedConfig sees correct world_size
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset
    print(f"Loading dataset from {args.dataset_path} ...")
    instances = load_text2text(args.dataset_path)
    print(f"  {len(instances)} instances")
    dataset = tokenize_instances(instances, tokenizer, args.block_size)
    print(f"  Tokenized: {len(dataset)} sequences")

    # Load model — two paths:
    # 1. DeepSpeed ZeRO-3: use HfDeepSpeedConfig so from_pretrained partitions
    #    weights across ranks during loading (avoids 8x full-model CPU OOM)
    # 2. No DeepSpeed: use device_map="auto" to shard via accelerate
    n_gpus = torch.cuda.device_count()
    print(f"Loading model {args.model_name_or_path} ({n_gpus} GPUs, deepspeed={use_deepspeed}) ...")

    if use_deepspeed:
        # HfDeepSpeedConfig registers the DS config so from_pretrained detects
        # ZeRO-3 and uses init_empty_weights + shard-by-shard loading.
        # Must be created BEFORE from_pretrained; kept alive until Trainer init.
        _dschf = HfDeepSpeedConfig(args.deepspeed)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            attn_implementation="eager",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        max_mem = {i: "32GiB" for i in range(n_gpus)}
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            attn_implementation="eager",
            device_map="auto",
            max_memory=max_mem,
            low_cpu_mem_usage=True,
        )
    model.config.use_cache = False

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
    )
    if args.max_steps > 0:
        training_kwargs["max_steps"] = args.max_steps
    if use_deepspeed:
        training_kwargs["deepspeed"] = args.deepspeed
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting training ...")
    trainer.train()

    # Save LoRA adapter + tokenizer
    print(f"Saving model to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
