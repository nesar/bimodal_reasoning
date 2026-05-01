#!/usr/bin/env python3
"""
run_experiment.py — Single fine-tuning experiment for the autoresearch loop.

Trains gpt-oss-20b with given hyperparameters, evaluates redshift MAE,
and prints a summary line. Designed for rapid iteration (~10 min/run).

Usage:
    python experiments/run_experiment.py \
        --lr 1e-4 --lora_r 8 --max_steps 100 --eval_samples 50 \
        --output_dir /tmp/experiment_001
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Triton 3.6 (installed for lm-eval MXFP4 path) removed AttrsDescriptor that
# torch._inductor still imports via DeepSpeed → torch.compile chain. Patch
# before any torch import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import experiments.triton_compat  # noqa: F401

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)

from eval.redshift_eval_peft import trim_spectrum_to_fit, extract_redshift
from experiments.benchmark_adapter import run_lm_eval, aggregate, TASKS_FAST


def load_text2text(path):
    with open(path) as f:
        return json.load(f)["instances"]


def tokenize_instances(instances, tokenizer, block_size):
    input_ids_all, labels_all = [], []
    for inst in instances:
        prompt = inst["input"] + "\n"
        completion = inst["output"]
        full_text = prompt + completion
        encoded = tokenizer(full_text, truncation=True, max_length=block_size)
        prompt_encoded = tokenizer(prompt, truncation=True, max_length=block_size)
        input_ids = encoded["input_ids"]
        labels = list(input_ids)
        for i in range(min(len(prompt_encoded["input_ids"]), len(labels))):
            labels[i] = -100
        input_ids_all.append(input_ids)
        labels_all.append(labels)
    return Dataset.from_dict({"input_ids": input_ids_all, "labels": labels_all})


def train_and_eval(args):
    t0 = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize
    instances = load_text2text(args.dataset)
    dataset = tokenize_instances(instances, tokenizer, args.block_size)

    # Load model
    n_gpus = torch.cuda.device_count()
    max_mem = {i: "32GiB" for i in range(n_gpus)}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, attn_implementation="eager",
        device_map="auto", max_memory=max_mem, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True, logging_steps=args.max_steps,  # log only at end
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=collator)
    result = trainer.train()
    train_loss = result.training_loss
    train_time = time.time() - t0

    # Save adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Eval: reload with merged adapter
    del model, trainer
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, attn_implementation="eager",
        device_map="auto", max_memory=max_mem, low_cpu_mem_usage=True,
    )
    eval_model = PeftModel.from_pretrained(base_model, args.output_dir)
    eval_model = eval_model.merge_and_unload()
    eval_model.eval()

    # Run redshift eval
    eval_instances = load_text2text(args.dataset)[:args.eval_samples]
    z_true_list, z_pred_list = [], []
    for inst in eval_instances:
        prompt = trim_spectrum_to_fit(inst["input"] + "\n", tokenizer, 440)
        z_true = extract_redshift(inst["output"])
        z_true_list.append(z_true)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=440)
        ids = {k: v.to(eval_model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = eval_model.generate(**ids, max_new_tokens=60, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        z_pred_list.append(extract_redshift(gen))

    z_true = np.array(z_true_list)
    z_pred = np.array(z_pred_list)
    mask = np.isfinite(z_true) & np.isfinite(z_pred)
    n_valid = mask.sum()
    mae = float(np.mean(np.abs(z_true[mask] - z_pred[mask]))) if n_valid > 0 else float("inf")
    total_time = time.time() - t0

    # Optional: lm-eval harness benchmarks on the fine-tuned adapter.
    # Releases GPU memory from the eval model first so lm-eval can reload.
    bench_agg = {}
    if args.run_benchmarks:
        del base_model, eval_model
        torch.cuda.empty_cache()
        bench_dir = os.path.join(args.output_dir, "lm_eval")
        bench_scores = run_lm_eval(args.model, args.output_dir, TASKS_FAST, bench_dir)
        bench_agg = aggregate(bench_scores)

    # Print summary
    print("---")
    print(f"mae:            {mae:.6f}")
    print(f"n_valid:        {n_valid}/{len(z_true_list)}")
    print(f"train_loss:     {train_loss:.4f}")
    print(f"train_seconds:  {train_time:.0f}")
    print(f"total_seconds:  {total_time:.0f}")
    print(f"lr:             {args.lr}")
    print(f"lora_r:         {args.lora_r}")
    print(f"lora_alpha:     {args.lora_alpha}")
    print(f"grad_accum:     {args.grad_accum}")
    print(f"max_steps:      {args.max_steps}")
    print(f"block_size:     {args.block_size}")
    print(f"warmup_ratio:   {args.warmup_ratio}")
    for k, v in bench_agg.items():
        print(f"bench_{k}: {v}")

    return mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--dataset", default="data/datasets/structured_verbalization_compact/text2text.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--run_benchmarks", action="store_true",
                        help="After training, run fast lm-eval harness subset on the adapter")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_and_eval(args)


if __name__ == "__main__":
    main()
