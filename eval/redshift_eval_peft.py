#!/usr/bin/env python3
"""
redshift_eval_peft.py — Evaluate a PEFT LoRA adapter on redshift prediction.

Loads the base model, applies the LoRA adapter, merges weights, runs
generation on the test set, and computes MAE + scatter plot.

Usage:
    python eval/redshift_eval_peft.py \
        --base_model openai/gpt-oss-120b \
        --adapter_path output_models/gpt-oss-120b_structured \
        --dataset data/datasets/structured_verbalization/text2text.json \
        --output_dir plots/eval_120b \
        --num_test 100
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.plots import setup_style, plot_redshift_scatter, COLORS


def load_peft_model(base_model_id: str, adapter_path: str):
    n_gpus = torch.cuda.device_count()
    max_mem = {i: "32GiB" for i in range(n_gpus)}
    print(f"Loading base model {base_model_id} on {n_gpus} GPUs ...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map="auto",
        max_memory=max_mem,
        low_cpu_mem_usage=True,
    )
    print(f"  Base model loaded in {time.time() - t0:.1f}s")

    print(f"Loading adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("Merging adapter weights ...")
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_base_model(base_model_id: str):
    n_gpus = torch.cuda.device_count()
    max_mem = {i: "32GiB" for i in range(n_gpus)}
    print(f"Loading base model {base_model_id} (no adapter) on {n_gpus} GPUs ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map="auto",
        max_memory=max_mem,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_redshift(text: str) -> float:
    """Try multiple patterns to extract a numeric redshift from generated text."""
    patterns = [
        r"[Rr]edshift[:\s]*\[?\s*([\d.]+)",
        r"z\s*=\s*([\d.]+)",
        r"([\d]+\.[\d]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return float("nan")


def run_generation(model, tokenizer, instances, max_new_tokens=32):
    z_true_list = []
    z_pred_list = []
    raw_outputs = []

    for i, inst in enumerate(instances):
        prompt = inst["input"] + "\n"
        true_output = inst["output"]

        z_true = extract_redshift(true_output)
        z_true_list.append(z_true)

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
        input_ids = {k: v.to(model.device) for k, v in input_ids.items()}

        with torch.no_grad():
            outputs = model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0][input_ids["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        z_pred = extract_redshift(generated)
        z_pred_list.append(z_pred)
        raw_outputs.append(generated.strip())

        if (i + 1) % 10 == 0:
            valid = np.isfinite(z_pred_list)
            n_valid = sum(valid)
            print(f"  [{i+1}/{len(instances)}] valid={n_valid}, "
                  f"last z_true={z_true:.4f}, z_pred={z_pred:.4f}, "
                  f"gen='{generated[:60]}...'")

    return np.array(z_true_list), np.array(z_pred_list), raw_outputs


def save_results(z_true, z_pred, raw_outputs, instances, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)

    mask = np.isfinite(z_true) & np.isfinite(z_pred)
    n_valid = mask.sum()
    mae = float(np.mean(np.abs(z_true[mask] - z_pred[mask]))) if n_valid > 0 else float("nan")
    median_ae = float(np.median(np.abs(z_true[mask] - z_pred[mask]))) if n_valid > 0 else float("nan")
    outlier_frac = float((np.abs(z_true[mask] - z_pred[mask]) > 0.1).mean()) if n_valid > 0 else float("nan")

    metrics = {
        "label": label,
        "n_total": len(z_true),
        "n_valid": int(n_valid),
        "mae": mae,
        "median_ae": median_ae,
        "outlier_fraction_gt_0.1": outlier_frac,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: MAE={mae:.4f}, MedianAE={median_ae:.4f}, "
          f"Valid={n_valid}/{len(z_true)}, Outliers={outlier_frac:.2%}")

    # Scatter plot
    if n_valid >= 5:
        fig = plot_redshift_scatter(z_true[mask], z_pred[mask], mae=mae,
                                     label=label,
                                     save_path=os.path.join(output_dir, "redshift_scatter.png"))
        print(f"Scatter plot saved to {output_dir}/redshift_scatter.png")

    # Raw outputs
    raw_path = os.path.join(output_dir, "raw_predictions.jsonl")
    with open(raw_path, "w") as f:
        for i, (zt, zp, raw) in enumerate(zip(z_true, z_pred, raw_outputs)):
            f.write(json.dumps({
                "index": i,
                "z_true": float(zt) if np.isfinite(zt) else None,
                "z_pred": float(zp) if np.isfinite(zp) else None,
                "raw_output": raw,
            }) + "\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", default=None,
                        help="Path to LoRA adapter dir (omit for base model eval)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    label = args.label or (Path(args.adapter_path).name if args.adapter_path else args.base_model)

    with open(args.dataset) as f:
        data = json.load(f)
    instances = data["instances"][:args.num_test]
    print(f"Evaluating on {len(instances)} instances")

    if args.adapter_path:
        model, tokenizer = load_peft_model(args.base_model, args.adapter_path)
    else:
        model, tokenizer = load_base_model(args.base_model)

    z_true, z_pred, raw_outputs = run_generation(model, tokenizer, instances)
    metrics = save_results(z_true, z_pred, raw_outputs, instances, args.output_dir, label)

    print(f"\nDone. Results in {args.output_dir}/")
    return metrics


if __name__ == "__main__":
    main()
