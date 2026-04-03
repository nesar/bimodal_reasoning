#!/usr/bin/env python3
"""
redshift_eval.py — Evaluate a fine-tuned (or base) model on the redshift prediction task.

Usage:
  # Evaluate fine-tuned model
  python redshift_eval.py /path/to/finetuned_model --output-dir /path/to/plots

  # Evaluate base model (pass 'false' to use HF model)
  python redshift_eval.py false --output-dir /path/to/plots

  # Evaluate with explicit HF model ID
  python redshift_eval.py gpt-oss-120b --output-dir /path/to/plots
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_BASE_MODEL = "gpt-oss-120b"
DEFAULT_DATASET = (
    "/lcrc/project/cosmo_ai/nramachandra/Projects/SpecFoundation"
    "/bimodal_reasoning/data/datasets/spec_text2text/text2text.json"
)


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name} on {device.upper()}")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=32,
        device_map="auto",
    )
    return gen, device


def run_eval(model_path_or_flag, dataset_path, output_dir, num_test=30):
    # Resolve model
    if model_path_or_flag is None or model_path_or_flag.lower() == "false":
        model_name = DEFAULT_BASE_MODEL
        use_local = False
    else:
        model_name = model_path_or_flag
        use_local = True

    generator, device = load_model(model_name)

    # Load dataset
    with open(dataset_path) as f:
        data = json.load(f)
    instances = data["instances"][:num_test]

    # Run generation
    output_texts = []
    for inst in instances:
        result = generator(inst["input"], pad_token_id=generator.tokenizer.eos_token_id)
        output_texts.append(result[0]["generated_text"])

    # Extract true redshift values
    pattern_true = r"Redshift: \[ (-?\d+)"
    true_params = []
    for inst in instances:
        m = re.findall(pattern_true, inst["output"])
        if m:
            true_params.extend(m)
    true_params = np.array(true_params, dtype="float32")

    # Extract predicted redshift values
    pattern_pred = r"Redshift: \[ (-?\d+) \]"
    pred_params = []
    for text in output_texts:
        m = re.search(pattern_pred, text)
        pred_params.append(m.group(1) if m else np.nan)
    pred_params = np.array(pred_params, dtype="float32")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    suffix = "local" if use_local else "hf"

    paired_path = os.path.join(output_dir, f"paired_output_{suffix}.txt")
    with open(paired_path, "w") as f:
        for true_val, text in zip(true_params, output_texts):
            f.write(f"Input Parameter: {true_val}\n")
            f.write(f"Output: {text}\n\n")
    print(f"Paired output saved to {paired_path}")

    # Plot
    z_true = true_params / 10000.0
    z_pred = pred_params / 10000.0
    mae = float(np.nanmean(np.abs(z_true - z_pred)))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(z_true, z_pred, s=25, alpha=0.7)
    ax.plot([0, 0.5], [0, 0.5], "r--", lw=1.5)
    ax.set_xlim(-0.05, 0.55)
    ax.set_ylim(-0.05, 0.55)
    ax.set_xlabel("z_true", fontsize=13)
    ax.set_ylabel("z_pred", fontsize=13)
    ax.set_title(f"True vs Predicted Redshift ({suffix})", fontsize=14)
    ax.text(0.05, 0.95, f"MAE = {mae:.4f}", transform=ax.transAxes,
            fontsize=13, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.grid(True, alpha=0.5)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"true_vs_pred_{suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path} | MAE = {mae:.4f}")

    # General QA test
    questions = [
        "What is a galaxy?",
        f"What type of galaxy is expected at redshift z={z_pred[-1]:.3f}? "
        "List telescope surveys where such a galaxy can be observed.",
    ]
    qa_lines = []
    for q in questions:
        ans = generator(q, pad_token_id=generator.tokenizer.eos_token_id)
        qa_lines.append(f"### Q: {q}\n\n**A:** {ans[0]['generated_text']}\n\n")
        print(f"Q: {q}\nA: {ans[0]['generated_text']}\n")

    qa_path = os.path.join(output_dir, f"qa_output_{suffix}.md")
    with open(qa_path, "w") as f:
        f.writelines(qa_lines)
    print(f"QA saved to {qa_path}")

    return mae


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on redshift prediction task.")
    parser.add_argument(
        "model_path_or_flag", nargs="?", default=None,
        help="Local model path, HF model ID, or 'false' to use default base model"
    )
    parser.add_argument("--output-dir", default=None, help="Directory to save plots/outputs")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to text2text.json")
    parser.add_argument("--num-test", type=int, default=30, help="Number of test instances")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "plots")
    run_eval(args.model_path_or_flag, args.dataset, output_dir, num_test=args.num_test)


if __name__ == "__main__":
    main()
