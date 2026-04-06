#!/usr/bin/env python3
"""
convert_to_bf16.py — Pre-dequantize an MXFP4 model to bf16 on CPU.

This avoids the per-rank dequantization OOM when loading with DeepSpeed ZeRO-3
(8 ranks × 240GB transient buffers > 1TB RAM).

Usage:
    python training/convert_to_bf16.py \
        --model_name_or_path openai/gpt-oss-120b \
        --output_dir /lcrc/project/cosmo_ai/nramachandra/hf_cache/gpt-oss-120b-bf16

Single-process, ~240GB peak RAM for 120B model. Takes ~20-30 min.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    print(f"Loading model {args.model_name_or_path} in bf16 (single process, CPU) ...")
    print("  This will dequantize MXFP4 → bf16 and requires ~240GB RAM for 120B models.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    print(f"Saving bf16 model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output_dir)

    print("Done. You can now load this with DeepSpeed without MXFP4 dequant overhead.")


if __name__ == "__main__":
    main()
