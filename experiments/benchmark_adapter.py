#!/usr/bin/env python3
"""
benchmark_adapter.py — Run lm-eval harness on a base or LoRA-adapted model.

Two modes:
  - "fast" (default): a curated task subset that runs in ~1-2 min on 8xA100-80GB.
    Used inside the autoresearch loop.
  - "full": the same task list as run_benchmarks.sh. Used for final reporting.

Output: dict with per-task accuracy (%) plus the two aggregates consumed by
autoresearch_stub.compute_objective: sci_reasoning (MMLU physics + GPQA) and
general_qa (BBH logical deduction).

CLI:
    python experiments/benchmark_adapter.py \
        --base-model openai/gpt-oss-20b \
        --adapter output_models/gpt-oss-20b_compact \
        --mode fast \
        --output-dir /tmp/bench_001
"""

import argparse
import glob
import json
import os
import subprocess
import sys

ENV_ROOT = "/lcrc/project/cosmo_ai/nramachandra/envs/bimodal"
LM_EVAL_PKG = "/lcrc/project/solitons/nramachandra/lm_eval_pkg"
PYTHON_BIN = f"{ENV_ROOT}/bin/python"

TASKS_FAST = [
    "mmlu_college_physics",
    "leaderboard_gpqa_diamond",
    "leaderboard_bbh_logical_deduction_three_objects",
]

TASKS_FULL = [
    "mmlu_college_physics", "mmlu_high_school_physics", "mmlu_astronomy",
    "leaderboard_gpqa", "leaderboard_bbh",
]

SCI_REASONING_TASKS = {"mmlu_college_physics", "mmlu_high_school_physics",
                       "leaderboard_gpqa_diamond", "leaderboard_gpqa_main",
                       "leaderboard_gpqa_extended"}
GENERAL_QA_PREFIXES = ("leaderboard_bbh",)


def _lm_eval_env():
    env = os.environ.copy()
    env["TMPDIR"] = "/tmp"
    env["HF_HOME"] = "/lcrc/project/cosmo_ai/nramachandra/hf_cache"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = f"{LM_EVAL_PKG}:{env.get('PYTHONPATH', '')}"

    nvidia_root = f"{ENV_ROOT}/lib/python3.11/site-packages/nvidia"
    ld_paths = [p for p in glob.glob(f"{nvidia_root}/*/lib")]
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths + [env.get("LD_LIBRARY_PATH", "")])
    return env


def run_lm_eval(base_model, adapter, tasks, output_dir,
                max_memory_per_gpu="50GiB", batch_size=1, num_fewshot=0):
    os.makedirs(output_dir, exist_ok=True)
    model_args = (
        f"pretrained={base_model},trust_remote_code=True,dtype=bfloat16,"
        f"parallelize=True,max_memory_per_gpu={max_memory_per_gpu},"
        f"attn_implementation=eager"
    )
    if adapter:
        model_args = f"pretrained={base_model},peft={adapter}," + model_args.split(",", 1)[1]

    cmd = [
        PYTHON_BIN, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", output_dir,
    ]

    log_path = os.path.join(output_dir, "lm_eval.log")
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, env=_lm_eval_env(), stdout=logf,
                                stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"lm_eval failed (exit {result.returncode}); see {log_path}")

    return parse_results(output_dir)


def parse_results(output_dir):
    """Walk output_dir for results_*.json; return {task: acc_percent}."""
    scores = {}
    for f in glob.glob(os.path.join(output_dir, "**", "results_*.json"), recursive=True):
        with open(f) as fh:
            data = json.load(fh)
        for task, entry in data.get("results", {}).items():
            acc = entry.get("acc,none",
                  entry.get("acc_norm,none",
                  entry.get("exact_match,none")))
            if acc is not None:
                scores[task] = round(float(acc) * 100, 2)
    return scores


def aggregate(scores):
    """Collapse per-task scores into the two groups used by compute_objective."""
    sci = [v for t, v in scores.items() if t in SCI_REASONING_TASKS]
    qa = [v for t, v in scores.items() if t.startswith(GENERAL_QA_PREFIXES)]
    agg = {}
    if sci:
        agg["sci_reasoning"] = round(sum(sci) / len(sci), 2)
    if qa:
        agg["general_qa"] = round(sum(qa) / len(qa), 2)
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="openai/gpt-oss-20b")
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter (optional)")
    p.add_argument("--mode", choices=["fast", "full"], default="fast")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tasks", default=None,
                   help="Comma-separated task override (ignores --mode)")
    args = p.parse_args()

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = TASKS_FAST if args.mode == "fast" else TASKS_FULL

    scores = run_lm_eval(args.base_model, args.adapter, tasks, args.output_dir)
    agg = aggregate(scores)

    print("--- benchmark results ---")
    for t, v in sorted(scores.items()):
        print(f"bench_{t}: {v}")
    for t, v in agg.items():
        print(f"bench_agg_{t}: {v}")


if __name__ == "__main__":
    main()
