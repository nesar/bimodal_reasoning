#!/usr/bin/env python3
"""
pareto_loop.py — Multi-objective autoresearch over fine-tuning hyperparameters.

Two objectives, both reported per trial:
  • redshift MAE      (minimize)  — domain task, from run_experiment.py
  • MCQ score         (maximize)  — mean of bench_sci_reasoning + bench_general_qa,
                                     from the lm-eval harness fast subset

Random samples the search space defined in `SEARCH_SPACE`, runs each trial via
`experiments/run_experiment.py --run_benchmarks`, parses the printed metrics,
appends a JSON line to `results.jsonl`, recomputes the Pareto front, and
re-renders the money plot. Resumable: trials with a `trial` index already in
`results.jsonl` are skipped.

Usage:
    python experiments/pareto_loop.py --n-trials 50 \
        --output-dir experiments/autoresearch_runs/pareto
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PYTHON = "/lcrc/project/cosmo_ai/nramachandra/envs/bimodal/bin/python"

SEARCH_SPACE = {
    "lr":           [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "lora_r":       [4, 8, 16, 32, 64],
    "lora_alpha":   [8, 16, 32, 64, 128],
    "lora_dropout": [0.0, 0.05, 0.1],
    "grad_accum":   [2, 4, 8],          # capped at 8 to keep wall time per trial ≲15 min
    "max_steps":    [50, 100, 200],     # capped at 200 for the same reason
    "warmup_ratio": [0.0, 0.05, 0.15],
}


def sample_config(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def parse_trial_log(text):
    """Extract metrics from the run_experiment.py stdout."""
    out = {}
    patterns = {
        "mae":              r"^mae:\s+([\d.eE+-]+)",
        "train_loss":       r"^train_loss:\s+([\d.eE+-]+)",
        "train_seconds":    r"^train_seconds:\s+([\d.eE+-]+)",
        "total_seconds":    r"^total_seconds:\s+([\d.eE+-]+)",
        "sci_reasoning":    r"^bench_sci_reasoning:\s+([\d.eE+-]+)",
        "general_qa":       r"^bench_general_qa:\s+([\d.eE+-]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def run_trial(idx, cfg, trials_dir, dataset, model):
    out_dir = trials_dir / f"trial_{idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    cmd = [
        PYTHON, str(REPO / "experiments" / "run_experiment.py"),
        "--model",        model,
        "--dataset",      dataset,
        "--output_dir",   str(out_dir),
        "--lr",           str(cfg["lr"]),
        "--lora_r",       str(cfg["lora_r"]),
        "--lora_alpha",   str(cfg["lora_alpha"]),
        "--lora_dropout", str(cfg["lora_dropout"]),
        "--grad_accum",   str(cfg["grad_accum"]),
        "--max_steps",    str(cfg["max_steps"]),
        "--warmup_ratio", str(cfg["warmup_ratio"]),
        "--run_benchmarks",
    ]

    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    text = log_path.read_text()
    metrics = parse_trial_log(text)

    return {
        "trial":     idx,
        "wall_time": datetime.now().isoformat(timespec="seconds"),
        "wall_secs": round(elapsed),
        "exit_code": proc.returncode,
        **cfg,
        **metrics,
    }


def mcq_score(rec):
    sci = rec.get("sci_reasoning")
    qa  = rec.get("general_qa")
    if sci is None or qa is None:
        return None
    return 0.5 * (sci + qa)


def is_pareto(records, candidate):
    """True if no other record dominates the candidate. Minimize MAE, maximize MCQ."""
    cm = candidate["mae"]
    cs = candidate["mcq"]
    for r in records:
        if r is candidate:
            continue
        if r["mae"] <= cm and r["mcq"] >= cs and (r["mae"] < cm or r["mcq"] > cs):
            return False
    return True


def label_pareto(records):
    """Mutate each record's `pareto` field based on (mae, mcq) dominance.
    Records missing either metric are excluded from the front."""
    valid = [r for r in records if r.get("mae") is not None and r.get("mcq") is not None]
    for r in records:
        r["pareto"] = False
    for r in valid:
        r["pareto"] = is_pareto(valid, r)


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_jsonl(path):
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def update_plot(jsonl_path, plot_path):
    """Best-effort: don't kill the loop if plotting fails."""
    try:
        from analysis.plot_pareto_money import plot_pareto
        plot_pareto(jsonl_path, plot_path)
    except Exception as e:
        print(f"  [warn] plot update failed: {e}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials",   type=int, default=50)
    p.add_argument("--output-dir", default="experiments/autoresearch_runs/pareto")
    p.add_argument("--dataset",
                   default="data/datasets/structured_verbalization_compact/text2text.json")
    p.add_argument("--model",      default="openai/gpt-oss-20b")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--plot",
                   default="plots/autoresearch_pareto.png")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    plot_path  = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    existing = load_jsonl(jsonl_path)
    done = {r["trial"] for r in existing}
    start_idx = max(done) + 1 if done else 0
    print(f"[pareto-loop] {len(existing)} prior trials; resuming from trial {start_idx}")

    target = start_idx + args.n_trials
    for idx in range(start_idx, target):
        cfg = sample_config(rng)
        print(f"[pareto-loop] trial {idx} cfg={cfg}", flush=True)

        rec = run_trial(idx, cfg, trials_dir, args.dataset, args.model)
        rec["mcq"] = mcq_score(rec)
        append_jsonl(jsonl_path, rec)

        all_records = load_jsonl(jsonl_path)
        label_pareto(all_records)
        # Rewrite jsonl with pareto labels (small file, fine to rewrite each trial)
        with open(jsonl_path, "w") as f:
            for r in all_records:
                f.write(json.dumps(r) + "\n")

        mae = rec.get("mae")
        mcq = rec.get("mcq")
        mae_s = f"{mae:.4f}" if mae is not None else "—"
        mcq_s = f"{mcq:.2f}" if mcq is not None else "—"
        print(f"[pareto-loop] trial {idx} done: MAE={mae_s}  MCQ={mcq_s}  "
              f"({rec['wall_secs']}s, exit={rec['exit_code']})", flush=True)

        update_plot(jsonl_path, plot_path)


if __name__ == "__main__":
    main()
