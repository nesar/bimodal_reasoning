#!/usr/bin/env python3
"""
benchmark_extraction.py — Parse LM eval harness JSON results into metrics.json.

Usage:
  python benchmark_extraction.py /path/to/experiment_dir
"""

import os
import json
import sys
import numpy as np
import traceback
import glob
import datetime


def find_results_files(directory, subdir):
    target = os.path.join(directory, subdir)
    if not os.path.exists(target):
        print(f"Warning: {subdir} directory not found at {target}")
        return []
    files = []
    for root, _, fnames in os.walk(target):
        for fname in fnames:
            if fname.startswith('results') and fname.endswith('.json'):
                files.append(os.path.join(root, fname))
    return files


def extract_metrics_from_file(results_path):
    try:
        with open(results_path) as f:
            data = json.load(f)

        out = {
            "general_qa": "N/A",
            "general_scientific": "N/A",
            "specialized_astronomy": "N/A",
            "source": os.path.basename(results_path),
        }

        # General QA — BBH tasks
        bbh = [k for k in data['results'] if k.startswith('leaderboard_bbh')]
        bbh_scores = []
        for bench in bbh:
            keys = [k for k in data['results'][bench] if k.startswith('acc_norm') or k.startswith('acc')]
            if keys:
                bbh_scores.append(data['results'][bench][keys[0]] * 100)
        if bbh_scores:
            out["general_qa"] = f"{np.mean(bbh_scores):.1f}%"

        # General Scientific — physics, GPQA, math_hard
        sci_benches = ['mmlu_college_physics', 'mmlu_high_school_physics']
        sci_benches += [k for k in data['results'] if k.startswith('leaderboard_gpqa')]
        sci_benches += [k for k in data['results'] if k.startswith('leaderboard_math_hard')]
        sci_scores = []
        for bench in sci_benches:
            if bench in data['results']:
                keys = [k for k in data['results'][bench] if k.startswith('acc')]
                if keys:
                    sci_scores.append(data['results'][bench][keys[0]] * 100)
        if sci_scores:
            out["general_scientific"] = f"{np.mean(sci_scores):.1f}%"

        # Specialized Astronomy
        astro_benches = ['astro_mlab_araa_mcq_gemini15']
        astro_scores = []
        for bench in astro_benches:
            if bench in data['results']:
                keys = [k for k in data['results'][bench] if k.startswith('acc')]
                if keys:
                    astro_scores.append(data['results'][bench][keys[0]] * 100)
        if astro_scores:
            out["specialized_astronomy"] = f"{np.mean(astro_scores):.1f}%"

        return out

    except Exception as e:
        print(f"Error extracting from {results_path}: {e}")
        traceback.print_exc()
        return {"general_qa": "ERROR", "general_scientific": "ERROR",
                "specialized_astronomy": "ERROR", "source": str(e)}


def extract_benchmark_results(plots_dir):
    results = {}
    for label, subdir in [("finetuned_model", "eval_local"), ("base_model", "eval_hf")]:
        files = find_results_files(plots_dir, subdir)
        if files:
            files.sort(reverse=True)
            print(f"Using {subdir} results: {files[0]}")
            results[label] = extract_metrics_from_file(files[0])
        else:
            results[label] = {
                "general_qa": f"N/A (no {subdir})",
                "general_scientific": f"N/A (no {subdir})",
                "specialized_astronomy": f"N/A (no {subdir})",
                "source": "none",
            }
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_extraction.py /path/to/experiment_dir")
        sys.exit(1)

    experiment_dir = sys.argv[1]
    plots_dir = os.path.join(experiment_dir, "plots")
    results_dir = os.path.join(experiment_dir, "results")
    metrics_file = os.path.join(results_dir, "metrics.json")

    os.makedirs(results_dir, exist_ok=True)

    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        metrics = {"metrics": {}, "metadata": {}}

    metrics.setdefault("metadata", {})
    metrics["metadata"]["updated_at"] = datetime.datetime.now().isoformat()
    metrics["metadata"]["benchmark_extraction_version"] = "2.1"

    print(f"Extracting benchmarks from {plots_dir} ...")
    benchmark_results = extract_benchmark_results(plots_dir)
    metrics["benchmarks"] = benchmark_results

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {metrics_file}")
    for model, res in benchmark_results.items():
        print(f"\n{model}:")
        for k, v in res.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
