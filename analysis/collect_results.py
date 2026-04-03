#!/usr/bin/env python3
"""
collect_results.py — Aggregate experiment metrics into ASCII and LaTeX tables.

Usage:
  python collect_results.py --base-dir /path/to/bimodal_reasoning
"""

import os
import json
import argparse
import glob
import numpy as np
from tabulate import tabulate


def find_experiment_dirs(base_dir):
    dirs = []
    for d in glob.glob(os.path.join(base_dir, "experiments", "*")):
        if os.path.exists(os.path.join(d, "results", "metrics.json")):
            dirs.append(d)
    return dirs


def group_experiments(experiment_dirs):
    grouped = {}
    for d in experiment_dirs:
        name = os.path.basename(d).rsplit('_', 3)[0]  # strip timestamp
        grouped.setdefault(name, []).append(d)
    return grouped


def load_metrics(exp_dir):
    with open(os.path.join(exp_dir, "results", "metrics.json")) as f:
        return json.load(f)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.2e}" if value < 0.001 else f"{value:.5f}"
    return str(value)


def generate_table(exp_dirs, group_name):
    headers = [
        "LR", "LoRA rank", "Epochs", "Train samples",
        "Tokenization", "Redshift MAE", "Sci. Reasoning", "General QA"
    ]
    rows = []
    for d in exp_dirs:
        m = load_metrics(d)
        p = m.get('parameters', {})
        met = m.get('metrics', {})
        bm = m.get('benchmarks', {}).get('finetuned_model', {})
        rows.append([
            format_value(p.get('learning_rate', 'N/A')),
            p.get('lora_r', 'N/A'),
            p.get('num_train_epochs', 'N/A'),
            p.get('training_samples', 'N/A'),
            p.get('tokenization_strategy', 'N/A'),
            met.get('redshift_mae', bm.get('redshift_mae', 'N/A')),
            bm.get('general_scientific', met.get('scientific_reasoning', 'N/A')),
            bm.get('general_qa', met.get('general_qa', 'N/A')),
        ])
    return tabulate(rows, headers=headers, tablefmt="grid")


def generate_latex(grouped):
    lines = [
        "\\begin{table}",
        "\\caption{Performance Across Fine-tuning Configurations}",
        "\\begin{tabular}{cccccccc}",
        "\\hline",
        "LR & Rank & Epochs & Samples & Tokenization & Redshift MAE & Sci. & QA \\\\",
        "\\hline",
    ]

    for group_name, dirs in grouped.items():
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{group_name.replace('_', ' ').title()}}}}} \\\\")
        for d in sorted(dirs):
            m = load_metrics(d)
            p = m.get('parameters', {})
            met = m.get('metrics', {})
            bm = m.get('benchmarks', {}).get('finetuned_model', {})
            lines.append(
                f"{format_value(p.get('learning_rate', 'N/A'))} & "
                f"{p.get('lora_r', 'N/A')} & "
                f"{p.get('num_train_epochs', 'N/A')} & "
                f"{p.get('training_samples', 'N/A')} & "
                f"{p.get('tokenization_strategy', 'N/A')} & "
                f"{met.get('redshift_mae', 'N/A')} & "
                f"{bm.get('general_scientific', 'N/A')} & "
                f"{bm.get('general_qa', 'N/A')} \\\\"
            )
        lines.append("\\hline")

    lines += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', required=True)
    parser.add_argument('--output', default='results_table.txt')
    parser.add_argument('--latex', default='results_table.tex')
    args = parser.parse_args()

    dirs = find_experiment_dirs(args.base_dir)
    if not dirs:
        print(f"No results found in {args.base_dir}/experiments/")
        return

    print(f"Found {len(dirs)} experiments")
    grouped = group_experiments(dirs)

    with open(args.output, 'w') as f:
        for group_name, exp_dirs in grouped.items():
            print(f"Generating table for {group_name}...")
            table = generate_table(exp_dirs, group_name)
            f.write(f"# {group_name}\n\n{table}\n\n")
    print(f"ASCII table: {args.output}")

    latex = generate_latex(grouped)
    with open(args.latex, 'w') as f:
        f.write(latex)
    print(f"LaTeX table: {args.latex}")


if __name__ == "__main__":
    main()
