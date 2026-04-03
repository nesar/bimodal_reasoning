#!/usr/bin/env python3
"""
generate_experiments.py — Generate experiment directories and run scripts from config.yaml.

Usage:
  python generate_experiments.py --config ../config.yaml
"""

import os
import yaml
import json
import argparse
import shutil
import stat
import datetime
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_vars(config):
    base_config = config['base_config']
    for key, value in base_config.items():
        if isinstance(value, str) and "${base_dir}" in value:
            base_config[key] = value.replace("${base_dir}", base_config['base_dir'])
    return config


def generate_experiment_id(experiment_name, variation_values):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = '_'.join([f"{k}_{v}" for k, v in variation_values.items()])
    return f"{experiment_name}_{param_str}_{timestamp}"


def create_experiment_dir(base_dir, experiment_id):
    experiment_dir = os.path.join(base_dir, "experiments", experiment_id)
    for subdir in ["models", "plots", "logs", "results"]:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    return experiment_dir


def generate_run_script(experiment_dir, config_path, base_config):
    template_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "run_experiment_template.sh"
    )
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template script not found at {template_path}")

    with open(template_path, 'r') as f:
        template = f.read()

    script_content = template.replace("{{CONFIG_PATH}}", config_path)
    script_content = script_content.replace("{{BASE_DIR}}", base_config['base_dir'])

    script_path = os.path.join(experiment_dir, "run_experiment.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)
    return script_path


def generate_all_experiments(config_path):
    config = load_config(config_path)
    config = resolve_vars(config)

    base_config = config['base_config']
    experiments = config['experiments']

    all_experiment_dirs = []
    all_run_scripts = []

    for experiment in experiments:
        name = experiment['name']
        fixed_params = experiment.get('fixed', {})
        variations = experiment['variations']

        print(f"\nProcessing experiment: {name}")
        print(f"  Fixed: {fixed_params}")

        for param_name, param_values in variations.items():
            if not isinstance(param_values, list):
                param_values = [param_values]

            print(f"  Varying {param_name}: {param_values}")

            for value in param_values:
                variation_values = {param_name: value}
                experiment_id = generate_experiment_id(name, variation_values)
                experiment_dir = create_experiment_dir(base_config['base_dir'], experiment_id)

                experiment_config = base_config.copy()
                experiment_config.update(fixed_params)
                experiment_config.update(variation_values)

                cfg_path = os.path.join(experiment_dir, "experiment_config.json")
                with open(cfg_path, 'w') as f:
                    json.dump(experiment_config, f, indent=2)

                script_path = generate_run_script(experiment_dir, cfg_path, base_config)

                all_experiment_dirs.append(experiment_dir)
                all_run_scripts.append(script_path)

                print(f"    Generated: {experiment_id}")

    # Master script
    master_script_path = os.path.join(base_config['base_dir'], "run_all_experiments.sh")
    with open(master_script_path, 'w') as f:
        f.write("#!/bin/bash\n\n# Auto-generated master script\n\n")
        for script_path in all_run_scripts:
            f.write(f"echo 'Running {os.path.basename(os.path.dirname(script_path))}...'\n")
            f.write(f"{script_path} | tee {os.path.dirname(script_path)}/logs/run.log\n\n")

    os.chmod(master_script_path, os.stat(master_script_path).st_mode | stat.S_IEXEC)

    print(f"\nMaster script: {master_script_path}")
    print(f"Total experiments: {len(all_experiment_dirs)}")
    return all_experiment_dirs, all_run_scripts, master_script_path


def main():
    parser = argparse.ArgumentParser(description='Generate experiment scripts from config.yaml.')
    parser.add_argument('--config', default='../config.yaml', help='Path to config.yaml')
    args = parser.parse_args()
    generate_all_experiments(args.config)


if __name__ == "__main__":
    main()
