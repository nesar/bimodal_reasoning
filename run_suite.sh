#!/bin/bash
# run_suite.sh — High-level experiment suite runner for bimodal_reasoning
#
# Usage:
#   bash run_suite.sh --generate-only
#   bash run_suite.sh --run-experiment <experiment_id>
#   bash run_suite.sh --run-all
#   bash run_suite.sh --collect-results

BASE_DIR="$(cd "$(dirname "$0")"; pwd)"
CONFIG_FILE="$BASE_DIR/config.yaml"

show_help() {
    echo "Usage: $0 [options]"
    echo "  --generate-only         Generate experiment directories only"
    echo "  --run-experiment ID     Run a specific experiment by ID"
    echo "  --run-all               Run all generated experiments"
    echo "  --collect-results       Collect and tabulate results"
    echo "  --help                  Show this message"
}

GENERATE_ONLY=false
RUN_EXPERIMENT=""
RUN_ALL=false
COLLECT_RESULTS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --generate-only)   GENERATE_ONLY=true; shift ;;
        --run-experiment)  RUN_EXPERIMENT="$2"; shift 2 ;;
        --run-all)         RUN_ALL=true; shift ;;
        --collect-results) COLLECT_RESULTS=true; shift ;;
        --help)            show_help; exit 0 ;;
        *)                 echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

if $GENERATE_ONLY; then
    echo "Generating experiment directories..."
    python "$BASE_DIR/experiments/generate_experiments.py" --config "$CONFIG_FILE"
    echo "Done. Run with --run-all or --run-experiment <id> to execute."
    exit 0
fi

if $RUN_ALL; then
    MASTER="$BASE_DIR/run_all_experiments.sh"
    if [[ ! -f "$MASTER" ]]; then
        echo "Master script not found. Generating..."
        python "$BASE_DIR/experiments/generate_experiments.py" --config "$CONFIG_FILE"
    fi
    echo "Running all experiments..."
    bash "$MASTER"
fi

if [[ -n "$RUN_EXPERIMENT" ]]; then
    EXP_DIR="$BASE_DIR/experiments/$RUN_EXPERIMENT"
    if [[ -d "$EXP_DIR" ]]; then
        echo "Running experiment: $RUN_EXPERIMENT"
        bash "$EXP_DIR/run_experiment.sh"
    else
        echo "Experiment '$RUN_EXPERIMENT' not found."
        echo "Available experiments:"
        ls -1 "$BASE_DIR/experiments/" 2>/dev/null || echo "(none)"
        exit 1
    fi
fi

if $COLLECT_RESULTS; then
    echo "Collecting results..."
    python "$BASE_DIR/analysis/collect_results.py" \
        --base-dir "$BASE_DIR" \
        --output "$BASE_DIR/results_table.txt" \
        --latex "$BASE_DIR/results_table.tex"
    echo "Results table: $BASE_DIR/results_table.txt"
fi

echo "Done."
