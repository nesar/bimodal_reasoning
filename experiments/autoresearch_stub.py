#!/usr/bin/env python3
"""
autoresearch_stub.py — Placeholder for autoresearch integration.

Goal: Replace manual grid search with Karpathy's autoresearch framework
to automatically discover optimal fine-tuning configurations.

Reference: https://github.com/karpathy/autoresearch

TODO (see TODO.md #3):
  1. Install autoresearch: pip install autoresearch (or clone from GitHub)
  2. Define the search space below (hyperparams + tokenization strategies)
  3. Define the objective function (redshift MAE + benchmark retention)
  4. Run autoresearch to explore the space and find the best config

For now this file just documents the intended interface.
"""

# ---------------------------------------------------------------------------
# SEARCH SPACE (to be passed to autoresearch)
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "lora_r": [4, 8, 16, 32],
    "num_train_epochs": [1, 2, 3, 5],
    "training_samples": [4000, 8000, 10000],
    "tokenization_strategy": [
        "digit_base10",
        "digit_base16",
        "log_scaled",
        "patch_mean",
        "wavelength_value",
    ],
    "block_size": [256, 512, 1024],
}


# ---------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# ---------------------------------------------------------------------------

def compute_objective(metrics: dict) -> float:
    """
    Compute a scalar objective to maximize (higher = better).

    Design:
      - Primary: minimize redshift MAE → maximize (1 - normalized_mae)
      - Secondary: preserve benchmark scores (penalize catastrophic forgetting)

    Args:
        metrics: dict with keys:
            "redshift_mae": float — MAE of redshift prediction
            "sci_reasoning_base": float  — base model scientific reasoning %
            "sci_reasoning_ft": float    — fine-tuned model scientific reasoning %
            "general_qa_base": float
            "general_qa_ft": float

    Returns:
        scalar objective value
    """
    mae = metrics.get("redshift_mae", 1.0)
    sci_base = metrics.get("sci_reasoning_base", 50.0)
    sci_ft = metrics.get("sci_reasoning_ft", 50.0)
    qa_base = metrics.get("general_qa_base", 50.0)
    qa_ft = metrics.get("general_qa_ft", 50.0)

    # Normalize MAE (lower is better → invert)
    mae_score = max(0.0, 1.0 - mae / 0.1)  # 0.1 = expected bad MAE

    # Benchmark retention penalty: penalize if fine-tuning drops scores
    sci_retention = sci_ft / max(sci_base, 1.0)
    qa_retention = qa_ft / max(qa_base, 1.0)

    # Catastrophic forgetting penalty: if retention < 0.95 (5% drop), penalize heavily
    forgetting_penalty = 0.0
    if sci_retention < 0.95:
        forgetting_penalty += (0.95 - sci_retention) * 5.0
    if qa_retention < 0.95:
        forgetting_penalty += (0.95 - qa_retention) * 5.0

    objective = mae_score - forgetting_penalty
    return float(objective)


# ---------------------------------------------------------------------------
# AUTORESEARCH RUNNER (stub)
# ---------------------------------------------------------------------------

def run_autoresearch():
    """
    TODO: Replace with actual autoresearch calls once the library is installed.

    Intended usage (pseudocode):
        from autoresearch import AutoResearcher
        researcher = AutoResearcher(
            search_space=SEARCH_SPACE,
            objective_fn=compute_objective,
            max_trials=50,
            output_dir="experiments/autoresearch_runs",
        )
        best_config = researcher.run()
        print("Best config found:", best_config)
    """
    raise NotImplementedError(
        "autoresearch not yet integrated. "
        "See TODO.md #3 and https://github.com/karpathy/autoresearch"
    )


if __name__ == "__main__":
    print("autoresearch stub — not yet implemented.")
    print("Search space:")
    for k, v in SEARCH_SPACE.items():
        print(f"  {k}: {v}")
    print("\nObjective function defined: compute_objective(metrics)")
    print("Next step: pip install autoresearch, then implement run_autoresearch()")
