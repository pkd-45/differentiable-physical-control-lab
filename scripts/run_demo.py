from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from physctrl.experiment import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the physical-control benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("products"),
        help="Directory for generated JSON metrics and figures (default: products).",
    )
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    result = run_benchmark()
    summary = {
        "model": "2-D controlled Duffing oscillator",
        "optimised_open_loop_rmse": result["optimised_rmse"],
        "feedback_reference_rmse": result["reference_rmse"],
        "feedback_reference_objective": result["reference_objective"],
        "feedback_reference_kp": result["reference_kp"],
        "feedback_reference_kd": result["reference_kd"],
        "robustness_cases": len(result["robustness_cases"]),
        "open_loop_robustness_median_rmse": result["open_loop_robustness_median_rmse"],
        "open_loop_robustness_p95_rmse": result["open_loop_robustness_p95_rmse"],
        "open_loop_robustness_max_rmse": result["open_loop_robustness_max_rmse"],
        "feedback_robustness_median_rmse": result["feedback_robustness_median_rmse"],
        "feedback_robustness_p95_rmse": result["feedback_robustness_p95_rmse"],
        "feedback_robustness_max_rmse": result["feedback_robustness_max_rmse"],
        "peak_force": result["peak_force"],
        "peak_force_component": result["peak_force_component"],
        "rms_force_slew": result["rms_force_slew"],
        "final_objective": result["final_objective"],
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "robustness_cases.json").write_text(
        json.dumps(result["robustness_cases"], indent=2) + "\n"
    )

    targets = result["targets"]
    optimised = result["optimised_states"]
    reference = result["reference_states"]
    worst_open = result["worst_open_states"]
    worst_feedback = result["worst_feedback_states"]

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(targets[:, 0], targets[:, 1], "--", label="target")
    ax.plot(optimised[:, 0], optimised[:, 1], label="differentiable open-loop optimisation")
    ax.plot(reference[:, 0], reference[:, 1], label="model-based feedback reference")
    ax.set(xlabel="x", ylabel="y", title="Nominal nonlinear trajectory tracking")
    ax.axis("equal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(targets[:, 0], targets[:, 1], "--", label="target")
    ax.plot(worst_open[:, 0], worst_open[:, 1], label="open loop")
    ax.plot(worst_feedback[:, 0], worst_feedback[:, 1], label="feedback reference")
    ax.set(xlabel="x", ylabel="y", title="Worst sampled open-loop mismatch case")
    ax.axis("equal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "worst_case_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result["history"])
    ax.set(
        xlabel="optimisation iteration",
        ylabel="objective",
        title="Autodiff trajectory-optimisation convergence",
    )
    fig.tight_layout()
    fig.savefig(out / "optimisation_history.png", dpi=180)
    plt.close(fig)

    open_rmses = np.asarray([row["open_loop_rmse"] for row in result["robustness_cases"]])
    feedback_rmses = np.asarray(
        [row["feedback_reference_rmse"] for row in result["robustness_cases"]]
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(open_rmses, feedback_rmses)
    lo = float(min(open_rmses.min(), feedback_rmses.min()))
    hi = float(max(open_rmses.max(), feedback_rmses.max()))
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1)
    ax.set(
        xlabel="open-loop RMSE",
        ylabel="feedback-reference RMSE",
        title="Plant-mismatch sensitivity across 18 cases",
    )
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
