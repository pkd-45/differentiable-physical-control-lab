"""Validate generated benchmark products and headline scientific behaviour."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/validate_outputs.py OUTPUT_DIR")
    out = Path(sys.argv[1])
    required = [
        "metrics.json",
        "robustness_cases.json",
        "trajectory.png",
        "optimisation_history.png",
        "robustness.png",
        "worst_case_comparison.png",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"Missing generated products: {missing}")

    metrics = json.loads((out / "metrics.json").read_text())
    cases = json.loads((out / "robustness_cases.json").read_text())
    checks = {
        "18 mismatch cases": metrics["robustness_cases"] == 18 and len(cases) == 18,
        "force component bounded": metrics["peak_force_component"] <= 2.50001,
        "nominal optimiser RMSE lower": (
            metrics["optimised_open_loop_rmse"] < metrics["feedback_reference_rmse"]
        ),
        "feedback median mismatch lower": (
            metrics["feedback_robustness_median_rmse"]
            < metrics["open_loop_robustness_median_rmse"]
        ),
        "feedback sampled maximum lower": (
            metrics["feedback_robustness_max_rmse"]
            < metrics["open_loop_robustness_max_rmse"]
        ),
        "feedback lower in every sampled mismatch case": all(
            row["feedback_reference_rmse"] < row["open_loop_rmse"] for row in cases
        ),
        "all numeric metrics finite": all(
            math.isfinite(value)
            for value in metrics.values()
            if isinstance(value, (int, float))
        ),
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("Scientific validation failed: " + ", ".join(failed))

    print("\nGenerated metrics:")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
