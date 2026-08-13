"""Validate the generated simplified acoustic-hologram products."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/validate_acoustic_outputs.py OUTPUT_DIR")
    out = Path(sys.argv[1])
    required = ["acoustic_metrics.json", "acoustic_hologram.png"]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"Missing generated acoustic products: {missing}")

    metrics = json.loads((out / "acoustic_metrics.json").read_text())
    opt = metrics["optimised_hologram"]
    checks = {
        "64 transducers": metrics["transducers"] == 64,
        "small-ka particle": metrics["rayleigh_ka"] < 0.3,
        "pressure node relative to focus": metrics["pressure_node_ratio"] > 50.0,
        "positive-definite Gor'kov Hessian": opt["positive_definite_hessian"],
        "positive weakest principal stiffness": opt["minimum_principal_stiffness"] > 0.0,
        "near-equilibrium acoustic force": opt["force_ratio_to_focus_baseline"] < 0.01,
        "small linearised equilibrium offset": (
            opt["linearised_equilibrium_offset_m"] < 0.01 * metrics["wavelength_m"]
        ),
        "objective reduced": metrics["objective_final"] < metrics["objective_initial"],
    }
    finite_values = [
        metrics["pressure_node_ratio"],
        metrics["objective_initial"],
        metrics["objective_final"],
        opt["minimum_principal_stiffness"],
        opt["force_ratio_to_focus_baseline"],
        opt["linearised_equilibrium_offset_m"],
    ]
    checks["reported numerical metrics finite"] = all(math.isfinite(v) for v in finite_values)

    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("Acoustic validation failed: " + ", ".join(failed))
    print("\nAcoustic metrics:")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
