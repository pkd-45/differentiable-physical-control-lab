"""Run the simplified differentiable acoustic-hologram experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physctrl.acoustic import (
    ArrayConfig,
    evaluate,
    focus_phases,
    optimise_phases,
    pressure_magnitude_sq,
)


def pressure_slice(phases, cfg, extent=0.02, centre_z=0.06, n=121):
    xs = jnp.linspace(-extent, extent, n)
    zs = jnp.linspace(centre_z - extent, centre_z + extent, n)
    grid = jnp.stack(jnp.meshgrid(xs, zs, indexing="ij"), axis=-1)
    points = jnp.stack(
        [grid[..., 0], jnp.zeros_like(grid[..., 0]), grid[..., 1]], axis=-1
    ).reshape(-1, 3)
    field = jax.vmap(lambda p: jnp.sqrt(pressure_magnitude_sq(p, phases, cfg)))(points)
    return xs, zs, field.reshape(n, n)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("products"))
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ArrayConfig()
    target = jnp.array([0.0, 0.0, 0.06])
    baseline_phases = focus_phases(target, cfg)
    baseline = evaluate(baseline_phases, target, cfg)
    optimised_phases, history = optimise_phases(target, cfg, steps=args.steps)
    optimised = evaluate(optimised_phases, target, cfg)

    pressure_ratio = (
        baseline["pressure_magnitude_model_units"]
        / max(optimised["pressure_magnitude_model_units"], 1e-12)
    )
    metrics = {
        "model": "simplified single-sided 8x8 phased array, 40 kHz, Gor'kov potential",
        "transducers": int(cfg.n_side**2),
        "target_height_m": float(target[2]),
        "wavelength_m": float(cfg.wavelength),
        "particle_radius_m": float(cfg.particle_radius),
        "rayleigh_ka": float(cfg.rayleigh_ka),
        "focus_baseline": baseline,
        "optimised_hologram": optimised,
        "pressure_node_ratio": float(pressure_ratio),
        "objective_initial": float(history["objective_history"][0]),
        "objective_final": float(history["objective_history"][-1]),
        "optimiser": {"algorithm": "Adam", "steps": args.steps, "seed": 0},
    }
    (args.output_dir / "acoustic_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, phases, label in (
        (axes[0], baseline_phases, "Phase-conjugation focus"),
        (axes[1], optimised_phases, "Optimised acoustic hologram"),
    ):
        xs, zs, field = pressure_slice(phases, cfg)
        im = ax.pcolormesh(xs * 1e3, zs * 1e3, field.T, shading="auto", cmap="magma")
        ax.plot(0.0, target[2] * 1e3, "o", mfc="none", mec="w", ms=11, mew=1.6)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax, label="relative |p|")

    axes[2].plot(history["objective_history"], lw=1.4)
    axes[2].set_xlabel("Adam step")
    axes[2].set_ylabel("normalised objective")
    axes[2].set_title("Differentiation through the acoustic model", fontsize=10)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output_dir / "acoustic_hologram.png", dpi=140)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
