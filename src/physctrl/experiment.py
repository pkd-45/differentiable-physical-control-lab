from __future__ import annotations

from dataclasses import replace
from itertools import product

import jax.numpy as jnp
import numpy as np

from .control import (
    computed_force_pd_rollout,
    optimise_controls,
    tracking_objective,
    tune_reference_controller,
)
from .dynamics import Params, bounded_force, simulate


def figure_eight_targets(steps: int = 120, amplitude: float = 0.8) -> jnp.ndarray:
    """Smooth closed target that begins at the origin."""
    if steps < 2:
        raise ValueError("steps must be >= 2")
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, steps, endpoint=False)
    return jnp.stack(
        [amplitude * jnp.sin(phase), 0.5 * amplitude * jnp.sin(2.0 * phase)],
        axis=1,
    )


def rmse(states: jnp.ndarray, targets: jnp.ndarray) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.sum((states[:, :2] - targets) ** 2, axis=1))))


def robustness_sweep(
    initial_state: jnp.ndarray,
    raw_controls: jnp.ndarray,
    targets: jnp.ndarray,
    nominal: Params,
    reference_kp: float,
    reference_kd: float,
) -> tuple[list[dict[str, float]], np.ndarray, np.ndarray]:
    """Compare open-loop optimisation and feedback reference under model mismatch."""
    cases: list[dict[str, float]] = []
    worst_open_states: np.ndarray | None = None
    worst_feedback_states: np.ndarray | None = None
    worst_open_rmse = -np.inf

    for mass_factor, damping_factor, stiffness_factor in product(
        (0.85, 1.0, 1.15), (0.75, 1.0, 1.25), (0.90, 1.10)
    ):
        plant = replace(
            nominal,
            mass=nominal.mass * mass_factor,
            damping=nominal.damping * damping_factor,
            stiffness=nominal.stiffness * stiffness_factor,
        )
        open_states = simulate(initial_state, raw_controls, plant)
        feedback_states, _ = computed_force_pd_rollout(
            initial_state,
            targets,
            nominal,
            reference_kp,
            reference_kd,
            plant_params=plant,
        )
        open_score = rmse(open_states, targets)
        feedback_score = rmse(feedback_states, targets)
        row = {
            "mass_factor": mass_factor,
            "damping_factor": damping_factor,
            "stiffness_factor": stiffness_factor,
            "open_loop_rmse": open_score,
            "feedback_reference_rmse": feedback_score,
        }
        cases.append(row)
        if open_score > worst_open_rmse:
            worst_open_rmse = open_score
            worst_open_states = np.asarray(open_states)
            worst_feedback_states = np.asarray(feedback_states)

    assert worst_open_states is not None and worst_feedback_states is not None
    return cases, worst_open_states, worst_feedback_states


def run_benchmark(steps: int = 120, iterations: int = 350) -> dict[str, object]:
    params = Params()
    initial = jnp.array([0.0, 0.0, 0.0, 0.0])
    targets = figure_eight_targets(steps)

    kp, kd, reference_states, reference_controls, reference_rmse, reference_objective = (
        tune_reference_controller(initial, targets, params)
    )
    optimised_controls, history = optimise_controls(
        initial,
        targets,
        params,
        iterations=iterations,
        initial_controls=reference_controls,
    )
    optimised_states = simulate(initial, optimised_controls, params)
    cases, worst_open_states, worst_feedback_states = robustness_sweep(
        initial, optimised_controls, targets, params, kp, kd
    )
    open_rmse = np.asarray([case["open_loop_rmse"] for case in cases])
    feedback_rmse = np.asarray([case["feedback_reference_rmse"] for case in cases])
    forces = np.asarray(bounded_force(optimised_controls, params.max_force))
    force_slew = np.diff(forces, axis=0)

    return {
        "params": params,
        "targets": np.asarray(targets),
        "optimised_controls": np.asarray(optimised_controls),
        "optimised_forces": forces,
        "optimised_states": np.asarray(optimised_states),
        "reference_states": np.asarray(reference_states),
        "reference_controls": np.asarray(reference_controls),
        "reference_kp": kp,
        "reference_kd": kd,
        "reference_rmse": reference_rmse,
        "reference_objective": reference_objective,
        "history": history,
        "optimised_rmse": rmse(optimised_states, targets),
        "robustness_cases": cases,
        "open_loop_robustness_median_rmse": float(np.median(open_rmse)),
        "open_loop_robustness_p95_rmse": float(np.percentile(open_rmse, 95)),
        "open_loop_robustness_max_rmse": float(np.max(open_rmse)),
        "feedback_robustness_median_rmse": float(np.median(feedback_rmse)),
        "feedback_robustness_p95_rmse": float(np.percentile(feedback_rmse, 95)),
        "feedback_robustness_max_rmse": float(np.max(feedback_rmse)),
        "worst_open_states": worst_open_states,
        "worst_feedback_states": worst_feedback_states,
        "peak_force": float(np.max(np.linalg.norm(forces, axis=1))),
        "peak_force_component": float(np.max(np.abs(forces))),
        "rms_force_slew": float(np.sqrt(np.mean(force_slew**2))),
        "final_objective": float(
            tracking_objective(optimised_controls, initial, targets, params)
        ),
    }
