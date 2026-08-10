from __future__ import annotations

from itertools import product

import jax
import jax.numpy as jnp

from .dynamics import Array, Params, bounded_force, simulate, step


def tracking_objective(
    raw_controls: Array,
    initial_state: Array,
    targets: Array,
    params: Params,
    control_weight: float = 2e-3,
    slew_weight: float = 2e-2,
) -> Array:
    """Trajectory loss = tracking error + effort + force-slew regularisation."""
    states = simulate(initial_state, raw_controls, params)
    position_error = states[:, :2] - targets
    tracking = jnp.mean(jnp.sum(position_error**2, axis=1))
    forces = bounded_force(raw_controls, params.max_force)
    effort = jnp.mean(jnp.sum(forces**2, axis=1))
    if raw_controls.shape[0] > 1:
        slew = jnp.mean(jnp.sum(jnp.diff(forces, axis=0) ** 2, axis=1))
    else:
        slew = jnp.asarray(0.0)
    return tracking + control_weight * effort + slew_weight * slew


def optimise_controls(
    initial_state: Array,
    targets: Array,
    params: Params,
    iterations: int = 350,
    learning_rate: float = 0.04,
    initial_controls: Array | None = None,
) -> tuple[Array, list[float]]:
    """Optimise an open-loop control sequence with autodiff and Adam.

    This is trajectory optimisation, not a claim of closed-loop robust control.
    """
    controls = (
        jnp.zeros((targets.shape[0], 2), dtype=jnp.asarray(targets).dtype)
        if initial_controls is None
        else jnp.asarray(initial_controls)
    )
    if controls.shape != targets.shape:
        raise ValueError("initial_controls and targets must both have shape (T, 2)")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    value_and_grad = jax.jit(jax.value_and_grad(tracking_objective))
    first_moment = jnp.zeros_like(controls)
    second_moment = jnp.zeros_like(controls)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history: list[float] = []

    for step_index in range(1, iterations + 1):
        value, gradient = value_and_grad(controls, initial_state, targets, params)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        m_hat = first_moment / (1.0 - beta1**step_index)
        v_hat = second_moment / (1.0 - beta2**step_index)
        controls = controls - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        history.append(float(value))

    return controls, history


def reference_kinematics(targets: Array, dt: float) -> tuple[Array, Array]:
    """Periodic centred differences for reference velocity and acceleration."""
    targets = jnp.asarray(targets)
    velocity = (jnp.roll(targets, -1, axis=0) - jnp.roll(targets, 1, axis=0)) / (2.0 * dt)
    acceleration = (
        jnp.roll(targets, -1, axis=0) - 2.0 * targets + jnp.roll(targets, 1, axis=0)
    ) / dt**2
    return velocity, acceleration


def computed_force_pd_rollout(
    initial_state: Array,
    targets: Array,
    nominal_params: Params,
    kp: float,
    kd: float,
    plant_params: Params | None = None,
) -> tuple[Array, Array]:
    """Closed-loop physics feed-forward + PD reference controller.

    Feed-forward force is computed from the nominal Duffing model and the known
    reference kinematics.  Feedback uses the observed simulated state.  When
    ``plant_params`` differs from ``nominal_params``, the controller still uses the
    nominal model, so the rollout measures feedback response to model mismatch.
    """
    plant = nominal_params if plant_params is None else plant_params
    ref_velocity, ref_acceleration = reference_kinematics(targets, nominal_params.dt)

    def _scan(
        state: Array, inputs: tuple[Array, Array, Array]
    ) -> tuple[Array, tuple[Array, Array]]:
        target, target_velocity, target_acceleration = inputs
        radius_sq = jnp.dot(target, target)
        feed_forward = (
            nominal_params.mass * target_acceleration
            + nominal_params.damping * target_velocity
            + nominal_params.stiffness * target
            + nominal_params.cubic_stiffness * radius_sq * target
        )
        feedback = kp * (target - state[:2]) + kd * (target_velocity - state[2:])
        raw_control = feed_forward + feedback
        next_state, _ = step(state, raw_control, plant)
        return next_state, (next_state, raw_control)

    _, (states, controls) = jax.lax.scan(
        _scan,
        jnp.asarray(initial_state),
        (jnp.asarray(targets), ref_velocity, ref_acceleration),
    )
    return states, controls


def _trajectory_rmse(states: Array, targets: Array) -> Array:
    return jnp.sqrt(jnp.mean(jnp.sum((states[:, :2] - targets) ** 2, axis=1)))


def tune_reference_controller(
    initial_state: Array,
    targets: Array,
    params: Params,
    kp_grid: tuple[float, ...] = (2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 30.0),
    kd_grid: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0),
) -> tuple[float, float, Array, Array, float, float]:
    """Tune the reference controller on the same composite nominal objective."""
    rollout = jax.jit(computed_force_pd_rollout)
    best: tuple[float, float, Array, Array, float, float] | None = None
    for kp, kd in product(kp_grid, kd_grid):
        states, controls = rollout(initial_state, targets, params, kp, kd)
        objective = float(tracking_objective(controls, initial_state, targets, params))
        score = float(_trajectory_rmse(states, targets))
        if best is None or objective < best[-1]:
            best = (kp, kd, states, controls, score, objective)
    assert best is not None
    return best
