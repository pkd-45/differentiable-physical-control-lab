import jax
import jax.numpy as jnp

from physctrl.control import (
    computed_force_pd_rollout,
    optimise_controls,
    reference_kinematics,
    tracking_objective,
    tune_reference_controller,
)
from physctrl.dynamics import Params, bounded_force
from physctrl.experiment import figure_eight_targets, robustness_sweep


def test_target_starts_at_initial_position():
    targets = figure_eight_targets(40, 0.5)
    assert jnp.allclose(targets[0], jnp.zeros(2), atol=1e-7)


def test_reference_kinematics_shapes_and_finiteness():
    targets = figure_eight_targets(40, 0.5)
    velocity, acceleration = reference_kinematics(targets, 0.04)
    assert velocity.shape == targets.shape
    assert acceleration.shape == targets.shape
    assert jnp.all(jnp.isfinite(velocity))
    assert jnp.all(jnp.isfinite(acceleration))


def test_objective_gradient_is_finite():
    params = Params()
    initial = jnp.zeros(4)
    targets = figure_eight_targets(30, 0.4)
    controls = jnp.zeros((30, 2))
    gradient = jax.grad(tracking_objective)(controls, initial, targets, params)
    assert gradient.shape == controls.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_single_step_objective_is_finite():
    params = Params()
    value = tracking_objective(jnp.zeros((1, 2)), jnp.zeros(4), jnp.zeros((1, 2)), params)
    assert jnp.isfinite(value)


def test_optimisation_reduces_objective_from_reference_warm_start():
    params = Params()
    initial = jnp.zeros(4)
    targets = figure_eight_targets(50, 0.5)
    _, _, _, reference_controls, _, _ = tune_reference_controller(initial, targets, params)
    start = float(tracking_objective(reference_controls, initial, targets, params))
    controls, _ = optimise_controls(
        initial,
        targets,
        params,
        iterations=100,
        learning_rate=0.06,
        initial_controls=reference_controls,
    )
    end = float(tracking_objective(controls, initial, targets, params))
    assert end < 0.98 * start


def test_feedback_commands_respond_to_plant_induced_state_mismatch():
    nominal = Params()
    plant = Params(mass=1.15, damping=0.24)
    targets = figure_eight_targets(30, 0.4)
    initial = jnp.zeros(4)

    nominal_states, nominal_controls = computed_force_pd_rollout(
        initial, targets, nominal, 10.0, 4.0
    )
    perturbed_states, perturbed_controls = computed_force_pd_rollout(
        initial, targets, nominal, 10.0, 4.0, plant_params=plant
    )

    assert nominal_states.shape == perturbed_states.shape == (30, 4)
    assert nominal_controls.shape == perturbed_controls.shape == (30, 2)
    assert jnp.all(jnp.isfinite(perturbed_states))
    assert jnp.all(jnp.isfinite(perturbed_controls))

    # Both rollouts begin from the same state, so their first commands match.
    # Once the perturbed plant evolves differently, state feedback must alter
    # subsequent commands; otherwise this would behave like a fixed open-loop sequence.
    assert jnp.allclose(nominal_controls[0], perturbed_controls[0], atol=1e-7)
    assert not jnp.allclose(nominal_controls[1:], perturbed_controls[1:], atol=1e-6)


def test_optimised_force_respects_component_limit():
    params = Params(max_force=1.3)
    initial = jnp.zeros(4)
    targets = figure_eight_targets(30, 0.4)
    controls, _ = optimise_controls(initial, targets, params, iterations=30)
    force = bounded_force(controls, params.max_force)
    assert float(jnp.max(jnp.abs(force))) <= params.max_force + 1e-6


def test_robustness_sweep_is_complete_and_finite():
    params = Params()
    initial = jnp.zeros(4)
    targets = figure_eight_targets(30, 0.4)
    controls = jnp.zeros((30, 2))
    cases, worst_open, worst_feedback = robustness_sweep(
        initial, controls, targets, params, 10.0, 4.0
    )
    assert len(cases) == 18
    assert worst_open.shape == (30, 4)
    assert worst_feedback.shape == (30, 4)
    assert all(jnp.isfinite(row["open_loop_rmse"]) for row in cases)
    assert all(jnp.isfinite(row["feedback_reference_rmse"]) for row in cases)
