"""Physical and numerical checks for the simplified acoustic hologram model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from physctrl.acoustic import (
    ArrayConfig,
    evaluate,
    focus_phases,
    gorkov_hessian,
    gorkov_potential,
    optimise_phases,
    pressure_magnitude_sq,
    pressure_parts,
    radiation_force,
    transducer_positions,
    trap_objective,
    trap_stiffness,
)

CFG = ArrayConfig()
TARGET = jnp.array([0.0, 0.0, 0.06])


def test_array_geometry_is_centred_and_planar():
    pos = transducer_positions(CFG)
    assert pos.shape == (CFG.n_side**2, 3)
    assert jnp.allclose(jnp.mean(pos[:, :2], axis=0), 0.0, atol=1e-8)
    assert jnp.allclose(pos[:, 2], 0.0)


def test_default_particle_is_in_small_ka_regime():
    assert float(CFG.rayleigh_ka) < 0.3


def test_focus_phases_outperform_random_phases_at_target_pressure():
    focus = focus_phases(TARGET, CFG)
    focused = pressure_magnitude_sq(TARGET, focus, CFG)
    key = jax.random.PRNGKey(0)
    for k in jax.random.split(key, 5):
        random_phases = jax.random.uniform(k, (CFG.n_side**2,), maxval=2 * jnp.pi)
        assert focused > pressure_magnitude_sq(TARGET, random_phases, CFG)


def test_global_phase_offset_preserves_pressure_magnitude():
    focus = focus_phases(TARGET, CFG)
    assert jnp.allclose(
        pressure_magnitude_sq(TARGET, focus, CFG),
        pressure_magnitude_sq(TARGET, focus + 0.7, CFG),
        rtol=1e-5,
    )


def test_pressure_is_linear_in_source_amplitude():
    cfg2 = ArrayConfig(source_amplitude=2.0 * CFG.source_amplitude)
    focus = focus_phases(TARGET, CFG)
    assert jnp.allclose(
        pressure_parts(TARGET, focus, cfg2),
        2.0 * pressure_parts(TARGET, focus, CFG),
        rtol=1e-5,
    )


def test_phase_conjugation_baseline_is_not_locally_restoring_in_all_axes():
    principal = trap_stiffness(TARGET, focus_phases(TARGET, CFG), CFG)
    assert jnp.min(principal) < 0.0


def test_gradients_flow_through_the_complete_acoustic_objective():
    grad = jax.grad(trap_objective)(focus_phases(TARGET, CFG), TARGET, CFG)
    assert grad.shape == (CFG.n_side**2,)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.linalg.norm(grad) > 0.0


def test_hessian_diagonal_matches_finite_difference_second_derivatives():
    phases = focus_phases(TARGET, CFG)
    hessian = gorkov_hessian(TARGET, phases, CFG)
    step = 1e-4
    for axis in range(3):
        shift = jnp.zeros(3).at[axis].set(step)
        second = (
            gorkov_potential(TARGET + shift, phases, CFG)
            - 2.0 * gorkov_potential(TARGET, phases, CFG)
            + gorkov_potential(TARGET - shift, phases, CFG)
        ) / step**2
        assert jnp.allclose(hessian[axis, axis], second, rtol=0.05)


@pytest.mark.slow
def test_optimisation_produces_positive_definite_near_equilibrium_trap():
    phases, history = optimise_phases(TARGET, CFG, steps=1500)
    before_pressure = jnp.sqrt(
        pressure_magnitude_sq(TARGET, focus_phases(TARGET, CFG), CFG)
    )
    after = evaluate(phases, TARGET, CFG)

    assert history["objective_history"][-1] < history["objective_history"][0]
    assert after["positive_definite_hessian"] is True
    assert after["minimum_principal_stiffness"] > 0.0
    assert after["pressure_magnitude_model_units"] < 0.02 * float(before_pressure)
    assert after["force_ratio_to_focus_baseline"] < 0.01
    assert after["linearised_equilibrium_offset_m"] < 0.01 * CFG.wavelength
    assert jnp.all(jnp.isfinite(radiation_force(TARGET, phases, CFG)))


@pytest.mark.slow
def test_optimisation_is_deterministic_for_fixed_seed():
    a, _ = optimise_phases(TARGET, CFG, steps=200, seed=3)
    b, _ = optimise_phases(TARGET, CFG, steps=200, seed=3)
    assert jnp.allclose(a, b)
