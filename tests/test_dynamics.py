import jax.numpy as jnp
import pytest

from physctrl.dynamics import Params, bounded_force, mechanical_energy, simulate


def test_force_components_are_bounded():
    f = bounded_force(jnp.array([100.0, -100.0]), 2.0)
    assert jnp.all(jnp.abs(f) <= 2.0 + 1e-6)


def test_zero_controls_preserve_zero_state():
    states = simulate(jnp.zeros(4), jnp.zeros((20, 2)), Params())
    assert jnp.allclose(states, 0.0)


def test_unforced_damped_energy_decays_over_long_run():
    params = Params(dt=0.01)
    initial = jnp.array([0.5, -0.2, 0.1, 0.0])
    states = simulate(initial, jnp.zeros((600, 2)), params)
    energies = jnp.asarray([mechanical_energy(state, params) for state in states])
    assert jnp.all(jnp.diff(energies) <= 1e-6)
    assert float(energies[-1]) < 0.25 * float(mechanical_energy(initial, params))


def test_invalid_control_shape_fails_closed():
    with pytest.raises(ValueError, match="shape"):
        simulate(jnp.zeros(4), jnp.zeros((10,)), Params())


def test_time_step_refinement_reduces_discretisation_error():
    initial = jnp.array([0.5, -0.2, 0.1, 0.0])
    state_dt_002 = simulate(initial, jnp.zeros((100, 2)), Params(dt=0.02))[-1]
    state_dt_001 = simulate(initial, jnp.zeros((200, 2)), Params(dt=0.01))[-1]
    state_dt_0005 = simulate(initial, jnp.zeros((400, 2)), Params(dt=0.005))[-1]
    coarse_difference = jnp.linalg.norm(state_dt_002 - state_dt_001)
    fine_difference = jnp.linalg.norm(state_dt_001 - state_dt_0005)
    assert float(fine_difference) < 0.6 * float(coarse_difference)
