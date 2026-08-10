from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Params:
    """Parameters for a 2-D force-actuated Duffing oscillator.

    The state is ``[x, y, vx, vy]``.  The restoring force contains a linear term
    and an isotropic cubic term, so the plant is nonlinear while remaining small
    enough to inspect end to end.
    """

    mass: float = 1.0
    damping: float = 0.30
    stiffness: float = 0.55
    cubic_stiffness: float = 0.10
    dt: float = 0.04
    max_force: float = 2.5


def bounded_force(raw_control: Array, max_force: float) -> Array:
    """Map unconstrained controls smoothly into the actuator force limit."""
    return max_force * jnp.tanh(raw_control / max_force)


def acceleration(state: Array, raw_control: Array, params: Params) -> Array:
    """Return acceleration for the controlled Duffing oscillator."""
    pos = state[:2]
    vel = state[2:]
    force = bounded_force(raw_control, params.max_force)
    radius_sq = jnp.dot(pos, pos)
    restoring = params.stiffness * pos + params.cubic_stiffness * radius_sq * pos
    return (force - params.damping * vel - restoring) / params.mass


def step(state: Array, raw_control: Array, params: Params) -> tuple[Array, Array]:
    """Advance one semi-implicit Euler step.

    Velocity is updated before position.  Semi-implicit Euler is inexpensive,
    differentiable, and more stable for oscillatory systems than forward Euler.
    """
    vel_next = state[2:] + params.dt * acceleration(state, raw_control, params)
    pos_next = state[:2] + params.dt * vel_next
    next_state = jnp.concatenate([pos_next, vel_next])
    return next_state, next_state


def simulate(initial_state: Array, raw_controls: Array, params: Params) -> Array:
    """Simulate a trajectory for an array of unconstrained controls ``(T, 2)``."""
    raw_controls = jnp.asarray(raw_controls)
    if raw_controls.ndim != 2 or raw_controls.shape[1] != 2:
        raise ValueError("raw_controls must have shape (T, 2)")

    def _scan(state: Array, control: Array) -> tuple[Array, Array]:
        return step(state, control, params)

    _, states = jax.lax.scan(_scan, jnp.asarray(initial_state), raw_controls)
    return states


def mechanical_energy(state: Array, params: Params) -> Array:
    """Mechanical energy of the unforced Duffing plant.

    E = 1/2 m |v|^2 + 1/2 k |x|^2 + 1/4 alpha |x|^4.
    Damping dissipates this energy when no external force is applied.
    """
    pos = state[:2]
    vel = state[2:]
    radius_sq = jnp.dot(pos, pos)
    kinetic = 0.5 * params.mass * jnp.dot(vel, vel)
    potential = 0.5 * params.stiffness * radius_sq + 0.25 * params.cubic_stiffness * radius_sq**2
    return kinetic + potential
