"""Differentiable acoustic hologram optimisation for a phased array.

This module is a compact computational-holography benchmark. It optimises the
phases of a single-sided ultrasonic array through a simplified differentiable
acoustic model and the time-averaged Gor'kov radiation-force potential.

The model is deliberately transparent rather than hardware-faithful: isotropic
point sources in a homogeneous lossless medium, a small positive-contrast
particle, continuous phase control, no gravity, no streaming, no reflections,
and no experimental calibration. The useful claim is methodological: gradients
flow from a physical trapping objective back to all transducer phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class ArrayConfig:
    """Geometry and material parameters for the simplified array model.

    The 40 kHz / 10 mm-pitch geometry is representative of common airborne
    phased-array levitation experiments. ``source_amplitude`` is an uncalibrated
    model amplitude scale, so absolute pressure and stiffness values are not
    presented as hardware predictions.
    """

    n_side: int = 8
    pitch: float = 0.010
    frequency: float = 40_000.0
    sound_speed: float = 343.0
    density: float = 1.20
    source_amplitude: float = 1.0
    # 0.6 mm diameter keeps ka well below unity for the Gor'kov/Rayleigh model.
    particle_radius: float = 3.0e-4
    particle_density: float = 25.0
    particle_sound_speed: float = 2600.0

    @property
    def wavenumber(self) -> Array:
        return 2.0 * jnp.pi * self.frequency / self.sound_speed

    @property
    def wavelength(self) -> float:
        return self.sound_speed / self.frequency

    @property
    def rayleigh_ka(self) -> Array:
        return self.wavenumber * self.particle_radius


def transducer_positions(cfg: ArrayConfig) -> Array:
    """Return ``(n_side**2, 3)`` source centres on the ``z = 0`` plane."""
    offset = 0.5 * (cfg.n_side - 1) * cfg.pitch
    axis = jnp.arange(cfg.n_side) * cfg.pitch - offset
    xx, yy = jnp.meshgrid(axis, axis, indexing="ij")
    zz = jnp.zeros_like(xx)
    return jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)


def _gorkov_constants(cfg: ArrayConfig) -> tuple[Array, Array]:
    """Return Gor'kov coefficients for the chosen positive-contrast particle."""
    omega = 2.0 * jnp.pi * cfg.frequency
    volume = (4.0 / 3.0) * jnp.pi * cfg.particle_radius**3
    rho0, c0 = cfg.density, cfg.sound_speed
    rhop, cp = cfg.particle_density, cfg.particle_sound_speed
    f1 = 1.0 - (rho0 * c0**2) / (rhop * cp**2)
    f2 = 2.0 * (rhop - rho0) / (2.0 * rhop + rho0)
    k1 = volume * f1 / (4.0 * rho0 * c0**2)
    k2 = volume * 3.0 * f2 / (8.0 * rho0 * omega**2)
    return k1, k2


def pressure_parts(point: Array, phases: Array, cfg: ArrayConfig) -> Array:
    """Return ``[Re p, Im p]`` at ``point`` in the simplified source model."""
    sources = transducer_positions(cfg)
    delta = point[None, :] - sources
    distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-12)
    phase = cfg.wavenumber * distance + phases
    amplitude = cfg.source_amplitude / distance
    return jnp.stack(
        [jnp.sum(amplitude * jnp.cos(phase)), jnp.sum(amplitude * jnp.sin(phase))]
    )


def pressure_magnitude_sq(point: Array, phases: Array, cfg: ArrayConfig) -> Array:
    parts = pressure_parts(point, phases, cfg)
    return jnp.sum(parts * parts)


def gorkov_potential(point: Array, phases: Array, cfg: ArrayConfig) -> Array:
    """Time-averaged Gor'kov potential in the simplified Rayleigh-particle model."""
    k1, k2 = _gorkov_constants(cfg)
    parts = pressure_parts(point, phases, cfg)
    jac = jax.jacfwd(pressure_parts)(point, phases, cfg)
    return k1 * jnp.sum(parts * parts) - k2 * jnp.sum(jac * jac)


def radiation_force(point: Array, phases: Array, cfg: ArrayConfig) -> Array:
    """Acoustic radiation force ``-grad(U)``; gravity is intentionally omitted."""
    return -jax.grad(gorkov_potential)(point, phases, cfg)


def gorkov_hessian(point: Array, phases: Array, cfg: ArrayConfig) -> Array:
    """Full spatial Hessian of the Gor'kov potential."""
    return jax.hessian(gorkov_potential)(point, phases, cfg)


def trap_stiffness(point: Array, phases: Array, cfg: ArrayConfig) -> Array:
    """Principal local stiffnesses: eigenvalues of the Gor'kov Hessian."""
    return jnp.linalg.eigvalsh(gorkov_hessian(point, phases, cfg))


def focus_phases(target: Array, cfg: ArrayConfig) -> Array:
    """Phase-conjugation baseline that aligns all source phases at ``target``."""
    sources = transducer_positions(cfg)
    distance = jnp.linalg.norm(target[None, :] - sources, axis=-1)
    return -cfg.wavenumber * distance


def _softmin(values: Array, sharpness: float = 40.0) -> Array:
    """Smooth minimum used so the weakest principal stiffness drives the gradient."""
    return -jax.scipy.special.logsumexp(-sharpness * values) / sharpness


def objective_scales(target: Array, cfg: ArrayConfig) -> tuple[Array, Array, Array]:
    """Deterministic normalisation scales from the phase-conjugation baseline."""
    base = focus_phases(target, cfg)
    pressure_ref = pressure_magnitude_sq(target, base, cfg)
    stiffness_ref = jnp.max(jnp.abs(trap_stiffness(target, base, cfg)))
    force_ref = jnp.linalg.norm(radiation_force(target, base, cfg))
    return pressure_ref, stiffness_ref, force_ref


def trap_objective(
    phases: Array,
    target: Array,
    cfg: ArrayConfig,
    stiffness_weight: float = 1.0,
    force_weight: float = 0.3,
) -> Array:
    """Optimise a low-pressure, locally restoring, near-equilibrium field.

    The objective combines three dimensionless terms: pressure-node formation,
    a penalty on non-zero acoustic force at the requested target, and a reward
    for the weakest *principal* stiffness of the full Gor'kov Hessian.
    """
    pressure_ref, stiffness_ref, force_ref = objective_scales(target, cfg)
    null_term = pressure_magnitude_sq(target, phases, cfg) / pressure_ref
    stiffness = trap_stiffness(target, phases, cfg) / stiffness_ref
    weakest = _softmin(stiffness)
    force_ratio = jnp.linalg.norm(radiation_force(target, phases, cfg)) / (
        force_ref + 1e-30
    )
    return null_term + force_weight * force_ratio**2 - stiffness_weight * weakest


def optimise_phases(
    target: Array,
    cfg: ArrayConfig,
    steps: int = 1500,
    learning_rate: float = 0.02,
    seed: int = 0,
) -> tuple[Array, dict[str, Array]]:
    """Optimise all transducer phases with deterministic seeded Adam."""
    key = jax.random.PRNGKey(seed)
    phases = focus_phases(target, cfg) + 0.1 * jax.random.normal(
        key, (cfg.n_side**2,)
    )
    grad_fn = jax.jit(jax.value_and_grad(trap_objective), static_argnums=(2,))

    m = jnp.zeros_like(phases)
    v = jnp.zeros_like(phases)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []
    for step in range(1, steps + 1):
        value, grad = grad_fn(phases, target, cfg)
        m = b1 * m + (1.0 - b1) * grad
        v = b2 * v + (1.0 - b2) * grad * grad
        m_hat = m / (1.0 - b1**step)
        v_hat = v / (1.0 - b2**step)
        phases = phases - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        history.append(value)

    return phases, {"objective_history": jnp.stack(history)}


def evaluate(phases: Array, target: Array, cfg: ArrayConfig) -> dict[str, float | bool]:
    """Return diagnostics that bound what can be claimed about the model field."""
    hessian = gorkov_hessian(target, phases, cfg)
    principal = jnp.linalg.eigvalsh(hessian)
    force = radiation_force(target, phases, cfg)
    _, _, focus_force_ref = objective_scales(target, cfg)
    # Newton displacement is a local linear estimate of where grad(U)=0 would lie.
    displacement = jnp.linalg.solve(hessian, force)
    offset = jnp.linalg.norm(displacement)
    return {
        "pressure_magnitude_model_units": float(
            jnp.sqrt(pressure_magnitude_sq(target, phases, cfg))
        ),
        "gorkov_potential_model_units": float(gorkov_potential(target, phases, cfg)),
        "principal_stiffness_1": float(principal[0]),
        "principal_stiffness_2": float(principal[1]),
        "principal_stiffness_3": float(principal[2]),
        "minimum_principal_stiffness": float(jnp.min(principal)),
        "positive_definite_hessian": bool(jnp.all(principal > 0.0)),
        "acoustic_force_norm_model_units": float(jnp.linalg.norm(force)),
        "force_ratio_to_focus_baseline": float(
            jnp.linalg.norm(force) / (focus_force_ref + 1e-30)
        ),
        "linearised_equilibrium_offset_m": float(offset),
    }
