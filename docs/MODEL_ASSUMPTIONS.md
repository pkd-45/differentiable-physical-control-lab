# Model assumptions and limitations

This document bounds the scientific claims of the two experiments in this repository. Both are deliberately compact methods benchmarks. Neither is a calibrated model of a specific experimental apparatus.

## 1. Nonlinear-control experiment

### Plant

The first plant is an isotropic two-dimensional Duffing-type oscillator,

```text
m x_ddot + c x_dot + k x + alpha ||x||^2 x = u,
```

with state `[x, y, vx, vy]`. The radial cubic term couples the axes through displacement magnitude while preserving rotational symmetry. The nominal parameters are normalised benchmark values; no SI or laboratory units are assigned.

The model omits process noise, external disturbances, anisotropic unmodelled modes, actuator dynamics and sensing dynamics.

### Integration and target convention

State propagation uses semi-implicit Euler. The nominal time step is `dt = 0.04`, with 120 control steps in the default experiment. The target is a predefined closed figure-eight sequence known in full before the run.

The target derivatives used by the feedback reference are periodic centred finite differences. This convention is specific to the smooth closed reference and would need revision for non-periodic trajectories.

### Actuator model

Each raw force component is mapped through

```text
u_i = u_max * tanh(raw_i / u_max).
```

This is a differentiable component-wise bound. A soft penalty discourages force slew, but there is no hard rate, bandwidth, latency, thermal or power constraint.

### Open-loop optimisation

The decision variable is the complete raw control sequence. JAX differentiates through the numerical rollout and Adam updates the sequence. The objective combines position tracking, applied-force effort and applied-force slew.

The optimiser is warm-started from the nominal feedback-reference commands. No global optimality, recursive feasibility, robust optimality or formal stability guarantee is claimed.

### Feedback reference

The comparison controller combines nominal-model feed-forward with proportional position-error and derivative velocity-error feedback. Gains are selected from a fixed finite grid using the same nominal composite objective used to score the optimised sequence.

It assumes exact full-state feedback, known target kinematics and no sensing delay. It is not model-predictive, adaptive or reinforcement-learning control and no closed-loop stability proof is supplied.

### Mismatch study

The deterministic grid varies only mass, damping and linear stiffness:

- mass: `0.85, 1.00, 1.15`;
- damping: `0.75, 1.00, 1.25`;
- linear stiffness: `0.90, 1.10`.

This gives 18 sampled cases. “Maximum” means maximum among these cases, not a mathematical worst case over a continuous uncertainty set.

## 2. Acoustic-holography experiment

### Array and propagation model

The default geometry is an 8 x 8 flat single-sided array with 10 mm pitch, driven at 40 kHz in a homogeneous air-like medium (`c = 343 m/s`, `rho = 1.20 kg/m^3`). The requested target is 60 mm above the centre of the array.

Each source is an **isotropic point source** with complex contribution proportional to

```text
exp(i (k d + phase)) / d.
```

The source-amplitude scale is arbitrary and uncalibrated. Absolute pressure, potential, force and stiffness numbers are therefore model units rather than hardware predictions.

The propagation model omits finite piston directivity, measured transducer response, baffles, reflections, scattering, absorption, nonlinear acoustics, acoustic streaming, mutual coupling and apparatus geometry beyond source positions.

### Particle and Gor'kov regime

The model uses a 0.3 mm-radius positive-contrast particle. At 40 kHz in the configured medium, `ka ~= 0.22`, keeping the default benchmark in a small-particle regime for the Gor'kov approximation.

The time-averaged Gor'kov potential is computed from the complex pressure and its spatial derivatives. Model radiation force is

```text
F = -grad(U).
```

The implementation does not model particle deformation, back-scattering, rotation, non-spherical shape or multiple-particle interactions.

### Local trapping criterion

The code evaluates the **full 3 x 3 spatial Hessian** of `U`. Its eigenvalues are the principal local stiffnesses used by the benchmark. Positive values in all three principal directions imply locally restoring curvature of the model potential.

Positive curvature alone is not enough to place the requested target at equilibrium, so the optimisation objective also penalises non-zero `||grad(U)||`. `evaluate()` reports both the acoustic-force norm and a local Newton estimate of the displacement from the requested target to a nearby stationary point.

### Gravity and “levitation” language

**Gravity is not included.** The source-amplitude scale is not calibrated, so the benchmark cannot establish that acoustic force would support the particle's weight. Consequently:

- `positive_definite_hessian=True` means locally restoring curvature in the simplified acoustic model;
- a small force norm means the requested target is close to a model stationary point;
- neither result is a demonstration of physical levitation.

Any laboratory levitation claim would require calibrated acoustic amplitudes/directivity, gravity, apparatus geometry and experimental validation.

### Acoustic optimisation

The 64 transducer phases are initialised from phase conjugation plus a deterministic seeded perturbation. Adam minimises a dimensionless objective combining:

1. pressure magnitude at the requested target, normalised to the phase-conjugation focus;
2. acoustic-force norm at the target, normalised to the focus baseline; and
3. a smooth minimum of the three principal stiffnesses, normalised to a baseline curvature scale.

The reported solution is seed- and hyperparameter-dependent; no global optimum is claimed. The visual two-lobed field is described only as qualitatively twin-trap-like, not as a proof that the solution is identical to a specific experimental trap.

### Literature relationship

The design is inspired by the single-sided phased-array trapping framework in:

Marzo, A., Seah, S. A., Drinkwater, B. W., et al. (2015), *Holographic acoustic elements for manipulation of levitated objects*, Nature Communications 6, 8661, DOI `10.1038/ncomms9661`.

That work uses a more realistic transducer model and laboratory validation. This repository does not claim to reproduce its calibrated apparatus or performance.

## 3. Validation scope

Automated tests and generated-product validators check implementation invariants and the stored benchmark behaviours. They do not provide:

- a formal proof of numerical convergence in every regime;
- a control-theoretic stability proof;
- a robust-control certificate;
- an acoustic hardware model validation;
- an experimental levitation demonstration;
- a guarantee of performance under untested parameters or geometries.

The repository is intended to make the assumptions visible and auditable so that later higher-fidelity or hardware work has a clear starting point.
