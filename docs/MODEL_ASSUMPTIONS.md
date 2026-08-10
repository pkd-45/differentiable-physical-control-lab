# Model assumptions and limitations

This document defines the scientific scope of the benchmark and the assumptions behind the reported metrics. The implementation is intentionally small and inspectable; it is a methods demonstrator, not a validated model of a particular experimental apparatus.

## 1. Plant model and notation

The simulated plant is an isotropic two-dimensional **Duffing-type oscillator**,

```text
m x_ddot + c x_dot + k x + alpha ||x||^2 x = u,
```

where `x = [x, y]` is position, `u = [ux, uy]` is the applied control force, and `m`, `c`, `k`, and `alpha` are scalar mass, damping, linear-stiffness, and cubic-stiffness parameters. The state is `[x, y, vx, vy]`.

The cubic term is radial: `alpha ||x||^2 x`. This couples the two spatial axes through the displacement magnitude while preserving rotational symmetry. It is a compact nonlinear spring model with an analytic mechanical-energy function,

```text
E = 1/2 m ||v||^2 + 1/2 k ||x||^2 + 1/4 alpha ||x||^4.
```

The nominal parameters in `src/physctrl/dynamics.py` are **normalised benchmark values**. No SI or laboratory units are assigned. Consequently, reported positions, forces, time steps, RMSE values, and energies should be interpreted in consistent normalised units rather than as metres, newtons, seconds, or joules.

The model omits process noise, external disturbances, cross-axis anisotropy, unmodelled modes, actuator dynamics, sensing dynamics, and other effects that would be required for a higher-fidelity physical system.

## 2. Numerical integration and time convention

The state is advanced with semi-implicit Euler: velocity is updated first and position second. This method is inexpensive, differentiable, and suitable for the short benchmark rollout, but it is first order and is not presented as a high-accuracy long-duration integrator.

The nominal time step is `dt = 0.04` and the default benchmark uses 120 control steps. These values are in the same normalised time system as the plant parameters.

`simulate()` returns the **post-step** state for each control sample. The target array is therefore treated as a discrete sequence of waypoints used by the benchmark objective, not as a claim of exact continuous-time tracking at externally calibrated timestamps. The first waypoint is the initial position, which produces an initial hold step before the figure-eight path develops.

The validation suite includes a time-step-refinement test. That checks numerical behaviour of this implementation; it does not establish convergence for arbitrary trajectories, parameter regimes, or long integration horizons.

## 3. Actuator model

The optimiser and feedback reference both generate unconstrained raw control commands. Each component is mapped to the applied force through

```text
u_i = u_max * tanh(raw_i / u_max).
```

This provides a smooth differentiable approximation to a component-wise actuator limit. Each applied component satisfies `|u_i| < u_max`; the two-dimensional force norm can approach `sqrt(2) * u_max` when both components saturate simultaneously.

The objective also penalises the difference between consecutive **applied** force vectors. This is a soft slew regulariser only. There is no hard actuator-rate constraint, actuator bandwidth model, quantisation model, latency model, or power/thermal constraint.

Because the map is a smooth saturation rather than hard clipping, large changes in the raw command can produce small changes in applied force near saturation. Optimisation therefore occurs in raw-command space while effort and slew penalties are evaluated on the applied force.

## 4. Target trajectory and information available to the methods

The default reference is a predefined periodic figure-eight trajectory. It is known in full before the benchmark begins.

The open-loop optimiser has access to the entire target sequence. The feedback reference also uses the predefined target sequence: target velocity and acceleration are precomputed with periodic centred finite differences and used in the nominal-model feed-forward term.

The periodic derivative convention is appropriate for the closed, smooth default figure-eight reference. It should not be reused unchanged for a non-periodic trajectory, where endpoint treatment would need to be defined separately.

The feedback term uses the current **simulated state exactly**. There is no observation noise, estimator, partial observability, sensing latency, packet loss, or state-reconstruction error. The state-feedback calculation is closed loop, but the benchmark does not claim that the reference trajectory or its derivatives are generated causally online.

## 5. Differentiable open-loop trajectory optimisation

The optimised decision variable is the full sequence of raw two-axis control commands. JAX automatic differentiation propagates gradients through the complete numerical rollout and the controls are updated with Adam.

The objective combines:

1. mean squared position-tracking error;
2. applied-force effort;
3. applied-force slew regularisation.

The optimiser is warm-started from the nominal feedback-reference command sequence. This makes the reported optimisation a refinement of a physically informed starting point rather than a comparison against a deliberately weak initialisation.

The benchmark does **not** establish global optimality. Adam, the learning rate, the iteration count, the warm start, and the non-convex dynamics all influence the solution found. No formal stability, recursive feasibility, disturbance rejection, or robust-optimality guarantee is claimed for the open-loop solution.

The resulting sequence is fixed after optimisation. During the mismatch sweep it is replayed without observing or correcting the perturbed plant state.

## 6. Model-based feedback reference

The comparison controller consists of:

1. feed-forward force computed from the nominal Duffing-type model and the predefined target position, velocity, and acceleration;
2. proportional feedback on position error;
3. derivative feedback on velocity error;
4. the same smooth actuator saturation used by the optimiser.

The proportional and derivative gains are selected from a fixed documented grid using the same nominal composite objective used for trajectory optimisation. This avoids choosing an intentionally weak baseline, but the grid is finite and is not claimed to produce globally optimal gains.

The controller assumes full-state feedback and exact target kinematics. It is a model-based feed-forward + PD reference, not model-predictive control, reinforcement learning, adaptive control, or a formally robust controller. No closed-loop stability proof is supplied.

Under plant mismatch, the feed-forward calculation continues to use the **nominal** parameters while the feedback term reacts to the perturbed simulated state. This is the mechanism by which the reference can correct some model error that a fixed open-loop sequence cannot.

## 7. Model-mismatch sensitivity study

The benchmark evaluates both methods over 18 deterministic combinations of:

- mass factors: `0.85, 1.00, 1.15`;
- damping factors: `0.75, 1.00, 1.25`;
- linear-stiffness factors: `0.90, 1.10`.

Only these three parameters are perturbed. Cubic stiffness, actuator limit, target trajectory, numerical time step, state measurement, and controller settings are held fixed. No stochastic disturbance is injected.

For the open-loop method, the nominal optimised sequence is replayed unchanged on each perturbed plant. For the feedback reference, the nominal model remains in the feed-forward calculation while state feedback is recomputed at every step from the perturbed simulated state.

The reported median, 95th-percentile, maximum, and plotted case-by-case RMSE values describe **sensitivity over this finite grid only**. Terms such as “worst case” in the outputs mean the worst sampled case, not the mathematical worst case over a continuous uncertainty set.

The experiment therefore demonstrates model-mismatch sensitivity and the qualitative value of state feedback. It is not a robust-control certification or a proof that either method will behave similarly outside the sampled parameter range.

## 8. Interpretation of the nominal comparison

The two approaches have different structures:

- the differentiable method optimises one fixed control sequence against the nominal model;
- the reference combines model-based feed-forward with state feedback.

Both methods know the predefined target, use the same plant model for nominal design, use the same actuator map, and are evaluated with the same tracking metric. However, they are not information- or architecture-identical production controllers.

The benchmark is therefore intended to illustrate method behaviour, not to establish universal superiority. In the v0.2.0 reference run, the optimised open-loop sequence performs slightly better on the nominal model, while the feedback reference is less sensitive across the sampled parameter-mismatch grid.

## 9. Relation to acoustic levitation and experimental control

No acoustic-levitation model is implemented. In particular, the repository contains no:

- acoustic field or wave solver;
- radiation-force model;
- transducer geometry or phase optimisation;
- scattering or fluid-medium model;
- camera or sensor model;
- sensing/actuation latency model;
- hardware communication interface;
- laboratory calibration or experimental data.

The connection to physical-control research is methodological: differentiable nonlinear simulation, trajectory optimisation, bounded actuation, model-based feedback, and model-mismatch analysis. Extending the benchmark to acoustic levitation or another real apparatus would require a validated domain model, calibrated parameters, sensing and actuator dynamics, and experimental or trusted high-fidelity simulation data.

## 10. Scope of validation

The automated tests exercise numerical and physical properties of this implementation, including bounded actuation, equilibrium behaviour, dissipative energy behaviour, finite gradients, objective reduction, time-step refinement, feedback response under mismatch, and completeness of the predefined sensitivity grid.

Passing those tests establishes reproducibility and internal consistency of the benchmark. It does not constitute experimental validation, controller certification, or evidence of safety for a physical system.
