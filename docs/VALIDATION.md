# Validation strategy

The repository distinguishes **automated implementation checks** from **observed benchmark results**. Neither set of evidence is a proof of general controller optimality or hardware performance.

## 1. Test inventory

The v0.3.0 suite contains **23 tests**:

- 13 nonlinear-control tests;
- 10 acoustic-holography tests.

### Nonlinear dynamics and control

Tests cover:

- component-wise actuator bounds;
- preservation of the zero equilibrium under zero input;
- decay of mechanical energy in a damped unforced rollout;
- timestep-refinement behaviour;
- malformed-input rejection;
- target/reference shape and finiteness;
- finite autodiff gradients;
- finite one-step objectives;
- reduction of the nominal trajectory objective from its warm start;
- bounded optimised force;
- state-dependent feedback response under plant mismatch;
- complete and finite evaluation of all 18 mismatch cases.

These checks support the implementation of the stated benchmark, not a formal closed-loop stability theorem.

### Acoustic model and optimisation

Tests cover:

- centred planar 8 x 8 array geometry;
- a default particle with `ka < 0.3`;
- phase-conjugation focusing relative to random phases;
- invariance of pressure magnitude to a global phase offset;
- linear pressure scaling with the uncalibrated source-amplitude parameter;
- lack of fully restoring local curvature for the phase-conjugation baseline;
- finite non-zero gradients through the acoustic objective;
- finite-difference agreement with autodiff second derivatives;
- an optimised solution with a positive-definite Gor'kov Hessian, a low-pressure target, low residual acoustic force and a small linearised equilibrium offset;
- deterministic optimisation for a fixed seed.

The positive-definite test uses eigenvalues of the **full Hessian**, not only its diagonal entries.

## 2. Nonlinear benchmark observations

The stored reference comparison gives:

| Metric | Differentiable open loop | Feedback reference |
|---|---:|---:|
| tracking RMSE | 0.066962 | 0.071480 |
| composite objective | 0.013405 | 0.015672 |

The open-loop sequence is warm-started from the tuned feedback-reference command sequence and improves the nominal composite objective in this benchmark.

Across the fixed 18-case mismatch grid:

| Sampled statistic | Fixed open loop | Feedback reference |
|---|---:|---:|
| median RMSE | 0.117078 | 0.074047 |
| 95th-percentile RMSE | 0.164752 | 0.110870 |
| maximum RMSE | 0.183617 | 0.115777 |

The feedback reference has lower RMSE in all 18 stored mismatch cases. This supports only a sampled sensitivity conclusion; it is not a robust-control certificate.

## 3. Acoustic benchmark observations

The acoustic reference compares a phase-conjugation focus with the optimised 64-phase hologram at the requested target.

### Stored reference

| Quantity | Focus baseline | Optimised hologram |
|---|---:|---:|
| pressure magnitude (model units) | 945.31 | 0.7983 |
| minimum principal stiffness | -1.50e-6 | +7.75e-8 |
| positive-definite Hessian | no | yes |
| acoustic-force ratio to focus | 1.0 | 5.97e-4 |
| linearised equilibrium offset | 4.62e-3 m | 5.26e-6 m |

The generated-product validator additionally requires:

- 64 transducers;
- `ka < 0.3`;
- pressure-node ratio > 50 relative to focus;
- positive-definite full Gor'kov Hessian;
- positive weakest principal stiffness;
- target acoustic-force norm < 1% of the focus-force baseline;
- linearised equilibrium offset < 1% of the wavelength;
- reduced optimisation objective;
- finite reported metrics.

The stored run comfortably meets those thresholds.

### Interpretation boundary

The acoustic result establishes that the **simplified model** contains a low-pressure, locally restoring near-equilibrium at the requested point after phase optimisation. It does not establish physical levitation because gravity, calibrated acoustic amplitude/directivity and experimental validation are absent.

The x-z pressure slice is qualitatively twin-lobed. It may be described as twin-trap-like or literature-inspired, but the visual pattern alone is not treated as a formal classification or hardware result.

## 4. Generated products

### Nonlinear-control products

- `metrics.json`
- `robustness_cases.json`
- `trajectory.png`
- `optimisation_history.png`
- `robustness.png`
- `worst_case_comparison.png`

### Acoustic products

- `acoustic_metrics.json`
- `acoustic_hologram.png`

The acoustic JSON stores the baseline and optimised pressure, principal stiffnesses, force diagnostics, local equilibrium-offset estimate and optimiser metadata.

## 5. Claims deliberately not made

The validation evidence does **not** establish:

- real-time execution guarantees;
- acoustic hardware or laboratory validation;
- levitation against gravity;
- calibrated acoustic force/stiffness;
- formal controller stability;
- formal robust-control guarantees;
- global optimality of the Adam solutions;
- reinforcement-learning capability;
- universal performance outside the stated models and parameter ranges.

See `MODEL_ASSUMPTIONS.md` for the assumptions that bound these results.
