# Validation strategy

This benchmark separates three kinds of evidence that should not be conflated:

1. **automated invariants and numerical sanity checks** enforced by `pytest`;
2. **nominal benchmark observations** produced by the deterministic default run;
3. **sampled model-mismatch observations** over a fixed 18-case parameter grid.

The test suite checks properties of the implementation. The benchmark figures and JSON files report observed performance for the stated model, target, optimiser settings, controller tuning grid, and mismatch grid. Neither is a proof of general controller superiority or formal robust stability.

## 1. Automated checks

The repository currently contains 13 tests.

### Dynamics and numerical integration

Tests verify that:

- each applied force component respects the configured smooth actuator bound;
- the zero state remains an equilibrium under zero input;
- for the tested damped unforced rollout, discrete mechanical energy is non-increasing within a numerical tolerance and decays substantially over the run;
- halving the time step from 0.02 to 0.01 and then to 0.005 reduces the final-state refinement difference, with the finer difference required to be less than 60% of the coarser difference;
- malformed control arrays raise a `ValueError` rather than being silently accepted.

The time-step test is evidence of convergence behaviour for this validation case. It is **not** a general error bound or a formal order-of-accuracy proof for every trajectory.

### Target, objective, optimisation, and feedback

Tests verify that:

- the default target begins at the initial position;
- reference velocity and acceleration arrays have the expected shapes and finite values;
- automatic differentiation of the trajectory objective returns a finite gradient with the correct shape;
- the one-step objective remains finite;
- optimisation from the model-based reference warm start reduces the same composite objective by at least 2% in the reduced test problem;
- optimised applied-force components remain within the configured component limit;
- feedback rollouts remain finite under a perturbed plant;
- when plant mismatch changes the simulated state trajectory, subsequent feedback control commands change as well, distinguishing the state-feedback implementation from replay of a fixed open-loop sequence;
- the robustness sweep contains all 18 configured cases and finite RMSE values.

These checks exercise implementation properties. They do not establish closed-loop stability for arbitrary parameters or disturbances.

## 2. Nominal benchmark comparison

The reference controller is not an arbitrary low-gain PD baseline. It contains:

1. nominal-model feed-forward from the predefined target position, velocity, and acceleration;
2. proportional feedback on position error;
3. derivative feedback on velocity error;
4. the same smooth actuator mapping used for the optimised sequence.

A fixed 7 x 7 `(Kp, Kd)` grid is searched on the nominal plant using the same composite objective used by trajectory optimisation. The selected gains and nominal metrics are written to `products/metrics.json`.

For the checked v0.2.1 reference run:

| Metric | Differentiable open loop | Feedback reference |
| --- | ---: | ---: |
| tracking RMSE | 0.066962 | 0.071480 |
| composite objective | 0.013405 | 0.015672 |

The differentiable open-loop sequence therefore achieved both slightly lower nominal tracking RMSE and a lower composite objective in this particular benchmark. This is an **observed result**, not an assertion that open-loop optimisation is generally preferable to feedback control.

The optimiser is initialised from the tuned reference-controller command sequence, so the nominal comparison should be interpreted as asking whether gradient-based trajectory optimisation can improve that warm start under the nominal differentiable model.

## 3. Model-mismatch sensitivity

Both approaches are evaluated over the same fixed 18-case grid:

- mass factors: `0.85, 1.00, 1.15`;
- damping factors: `0.75, 1.00, 1.25`;
- linear-stiffness factors: `0.90, 1.10`.

For the differentiable method, the nominally optimised control sequence is replayed unchanged on each perturbed plant. For the feedback reference, the feed-forward calculation continues to use the nominal model, while the feedback terms are recomputed from the current perturbed simulated state.

For the checked v0.2.1 run:

| Sampled mismatch statistic | Fixed open loop | Feedback reference |
| --- | ---: | ---: |
| median RMSE | 0.117078 | 0.074047 |
| 95th-percentile RMSE | 0.164752 | 0.110870 |
| maximum RMSE | 0.183617 | 0.115777 |

The feedback reference had lower RMSE than the fixed open-loop sequence in **all 18 sampled mismatch cases** in the stored v0.2.1 result.

That result supports a narrow conclusion: for this model, target, tuning procedure, and sampled uncertainty grid, state feedback reduced sensitivity to the tested parameter mismatch relative to replaying the nominal open-loop sequence. It does **not** establish a robust-control guarantee or general superiority of the feedback controller.

Here, “maximum” or “worst case” means the largest value among the 18 sampled cases only. It is not a mathematical worst-case search over a continuous uncertainty set.

## 4. Generated validation artefacts

A successful default run writes:

- `products/metrics.json` — nominal and aggregate mismatch metrics;
- `products/robustness_cases.json` — all 18 parameter-factor combinations and RMSE values;
- `products/trajectory.png` — nominal target and realised trajectories;
- `products/optimisation_history.png` — objective history during Adam optimisation;
- `products/robustness.png` — case-by-case mismatch comparison;
- `products/worst_case_comparison.png` — trajectories for the sampled case with the largest open-loop RMSE.

The optimisation-history figure is diagnostic. Because Adam is not a monotone line-search method, the repository does not require every individual iteration to reduce the objective.

## 5. Reproduction checks

The intended local checks are:

```bash
pytest -v
python -m compileall -q src tests
python -m physctrl.experiment
```

The stored v0.2.1 release was reproduced on macOS with Python 3.11 and produced 13 passing tests and the metrics above. Exact final decimals may vary slightly with JAX/XLA versions and hardware, so automated tests target physical and numerical properties rather than hard-coding every benchmark decimal.

## Claims deliberately not made

This repository does **not** claim:

- real-time execution or timing guarantees;
- experimental or hardware validation;
- acoustic-levitation model fidelity;
- formal closed-loop stability guarantees;
- formal robust-control guarantees;
- global optimality of the Adam-optimised sequence;
- reinforcement-learning capability;
- neural-network or learned-dynamics capability;
- that either approach is generally superior outside the stated benchmark.

See `MODEL_ASSUMPTIONS.md` for the model, sensing, reference-preview, actuator, integration, and uncertainty assumptions that bound these validation results.
