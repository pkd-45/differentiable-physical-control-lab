# Changelog

## 0.3.0 - 2026-08-11

- Added `physctrl.acoustic`, a simplified differentiable acoustic-holography benchmark for a single-sided 8 x 8, 40 kHz phased array.
- Added coherent point-source pressure superposition and a small-particle Gor'kov potential with automatic spatial derivatives.
- Added model acoustic-radiation-force evaluation and the full 3 x 3 Gor'kov Hessian; local restoring behaviour is assessed from its principal stiffness eigenvalues rather than diagonal entries alone.
- Added a target force-balance term to the acoustic phase objective so positive local curvature is not mistaken for equilibrium at a point with non-zero force.
- Changed the default model particle radius to 0.3 mm (`ka ~= 0.22`) to keep the demonstration in a clearer small-particle regime.
- Added deterministic seeded Adam optimisation of all 64 transducer phases from a perturbed phase-conjugation baseline.
- Added `scripts/run_acoustic_demo.py`, vectorised field-slice generation and `scripts/validate_acoustic_outputs.py`.
- Added 10 acoustic regression tests, bringing the suite to 23 tests total.
- Added stored acoustic metrics and figure. The reference model solution has a positive-definite Gor'kov Hessian, pressure magnitude about 1.18e3 times below the focus baseline, acoustic-force norm about 5.97e-4 of the focus-force baseline, and a 5.26 um local linearised equilibrium offset.
- Rewrote model assumptions, validation and reproducibility documentation to distinguish a simplified acoustic trapping model from calibrated levitation or hardware validation. Gravity, measured directivity and apparatus calibration remain out of scope.
- Expanded GitHub Actions compatibility checks to Python 3.11-3.13; complete nonlinear and acoustic generated-product benchmarks run on Python 3.11.
- Updated package metadata to version 0.3.0 and repaired the metadata validator accordingly.
- Nonlinear Duffing dynamics, trajectory objective, controller, mismatch grid and stored v0.2.1-derived nonlinear reference products are unchanged.

## 0.2.1 - 2026-08-11

- Tightened README, model-assumption, validation and reproducibility documentation.
- Added a Python 3.11 reference-environment snapshot from the independently reproduced macOS run.
- Strengthened the feedback-under-mismatch test so plant-induced state divergence changes subsequent feedback commands.
- Fixed Ruff import ordering and sorted public exports reported by the clean macOS validation run.
- Added an optional output directory so reproduction need not overwrite committed reference products.
- Added frozen-tree hygiene, package-metadata and generated-output validators.
- Reproduced the complete fresh-install workflow on Apple-silicon macOS with Python 3.11.15: Ruff passed, all 13 tests passed, the benchmark completed, scientific checks passed and the frozen release-tree manifest revalidated.
- Added MIT package metadata and explicit major-version dependency ceilings.
- Raised the declared minimum Python version to 3.11 to match the environment validated end to end for that release.
- Hardened GitHub Actions with read-only permissions, superseded-run cancellation and isolated benchmark outputs.

## 0.2.0 - 2026-08-10

- Replaced the initial generic spring benchmark with an explicit two-dimensional Duffing-type oscillator.
- Changed the reference path to a smooth figure-eight beginning at the initial position.
- Added a nominal-model feed-forward plus PD closed-loop reference controller.
- Tuned reference gains deterministically using the same composite nominal objective used to evaluate the optimiser.
- Clarified that the differentiable method is open-loop trajectory optimisation, not robust control.
- Expanded the mismatch evaluation to an 18-case parameter grid and compared fixed open-loop replay with feedback response.
- Added force/slew metrics and mismatch diagnostic plots.
- Expanded physical, gradient, constraint and mismatch tests.

## 0.1.0 - 2026-08-10

- Initial differentiable physical-control benchmark.
