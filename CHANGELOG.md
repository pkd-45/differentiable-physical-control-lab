# Changelog

## 0.2.1 - 2026-08-11
- Corrected and tightened README, model-assumption, validation, and reproducibility documentation.
- Added a Python 3.11 reference-environment snapshot from the independently reproduced macOS run.
- Strengthened the feedback-under-mismatch test so it verifies that plant-induced state divergence changes subsequent feedback commands.
- Fixed Ruff import ordering and sorted public exports reported by the clean macOS validation run.
- Added an optional output directory to the demo runner so local reproduction need not overwrite committed reference products.
- Added consolidated release and validation scripts; no change was made to the Duffing dynamics, objective, optimiser, feedback law, or mismatch grid.
- Reproduced the complete fresh-install workflow on Apple-silicon macOS with Python 3.11.15: Ruff passed, all 13 tests passed, the benchmark completed, all six headline scientific checks passed, and the frozen release-tree manifest revalidated after execution.
- Confirmed the repository-owned source remains MIT licensed; direct Python dependencies are installed rather than vendored and retain their own licenses.
- Added PEP 639 package-license metadata, README/author/keyword/classifier metadata, and explicit major-version dependency ceilings for a cleaner public Python package definition.
- Raised the declared minimum Python version to 3.11 to match the environment validated end to end for this release.
- Removed accidentally packaged Python bytecode/cache artifacts and hardened source-tree hygiene checks so generated files cannot enter a frozen release unnoticed.
- Hardened GitHub Actions with read-only permissions, cancellation of superseded runs, explicit package-metadata and compile checks, isolated benchmark outputs, and shared scientific-output validation.
- Corrected README Python-support wording to match the actual `Requires-Python: >=3.11` package metadata.
- Corrected reproducibility output paths for isolated validation runs and clarified sampled-maximum mismatch wording in the public results/plot title.
- Removed the final context-specific wording from the public README and added canonical GitHub repository/issue URLs to package metadata for the publication-ready tree.

## 0.2.0 - 2026-08-10
- Replaced the initial generic spring benchmark with an explicit two-dimensional controlled Duffing-type oscillator.
- Changed the reference path to a smooth figure-eight beginning at the initial position.
- Added a nominal-model feed-forward plus PD closed-loop reference controller.
- Tuned reference gains deterministically using the same composite nominal objective used to evaluate the optimiser.
- Clarified that the differentiable method is open-loop trajectory optimisation, not robust control.
- Expanded the robustness evaluation to an 18-case parameter-mismatch grid and compared fixed open-loop replay with closed-loop feedback response.
- Added force and slew metrics, mismatch outputs, and sampled worst-case comparison plots.
- Expanded tests for gradients, edge cases, dissipative physics, actuator constraints, feedback response under mismatch, and robustness-sweep completeness.

## 0.1.0 - 2026-08-10
- Initial differentiable physical-control benchmark.
