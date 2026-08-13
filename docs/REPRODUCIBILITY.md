# Reproducibility

This repository separates **recorded reference products** from **compatibility testing**. Exact floating-point identity across JAX/XLA versions and hardware is not required; physical and numerical invariants are.

## 1. Python support and CI

Package metadata requires Python 3.11 or newer. GitHub Actions runs code-quality and test checks on:

- Python 3.11;
- Python 3.12;
- Python 3.13.

The two complete generated-product benchmarks run on the Python 3.11 CI job to avoid tripling the relatively expensive differentiable-physics compilation work.

CI performs:

```text
source-tree hygiene
editable installation
package-metadata validation
Ruff
pytest
compileall
nonlinear-control benchmark + product validator (Python 3.11)
acoustic-hologram benchmark + product validator (Python 3.11)
```

## 2. Recorded v0.2.1 nonlinear-control environment

The v0.2.1 nonlinear-control release was independently reproduced on Apple-silicon macOS with:

```text
Python       3.11.15
pip          26.2.1
JAX          0.10.2
jaxlib       0.10.2
NumPy        2.4.6
Matplotlib   3.11.1
pytest       9.1.1
Ruff         0.16.2
```

The complete package snapshot is stored in `requirements/reference-py311.txt`.

That run passed Ruff, 13 tests, the end-to-end nonlinear benchmark, scientific-output checks and the frozen-file manifest.

## 3. v0.3.0 acoustic reference generation

The stored v0.3.0 acoustic products are generated deterministically with `seed=0`. The acoustic optimiser therefore contains explicit pseudorandom initialisation, unlike the nonlinear-control optimiser. Reproduction requires the seed as well as algorithm settings.

The reference algorithm uses 1500 Adam steps and starts from phase conjugation plus a small seeded phase perturbation. The acceptance checks do not hard-code every final decimal. They require the scientific behaviour documented in `VALIDATION.md`.

## 4. Standard reproduction

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

ruff check .
pytest -q
python -m compileall -q src tests scripts

python scripts/run_demo.py --output-dir validation_runs/control
python scripts/validate_outputs.py validation_runs/control

python scripts/run_acoustic_demo.py --output-dir validation_runs/acoustic
python scripts/validate_acoustic_outputs.py validation_runs/acoustic
```

On systems where JAX compilation causes high memory use when all tests share one long process, the test files may be run in separate invocations without changing the test definitions:

```bash
pytest -q tests/test_control.py tests/test_dynamics.py
pytest -q tests/test_acoustic.py -m 'not slow'
pytest -q tests/test_acoustic.py -m slow
```

## 5. Reference observations

### Nonlinear-control experiment

| Metric | Stored value |
|---|---:|
| Optimised open-loop RMSE | 0.066962 |
| Feedback-reference RMSE | 0.071480 |
| Open-loop mismatch median RMSE | 0.117078 |
| Feedback mismatch median RMSE | 0.074047 |
| Open-loop sampled maximum RMSE | 0.183617 |
| Feedback sampled maximum RMSE | 0.115777 |
| Mismatch cases | 18 |

### Acoustic-holography experiment

| Metric | Stored value |
|---|---:|
| Transducers | 64 |
| Particle `ka` | 0.2198 |
| Focus pressure magnitude (model units) | 945.31 |
| Optimised pressure magnitude (model units) | 0.7983 |
| Pressure-node ratio | 1184.1 |
| Minimum principal stiffness (model units) | 7.75e-8 |
| Acoustic-force ratio to focus | 5.97e-4 |
| Linearised equilibrium offset | 5.26e-6 m |

These values are observations for the reference configuration, not claims of calibrated hardware performance.

## 6. Numerical variation and acceptance

A reproduction is scientifically acceptable when:

1. installation succeeds;
2. linting and tests pass;
3. expected products are generated;
4. all reported values are finite;
5. the nonlinear generated-product validator passes;
6. the acoustic validator passes; and
7. no physical/numerical invariant in the tests fails.

Small floating-point drift is expected across JAX/XLA versions, compilers and hardware. A changed scientific conclusion, failed invariant, missing product or non-finite value should be investigated rather than dismissed as ordinary numerical variation.

## 7. Frozen-file integrity

`MANIFEST.sha256` records the release-tree hashes. It should be regenerated only after the repository is frozen.

Verify a frozen tree with:

```bash
shasum -a 256 -c MANIFEST.sha256
```
