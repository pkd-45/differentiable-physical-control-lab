# Reproducibility

## Reproducibility scope

This repository is designed to make the **benchmark procedure** reproducible: install the package, run the automated checks, execute the deterministic benchmark, and regenerate the JSON results and figures in `products/`.

The benchmark itself uses no random sampling or stochastic initialisation. The target trajectory, controller-gain grid, optimiser warm start, Adam update sequence, and 18-case parameter-mismatch grid are deterministic.

That does **not** imply bit-for-bit identical floating-point output on every machine. JAX/XLA versions, compiler behaviour, operating system, CPU architecture, and numerical libraries can change the last digits of floating-point results. The validation suite therefore checks physical and numerical properties rather than requiring exact decimal identity with the committed reference metrics.

## Python support versus validated environment

The package metadata declares:

```text
Python >= 3.11
```

That is the package compatibility floor, not a statement that every supported Python/dependency combination has been independently validated.

The reference v0.2.1 benchmark was independently reproduced on the author's Apple-silicon macOS system with:

| Component | Reference version |
|---|---:|
| Python | 3.11.15 |
| pip | 26.2.1 |
| JAX | 0.10.2 |
| jaxlib | 0.10.2 |
| NumPy | 2.4.6 |
| Matplotlib | 3.11.1 |
| pytest | 9.1.1 |
| Ruff | 0.16.2 |

The complete package snapshot from that successful environment is recorded in [`requirements/reference-py311.txt`](../requirements/reference-py311.txt). It is an **environment snapshot**, not a cryptographically hashed universal lockfile: some wheels and low-level dependencies are platform-specific.

## Recommended reproduction path

Python 3.11 is recommended because it matches both the validated macOS run and the GitHub Actions configuration.

On macOS or Linux, with a `python3.11` executable available:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

ruff check .
python -m pytest -q
python scripts/run_demo.py --output-dir validation_runs/manual
```

On Windows, activate the environment with:

```text
.venv\Scripts\activate
```

With the command above, a successful demo writes:

```text
validation_runs/manual/metrics.json
validation_runs/manual/robustness_cases.json
validation_runs/manual/trajectory.png
validation_runs/manual/optimisation_history.png
validation_runs/manual/robustness.png
validation_runs/manual/worst_case_comparison.png
```

Running `python scripts/run_demo.py` without `--output-dir` writes to `products/` instead.

Inspect the headline metrics with:

```bash
python -m json.tool validation_runs/manual/metrics.json
```

## Reproducing the reference package versions

For the closest software match to the validated macOS v0.2.1 run, create a fresh Python 3.11 environment and install the recorded snapshot before installing the local project without re-resolving dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/reference-py311.txt
python -m pip install --no-deps -e .

ruff check .
python -m pytest -q
python scripts/run_demo.py --output-dir validation_runs/manual
```

The snapshot is intended to record the successful reference environment. If a pinned wheel is unavailable on a different platform, use the normal installation path above and treat small numerical differences as expected unless the automated validation properties fail.

## Reference numerical result

The committed v0.2.1 benchmark reports approximately:

| Metric | Reference value |
|---|---:|
| Optimised open-loop RMSE | 0.066962 |
| Feedback-reference RMSE | 0.071480 |
| Optimised composite objective | 0.013405 |
| Feedback-reference composite objective | 0.015672 |
| Open-loop mismatch median RMSE | 0.117078 |
| Feedback mismatch median RMSE | 0.074047 |
| Open-loop sampled maximum RMSE | 0.183617 |
| Feedback sampled maximum RMSE | 0.115777 |
| Mismatch cases | 18 |

These values are **reference observations**, not exact-value test assertions. Interpretation and validation boundaries are documented in [`VALIDATION.md`](VALIDATION.md).

## Continuous integration

The GitHub Actions workflow uses Python 3.11 on GitHub's `ubuntu-latest` runner and performs:

```text
frozen source-tree hygiene check
editable package installation
installed-package metadata check
Ruff linting
pytest validation
Python compile check
end-to-end benchmark execution in validation_runs/ci
generated-product and headline scientific-behaviour validation
```

The CI environment deliberately installs from the dependency ranges declared in `pyproject.toml`. It therefore tests compatibility with the currently resolved dependency set; it is **not** an immutable historical environment because `ubuntu-latest` and compatible package versions can change over time.

The reference package snapshot exists separately so that these two purposes are not confused:

- **CI:** detect whether the project continues to work with current compatible dependencies;
- **reference snapshot:** record the versions used for the validated v0.2.1 run.

## Determinism and numerical variation

There is no explicit random-number generator in the benchmark. Re-running with the same software/hardware stack should therefore follow the same algorithmic path.

Nevertheless, exact floating-point identity is not the scientific acceptance criterion. A reproduction should be considered successful when:

1. installation completes;
2. linting and automated tests pass;
3. the end-to-end benchmark completes;
4. all expected products are created;
5. headline metrics remain qualitatively consistent with the documented nominal-versus-mismatch result; and
6. no validation invariant in `tests/` fails.

A materially different scientific conclusion, a failed physical/numerical invariant, non-finite output, or missing product should be investigated rather than dismissed as ordinary floating-point variation.

## Repository integrity

`MANIFEST.sha256` records the release-tree file hashes. It should be regenerated only after the repository contents are frozen for a release. Because documentation and validation files may change during development, a manifest from an earlier release should not be interpreted as valid for an edited working tree.

For a frozen release tree, verify it with:

```bash
shasum -a 256 -c MANIFEST.sha256
```

Every listed file should report `OK`.
