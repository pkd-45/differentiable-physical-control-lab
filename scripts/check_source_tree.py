"""Fail if generated/local artifacts are present in a frozen source tree."""
from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    "validation_runs",
}
FORBIDDEN_FILE_NAMES = {".DS_Store", ".coverage", "coverage.xml"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    bad: list[Path] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if ".git" in rel.parts:
            continue
        if path.is_dir() and (
            path.name in FORBIDDEN_DIRS or path.name.endswith(".egg-info")
        ):
            bad.append(rel)
            continue
        if path.is_file() and (
            path.name in FORBIDDEN_FILE_NAMES or path.suffix in FORBIDDEN_SUFFIXES
        ):
            bad.append(rel)
    if bad:
        print("FAIL: generated/local artifacts are present in the frozen source tree:")
        for item in sorted(set(bad), key=str):
            print(f"  {item}")
        return 1
    print("PASS: frozen source tree contains no generated/local artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
