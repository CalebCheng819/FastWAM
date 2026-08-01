#!/usr/bin/env python3
"""Fail closed when the active Python environment drifts from pyproject pins."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path


CRITICAL_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "accelerate",
    "deepspeed",
    "hydra-core",
    "h5py",
    "numpy",
    "transformers",
)
_NAME_NORMALIZER = re.compile(r"[-_.]+")
_EXACT_REQUIREMENT = re.compile(
    r"^\s*([A-Za-z0-9_.-]+)(?:\[[^]]+\])?\s*==\s*([^\s;]+)(?:\s*;.*)?$"
)


def _canonical_name(name: str) -> str:
    return _NAME_NORMALIZER.sub("-", name).lower()


def _fallback_project_dependencies(text: str) -> list[str]:
    """Parse the simple project.dependencies string array on Python 3.10."""

    in_project = False
    in_dependencies = False
    dependencies: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]") and not in_dependencies:
            in_project = line == "[project]"
            continue
        if not in_project:
            continue
        if not in_dependencies:
            if re.match(r"^dependencies\s*=\s*\[\s*$", line):
                in_dependencies = True
            continue
        if line == "]":
            return dependencies
        match = re.match(r'^\s*["\']([^"\']+)["\']\s*,?\s*(?:#.*)?$', raw_line)
        if not match:
            raise ValueError(f"unsupported project.dependencies line: {raw_line!r}")
        dependencies.append(match.group(1))
    raise ValueError("pyproject.toml has no complete [project].dependencies array")


def _project_dependencies(pyproject_path: Path) -> list[str]:
    raw = pyproject_path.read_bytes()
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        return _fallback_project_dependencies(raw.decode("utf-8"))

    document = tomllib.loads(raw.decode("utf-8"))
    dependencies = document.get("project", {}).get("dependencies")
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise ValueError("pyproject.toml [project].dependencies must be an array of strings")
    return dependencies


def _critical_exact_pins(pyproject_path: Path) -> dict[str, str]:
    critical = {_canonical_name(name) for name in CRITICAL_DISTRIBUTIONS}
    seen: dict[str, str] = {}
    non_exact: set[str] = set()
    for requirement_text in _project_dependencies(pyproject_path):
        match = _EXACT_REQUIREMENT.match(requirement_text)
        if match:
            name = _canonical_name(match.group(1))
            if name in critical:
                if name in seen and seen[name] != match.group(2):
                    raise ValueError(f"conflicting exact pins for critical distribution {name}")
                seen[name] = match.group(2)
            continue
        approximate_name = _canonical_name(re.split(r"[<>=!~;\s\[]", requirement_text, maxsplit=1)[0])
        if approximate_name in critical:
            non_exact.add(approximate_name)

    missing = sorted(critical - seen.keys())
    if non_exact:
        raise ValueError(
            "critical distributions must use exact == pins: " + ", ".join(sorted(non_exact))
        )
    if missing:
        raise ValueError("critical exact pins missing from pyproject.toml: " + ", ".join(missing))
    return seen


def _run_pip_check(timeout_s: int) -> bool:
    check_env = None
    pip_source = "installed-module"
    if importlib.util.find_spec("pip") is None:
        # Some immutable training venvs were built by uv without installing
        # pip.  Python's stdlib ensurepip still ships a signed/versioned pip
        # wheel.  Importing directly from that wheel runs the same read-only
        # ``pip check`` without mutating the shared environment on CPFS.
        try:
            import ensurepip

            bundled = Path(ensurepip.__file__).resolve().parent / "_bundled"
            wheels = sorted(bundled.glob("pip-*.whl"))
        except (ImportError, OSError) as error:
            print(
                f"Error: pip is absent and bundled ensurepip could not be read: {error}",
                file=sys.stderr,
            )
            return False
        if len(wheels) != 1 or not wheels[0].is_file():
            print(
                "Error: pip is absent and ensurepip does not contain exactly one "
                f"bundled pip wheel: {[str(path) for path in wheels]}",
                file=sys.stderr,
            )
            return False
        check_env = os.environ.copy()
        existing_pythonpath = check_env.get("PYTHONPATH")
        check_env["PYTHONPATH"] = str(wheels[0]) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        pip_source = f"ensurepip-wheel:{wheels[0].name}"

    command = [sys.executable, "-m", "pip", "check"]
    try:
        result = subprocess.run(
            command,
            env=check_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"Error: pip check timed out after {timeout_s}s.", file=sys.stderr)
        return False

    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        print(f"Error: pip check failed with status={result.returncode}.", file=sys.stderr)
        return False
    print(f"[python_env] pip_check=PASS source={pip_source}")
    return True


def validate_environment(pyproject_path: Path, pip_check_timeout_s: int) -> int:
    try:
        pins = _critical_exact_pins(pyproject_path)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"Error: cannot load critical exact pins from {pyproject_path}: {error}", file=sys.stderr)
        return 2

    versions_ok = True
    for name in CRITICAL_DISTRIBUTIONS:
        canonical = _canonical_name(name)
        expected = pins[canonical]
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            print(
                f"[python_env] package={name} expected={expected} actual=MISSING status=MISSING",
                file=sys.stderr,
            )
            versions_ok = False
            continue
        if actual != expected:
            print(
                f"[python_env] package={name} expected={expected} actual={actual} status=MISMATCH",
                file=sys.stderr,
            )
            versions_ok = False
        else:
            print(f"[python_env] package={name} expected={expected} actual={actual} status=PASS")

    pip_ok = _run_pip_check(pip_check_timeout_s)
    if not versions_ok or not pip_ok:
        print("Error: Python environment preflight failed closed.", file=sys.stderr)
        return 1
    print(f"[python_env] status=PASS critical_packages={len(CRITICAL_DISTRIBUTIONS)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pyproject.toml",
    )
    parser.add_argument("--pip-check-timeout", type=int, default=120)
    args = parser.parse_args()
    if args.pip_check_timeout < 1:
        parser.error("--pip-check-timeout must be positive")
    return validate_environment(args.pyproject.resolve(), args.pip_check_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
