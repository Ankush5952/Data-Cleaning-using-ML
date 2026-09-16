"""Check and optionally install the project's Python dependencies."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
}


def missing_dependencies() -> list[str]:
    """Return package names whose import modules are unavailable."""
    return [
        package_name
        for module_name, package_name in DEPENDENCIES.items()
        if importlib.util.find_spec(module_name) is None
    ]


def main() -> int:
    missing = missing_dependencies()
    if not missing:
        print("All project dependencies are already installed.")
        return 0

    print("Missing dependencies:")
    for package_name in missing:
        print(f"  - {package_name}")
    answer = input("Install these packages now? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Installation cancelled.")
        return 1

    requirements_path = Path(__file__).with_name("requirements.txt")
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--requirement",
        str(requirements_path),
    ]
    subprocess.run(command, check=True)
    print("Dependencies installed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
