
"""General utilities for the project."""

import importlib.util
from pathlib import Path


def load_module(module_name: str, file_path: str | Path):
    """Load a Python module directly from its file path."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Module file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
