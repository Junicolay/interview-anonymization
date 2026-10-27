
"""Centralized paths for the interview-anonymization project."""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by locating the src/paths.py file."""
    start = (start or Path.cwd()).resolve()

    for candidate in (start, *start.parents):
        if (
            (candidate / "src" / "paths.py").is_file()
            and (candidate / "notebooks").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find the project root. "
        "Expected src/paths.py and notebooks/."
    )


PROJECT_ROOT = find_project_root(Path(__file__).parent)

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
RESULTS_DIR = DATA_DIR / "results"

# VoxCeleb1 dataset
VOXCELEB_ROOT = RAW_DATA_DIR / "voxceleb1"
VOXCELEB_WAV_DIR = VOXCELEB_ROOT / "wav"

# Project outputs
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
