
"""Centralized paths for the interview-anonymization project."""

from pathlib import Path

# Project root: interview-anonymization/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

# VoxCeleb1 dataset
VOXCELEB_ROOT = RAW_DATA_DIR / "voxceleb1"
VOXCELEB_WAV_DIR = VOXCELEB_ROOT / "wav"

# Project outputs
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
