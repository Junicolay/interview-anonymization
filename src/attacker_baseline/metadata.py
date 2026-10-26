
"""Utilities for processing VoxCeleb metadata."""

import pandas as pd


def normalize_language_metadata(language_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize language predictions and extract speaker/video IDs."""
    df = language_df.copy()

    normalized_filenames = (
        df["filename"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    path_parts = normalized_filenames.str.split("/")

    df["speaker_id"] = path_parts.str[-3]
    df["video_id"] = path_parts.str[-2]

    df["language"] = (
        df["lang"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["language_confidence"] = pd.to_numeric(
        df["confidence"],
        errors="coerce",
    )

    return df


def aggregate_language_by_video(
    language_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate language predictions to one row per speaker-video pair."""
    return (
        language_df
        .groupby(["speaker_id", "video_id"], as_index=False)
        .agg(
            language=("language", "first"),
            language_confidence=("language_confidence", "first"),
        )
    )
