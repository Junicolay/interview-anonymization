
"""Automatic speech recognition utilities."""

from pathlib import Path

import pandas as pd


def transcribe_audio(audio_path, model, language="en"):
    """Transcribe one audio file using a Faster-Whisper model."""
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    segments, _ = model.transcribe(
        str(audio_path),
        language=language,
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    )


def transcribe_pairs(
    pairs_df,
    model,
    language="en",
    original_column="original_path",
    anonymized_column="anonymized_path",
):
    """Transcribe both audio files for every original/anonymized pair."""
    required_columns = [original_column, anonymized_column]
    missing_columns = [
        column
        for column in required_columns
        if column not in pairs_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    rows = []

    for _, row in pairs_df.iterrows():
        result = row.to_dict()

        result["original_transcript"] = transcribe_audio(
            row[original_column],
            model=model,
            language=language,
        )

        result["anonymized_transcript"] = transcribe_audio(
            row[anonymized_column],
            model=model,
            language=language,
        )

        rows.append(result)

    return pd.DataFrame(rows)
