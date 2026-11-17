
"""Automatic speech recognition utilities."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Callable, Optional

import pandas as pd


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    """Return the Levenshtein edit distance between two word sequences."""
    if len(left) < len(right):
        left, right = right, left

    previous_row = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[right_index] + 1,
                    previous_row[right_index - 1] + (left_item != right_item),
                )
            )
        previous_row = current_row

    return previous_row[-1]


@dataclass(frozen=True)
class TimedASRSegment:
    """One native ASR segment and its timestamps in the source recording."""

    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class ASRTranscription:
    """Transcript text together with the timestamped ASR segments."""

    text: str
    segments: tuple[TimedASRSegment, ...]


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


def transcribe_audio_with_timestamps(
    audio_path,
    model: Any,
    language="en",
    progress_callback: Optional[Callable[[float], None]] = None,
) -> ASRTranscription:
    """Transcribe audio and preserve Faster-Whisper's native segment times."""
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    raw_segments, _ = model.transcribe(
        str(audio_path),
        language=language,
    )

    segments: list[TimedASRSegment] = []
    for index, segment in enumerate(raw_segments):
        text = segment.text.strip()
        if not text:
            continue

        try:
            start_seconds = float(segment.start)
            end_seconds = float(segment.end)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"ASR segment {index} has no usable start/end timestamps."
            ) from exc

        if (
            not math.isfinite(start_seconds)
            or not math.isfinite(end_seconds)
            or start_seconds < 0
            or end_seconds <= start_seconds
        ):
            raise ValueError(
                f"ASR segment {index} has invalid timestamps: "
                f"{start_seconds}, {end_seconds}."
            )

        segments.append(
            TimedASRSegment(
                text=text,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
        if progress_callback is not None:
            progress_callback(end_seconds)

    if not segments:
        raise ValueError(f"ASR returned no timestamped text for {audio_path}.")

    return ASRTranscription(
        text=" ".join(segment.text for segment in segments),
        segments=tuple(segments),
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute whitespace-independent word error rate after case/punctuation normalization."""
    reference_words = re.findall(
        r"\w+(?:['’]\w+)*",
        reference.lower(),
        flags=re.UNICODE,
    )
    hypothesis_words = re.findall(
        r"\w+(?:['’]\w+)*",
        hypothesis.lower(),
        flags=re.UNICODE,
    )
    if not reference_words:
        raise ValueError("Reference transcript must contain at least one word.")

    edit_distance = _levenshtein_distance(reference_words, hypothesis_words)
    return edit_distance / len(reference_words)


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
