"""Shared data types for dataset-independent speech pipelines."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


@dataclass(frozen=True)
class Segment:
    """A transcript segment with optional source timing and extracted audio."""

    segment_id: str
    asr_text: str
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    audio_path: Optional[Path] = None
    reference_text: Optional[str] = None


@dataclass(frozen=True)
class Recording:
    """One recording and its dataset-independent speaker and text information."""

    dataset_id: str
    recording_id: str
    speaker_id: str
    audio_path: Path
    transcript_path: Path
    reference_transcript: str
    segments: tuple[Segment, ...]
    metadata: dict[str, Optional[str]] = field(default_factory=dict)
    asr_transcript: Optional[str] = None
    asr_wer: Optional[float] = None


class DatasetAdapter(Protocol):
    """Interface implemented by each corpus-specific adapter."""

    dataset_id: str

    def load_recordings(self) -> list[Recording]:
        """Load and validate recordings in the shared representation."""
        ...
