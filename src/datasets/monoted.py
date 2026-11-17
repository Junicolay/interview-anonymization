"""Adapter for the local MonoTED talks dataset."""

import csv
from pathlib import Path
from typing import Union

from src.datasets.base import Recording


class MonoTEDAdapter:
    """Load MonoTED audio, reference transcripts, and talk-level metadata."""

    dataset_id = "monoted"
    required_columns = {"Name", "File"}

    def __init__(
        self,
        root: Union[str, Path],
        *,
        audio_dir: str = "audio_wav",
        transcript_dir: str = "transcripts",
        metadata_file: str = "names_and_labels.csv",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.audio_root = self.root / audio_dir
        self.transcript_root = self.root / transcript_dir
        self.metadata_path = self.root / metadata_file

    def load_recordings(self) -> list[Recording]:
        """Read every metadata row and require its audio and transcript files."""
        for path in (self.audio_root, self.transcript_root):
            if not path.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"MonoTED metadata file not found: {self.metadata_path}"
            )

        with self.metadata_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as csv_file:
            reader = csv.DictReader(csv_file)
            columns = set(reader.fieldnames or ())
            missing_columns = self.required_columns - columns
            if missing_columns:
                raise ValueError(
                    "MonoTED metadata is missing required columns: "
                    f"{sorted(missing_columns)}"
                )
            rows = list(reader)

        recordings: list[Recording] = []
        seen_recording_ids: set[str] = set()

        for row_number, row in enumerate(rows, start=2):
            recording_id = (row.get("File") or "").strip()
            speaker_id = (row.get("Name") or "").strip()
            if not recording_id or not speaker_id:
                raise ValueError(
                    f"Missing File or Name in {self.metadata_path}, "
                    f"row {row_number}"
                )
            if (
                recording_id in {".", ".."}
                or "/" in recording_id
                or "\\" in recording_id
            ):
                raise ValueError(
                    f"Invalid MonoTED recording ID {recording_id!r} "
                    f"in {self.metadata_path}, row {row_number}"
                )
            if recording_id in seen_recording_ids:
                raise ValueError(
                    f"Duplicate MonoTED recording ID {recording_id!r} "
                    f"in {self.metadata_path}"
                )
            seen_recording_ids.add(recording_id)

            audio_path = self.audio_root / f"{recording_id}.wav"
            transcript_path = self.transcript_root / f"{recording_id}.txt"
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio for recording {recording_id!r} not found: "
                    f"{audio_path}"
                )
            if not transcript_path.is_file():
                raise FileNotFoundError(
                    f"Transcript for recording {recording_id!r} not found: "
                    f"{transcript_path}"
                )

            transcript = transcript_path.read_text(encoding="utf-8").strip()
            if not transcript:
                raise ValueError(
                    f"Transcript is empty for recording {recording_id!r}: "
                    f"{transcript_path}"
                )

            metadata = {
                key: value.strip() or None if value is not None else None
                for key, value in row.items()
                if key not in self.required_columns and key is not None
            }
            recordings.append(
                Recording(
                    dataset_id=self.dataset_id,
                    recording_id=recording_id,
                    speaker_id=speaker_id,
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    reference_transcript=transcript,
                    segments=(),
                    metadata=metadata,
                )
            )

        return recordings
