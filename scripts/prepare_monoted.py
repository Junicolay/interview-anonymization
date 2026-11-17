"""Transcribe MonoTED talks and extract timestamped Faster-Whisper segments."""

import argparse
import json
import sys
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.base import Recording, Segment  # noqa: E402
from src.datasets.monoted import MonoTEDAdapter  # noqa: E402
from src.audio import (  # noqa: E402
    EmptyAudioSegmentError,
    extract_wav_segment,
    wav_duration_seconds,
)
from src.asr import transcribe_audio_with_timestamps, word_error_rate  # noqa: E402


def load_manifest_recording_ids(manifest_path: Path) -> set[str]:
    """Return the recording IDs already present in a JSONL manifest."""
    if not manifest_path.is_file():
        return set()

    processed_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {manifest_path}."
                ) from exc
            record_id = row.get("recording_id")
            if record_id:
                processed_ids.add(str(record_id))

    return processed_ids


def write_manifest(
    recordings: list[Recording],
    output_path: Path,
    *,
    append: bool = False,
) -> None:
    """Write one recording, its reference/ASR text, and ASR segments per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as output_file:
        for recording in recordings:
            json.dump(
                asdict(recording),
                output_file,
                ensure_ascii=False,
                default=_json_default,
            )
            output_file.write("\n")


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def prepare_recording(
    recording: Recording,
    model: Any,
    utterance_dir: Path,
    language: str,
    overwrite: bool = False,
    resume_incomplete: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Recording:
    """Transcribe one recording and extract each native ASR segment as audio."""
    transcription = transcribe_audio_with_timestamps(
        recording.audio_path,
        model=model,
        language=language,
        progress_callback=progress_callback,
    )

    segments = []
    for segment_index, asr_segment in enumerate(transcription.segments):
        segment_id = f"{recording.recording_id}:utt{segment_index:04d}"
        output_path = (
            utterance_dir
            / recording.recording_id
            / f"utt{segment_index:04d}.wav"
        )
        if output_path.exists() and not overwrite and not resume_incomplete:
            raise FileExistsError(
                f"Utterance audio already exists: {output_path}. "
                "Use --overwrite to regenerate cuts from the current ASR output."
            )
        try:
            extract_wav_segment(
                recording.audio_path,
                output_path,
                asr_segment.start_seconds,
                asr_segment.end_seconds,
            )
        except EmptyAudioSegmentError as exc:
            warnings.warn(
                f"Skipping ASR segment {segment_id!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        segments.append(
            Segment(
                segment_id=segment_id,
                asr_text=asr_segment.text,
                start_seconds=asr_segment.start_seconds,
                end_seconds=asr_segment.end_seconds,
                audio_path=output_path,
            )
        )

    return replace(
        recording,
        asr_transcript=transcription.text,
        asr_wer=word_error_rate(
            recording.reference_transcript,
            transcription.text,
        ),
        segments=tuple(segments),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe MonoTED audio, cut native timestamped ASR segments, "
            "and write a JSONL manifest."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/MonoTED",
        help="MonoTED dataset directory (default: data/raw/MonoTED)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/monoted",
        help="Output directory (default: data/processed/monoted)",
    )
    parser.add_argument(
        "--model-size",
        default="medium",
        help="Faster-Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Faster-Whisper device (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Faster-Whisper compute type (default: default)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Audio language code (default: en)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N recordings for a smoke test",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing utterance WAV files",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer.")

    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Faster-Whisper is required. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    recordings = MonoTEDAdapter(args.root).load_recordings()
    if args.limit is not None:
        recordings = recordings[: args.limit]
    output_dir = args.output_dir.expanduser().resolve()
    utterance_dir = output_dir / "utterance_audio"
    manifest_path = output_dir / "manifest.jsonl"
    processed_ids = load_manifest_recording_ids(manifest_path)
    if not args.overwrite:
        recordings = [
            recording
            for recording in recordings
            if recording.recording_id not in processed_ids
        ]

    if not recordings:
        print(
            f"All {len(processed_ids)} recordings are already present in the "
            f"manifest: {manifest_path}. Nothing to do."
        )
        return

    tqdm.write(f"Loading Faster-Whisper model '{args.model_size}'...")
    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    prepared_recordings = []
    total_segments = 0
    with tqdm(total=len(recordings), desc="Talks", unit="talk") as talk_progress:
        for recording in recordings:
            duration_seconds = wav_duration_seconds(recording.audio_path)
            with tqdm(
                total=duration_seconds,
                desc=f"ASR {recording.recording_id}",
                unit="s",
                leave=False,
            ) as audio_progress:
                def report_asr_progress(end_seconds: float) -> None:
                    processed_seconds = min(
                        max(audio_progress.n, end_seconds),
                        duration_seconds,
                    )
                    audio_progress.update(
                        processed_seconds - audio_progress.n
                    )

                prepared = prepare_recording(
                    recording,
                    model=model,
                    utterance_dir=utterance_dir,
                    language=args.language,
                    overwrite=args.overwrite,
                    resume_incomplete=not args.overwrite,
                    progress_callback=report_asr_progress,
                )
                audio_progress.update(
                    max(0, duration_seconds - audio_progress.n)
                )
            prepared_recordings.append(prepared)
            total_segments += len(prepared.segments)
            if not args.overwrite:
                write_manifest([prepared], manifest_path, append=True)
            talk_progress.update()

    if args.overwrite:
        write_manifest(prepared_recordings, manifest_path)
    print(
        f"Wrote {len(prepared_recordings)} recordings and {total_segments} "
        f"timestamped ASR segments. Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
