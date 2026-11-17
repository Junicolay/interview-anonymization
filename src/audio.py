"""Audio utilities for extracting timestamped WAV segments."""

import math
import wave
from pathlib import Path
from typing import Union


class EmptyAudioSegmentError(ValueError):
    """Raised when a valid timestamp interval contains no source audio frames."""


def wav_duration_seconds(audio_path: Union[str, Path]) -> float:
    """Return a WAV file's duration in seconds."""
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with wave.open(str(audio_path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def extract_wav_segment(
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    start_seconds: float,
    end_seconds: float,
) -> Path:
    """Write a variable-duration WAV cut using source-relative timestamps."""
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {source_path}")
    if source_path == output_path:
        raise ValueError("Output path must differ from the source audio path.")
    if (
        not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
        or start_seconds < 0
        or end_seconds <= start_seconds
    ):
        raise ValueError(
            f"Invalid segment times: {start_seconds}, {end_seconds}."
        )

    with wave.open(str(source_path), "rb") as source:
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        duration_seconds = total_frames / frame_rate

        # Some ASR timestamp boundaries can overshoot the final audio sample by
        # a small amount; clip the cut to the valid source range instead of
        # aborting the whole recording.
        if end_seconds > duration_seconds:
            end_seconds = duration_seconds
        if start_seconds > duration_seconds:
            start_seconds = duration_seconds

        if end_seconds <= start_seconds:
            raise EmptyAudioSegmentError(
                f"Segment [{start_seconds}, {end_seconds}] is outside "
                f"the {duration_seconds:.3f}s audio file."
            )

        start_frame = int(math.floor(start_seconds * frame_rate))
        end_frame = min(
            int(math.ceil(end_seconds * frame_rate)),
            total_frames,
        )

        if start_frame >= total_frames or end_frame <= start_frame:
            raise EmptyAudioSegmentError(
                f"Segment [{start_seconds}, {end_seconds}] is outside "
                f"the {total_frames / frame_rate:.3f}s audio file."
            )

        source.setpos(start_frame)
        audio_frames = source.readframes(end_frame - start_frame)
        params = source.getparams()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output:
        output.setparams(
            (
                params.nchannels,
                params.sampwidth,
                params.framerate,
                end_frame - start_frame,
                params.comptype,
                params.compname,
            )
        )
        output.writeframes(audio_frames)

    return output_path
