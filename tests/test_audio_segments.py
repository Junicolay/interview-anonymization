import wave

import pytest

from src.audio import (
    EmptyAudioSegmentError,
    extract_wav_segment,
    wav_duration_seconds,
)


def write_test_wav(path, frames=16, sample_rate=8):
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(
            b"".join(index.to_bytes(2, "little") for index in range(frames))
        )


def test_extract_wav_segment_preserves_source_frames_and_variable_duration(tmp_path):
    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "cuts" / "segment.wav"
    write_test_wav(source_path)

    result_path = extract_wav_segment(source_path, output_path, 0.5, 1.25)

    with wave.open(str(result_path), "rb") as segment:
        assert segment.getframerate() == 8
        assert segment.getnchannels() == 1
        assert segment.getnframes() == 6
        assert segment.readframes(6) == b"".join(
            index.to_bytes(2, "little") for index in range(4, 10)
        )


def test_wav_duration_seconds_reads_frame_count_and_sample_rate(tmp_path):
    source_path = tmp_path / "source.wav"
    write_test_wav(source_path, frames=24, sample_rate=8)

    assert wav_duration_seconds(source_path) == 3.0


def test_extract_wav_segment_clips_to_source_duration_when_asr_overshoots(tmp_path):
    source_path = tmp_path / "source.wav"
    write_test_wav(source_path, frames=16, sample_rate=8)

    result_path = extract_wav_segment(source_path, tmp_path / "segment.wav", 1.0, 3.0)

    with wave.open(str(result_path), "rb") as segment:
        assert segment.getnframes() == 8
        assert segment.readframes(8) == b"".join(
            index.to_bytes(2, "little") for index in range(8, 16)
        )


def test_extract_wav_segment_keeps_audio_for_start_just_before_eof(tmp_path):
    source_path = tmp_path / "source.wav"
    write_test_wav(source_path, frames=16, sample_rate=8)

    result_path = extract_wav_segment(
        source_path,
        tmp_path / "segment.wav",
        1.99,
        2.1,
    )

    with wave.open(str(result_path), "rb") as segment:
        assert segment.getnframes() == 1
        assert segment.readframes(1) == (15).to_bytes(2, "little")


def test_extract_wav_segment_rejects_interval_starting_at_eof(tmp_path):
    source_path = tmp_path / "source.wav"
    write_test_wav(source_path, frames=16, sample_rate=8)

    with pytest.raises(EmptyAudioSegmentError, match="outside"):
        extract_wav_segment(source_path, tmp_path / "segment.wav", 2.0, 2.1)


def test_extract_wav_segment_rejects_empty_or_reversed_intervals(tmp_path):
    source_path = tmp_path / "source.wav"
    write_test_wav(source_path)

    with pytest.raises(ValueError, match="Invalid segment times"):
        extract_wav_segment(source_path, tmp_path / "segment.wav", 1.0, 1.0)
