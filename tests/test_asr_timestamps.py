from types import SimpleNamespace

import pytest

from src.asr import transcribe_audio_with_timestamps, word_error_rate


class FakeWhisperModel:
    def __init__(self, segments):
        self.segments = segments
        self.arguments = None

    def transcribe(self, audio_path, **kwargs):
        self.arguments = (audio_path, kwargs)
        return iter(self.segments), SimpleNamespace(language="en")


def test_transcribe_audio_preserves_native_segment_times(tmp_path):
    audio_path = tmp_path / "talk.wav"
    audio_path.write_bytes(b"fake audio")
    model = FakeWhisperModel(
        [
            SimpleNamespace(start=0.25, end=1.5, text=" First sentence. "),
            SimpleNamespace(start=1.5, end=2.75, text="Second sentence."),
        ]
    )

    result = transcribe_audio_with_timestamps(audio_path, model)

    assert result.text == "First sentence. Second sentence."
    assert [(segment.start_seconds, segment.end_seconds) for segment in result.segments] == [
        (0.25, 1.5),
        (1.5, 2.75),
    ]
    assert [segment.text for segment in result.segments] == [
        "First sentence.",
        "Second sentence.",
    ]
    assert model.arguments == (str(audio_path), {"language": "en"})


def test_transcribe_audio_reports_segment_end_times(tmp_path):
    audio_path = tmp_path / "talk.wav"
    audio_path.write_bytes(b"fake audio")
    model = FakeWhisperModel(
        [
            SimpleNamespace(start=0.25, end=1.5, text="First."),
            SimpleNamespace(start=1.5, end=2.75, text="Second."),
        ]
    )
    progress = []

    transcribe_audio_with_timestamps(
        audio_path,
        model,
        progress_callback=progress.append,
    )

    assert progress == [1.5, 2.75]


def test_transcribe_audio_rejects_invalid_timestamps(tmp_path):
    audio_path = tmp_path / "talk.wav"
    audio_path.write_bytes(b"fake audio")
    model = FakeWhisperModel(
        [SimpleNamespace(start=2.0, end=1.0, text="invalid interval")]
    )

    with pytest.raises(ValueError, match="invalid timestamps"):
        transcribe_audio_with_timestamps(audio_path, model)


def test_transcribe_audio_rejects_empty_result(tmp_path):
    audio_path = tmp_path / "talk.wav"
    audio_path.write_bytes(b"fake audio")

    with pytest.raises(ValueError, match="no timestamped text"):
        transcribe_audio_with_timestamps(audio_path, FakeWhisperModel([]))


def test_word_error_rate_normalizes_case_and_punctuation():
    assert word_error_rate("Hello, there!", "hello there.") == 0
    assert word_error_rate("one two three", "one too three") == pytest.approx(1 / 3)


def test_word_error_rate_rejects_empty_reference():
    with pytest.raises(ValueError, match="at least one word"):
        word_error_rate(" , ! ", "some words")
