import csv
import json
import wave
from types import SimpleNamespace

import pytest

from scripts.prepare_monoted import (
    load_manifest_recording_ids,
    prepare_recording,
    write_manifest,
)
from src.datasets.monoted import MonoTEDAdapter


def test_write_manifest_appends_without_overwriting_existing_rows(tmp_path):
    dataset_root = make_dataset(tmp_path)
    recording = MonoTEDAdapter(dataset_root).load_recordings()[0]
    first = prepare_recording(
        recording,
        FakeWhisperModel(),
        tmp_path / "utterances_a",
        "en",
    )
    second = prepare_recording(
        recording,
        FakeWhisperModel(),
        tmp_path / "utterances_b",
        "en",
    )
    output_path = tmp_path / "manifest.jsonl"

    write_manifest([first], output_path)
    write_manifest([second], output_path, append=True)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["recording_id"] for row in rows] == [
        first.recording_id,
        second.recording_id,
    ]
    assert load_manifest_recording_ids(output_path) == {first.recording_id}


class FakeWhisperModel:
    def transcribe(self, audio_path, **kwargs):
        return iter(
            [
                SimpleNamespace(start=0.25, end=0.75, text="ASR first."),
                SimpleNamespace(start=0.75, end=1.5, text="ASR second."),
            ]
        ), SimpleNamespace(language="en")


def make_dataset(tmp_path):
    root = tmp_path / "MonoTED"
    (root / "audio_wav").mkdir(parents=True)
    (root / "transcripts").mkdir()
    with (root / "names_and_labels.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["Name", "File"])
        writer.writeheader()
        writer.writerow({"Name": "Speaker One", "File": "Talk_001"})
    with wave.open(str(root / "audio_wav" / "Talk_001.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8)
        audio.writeframes(b"\x01\x00" * 16)
    (root / "transcripts" / "Talk_001.txt").write_text(
        "Reference text differs from ASR.",
        encoding="utf-8",
    )
    return root


def test_prepare_recording_cuts_whisper_segments_and_keeps_reference_separate(
    tmp_path,
):
    recording = MonoTEDAdapter(make_dataset(tmp_path)).load_recordings()[0]

    prepared = prepare_recording(
        recording,
        FakeWhisperModel(),
        tmp_path / "utterances",
        language="en",
    )

    assert prepared.reference_transcript == "Reference text differs from ASR."
    assert prepared.asr_transcript == "ASR first. ASR second."
    assert prepared.asr_wer is not None and prepared.asr_wer > 0
    assert [segment.asr_text for segment in prepared.segments] == [
        "ASR first.",
        "ASR second.",
    ]
    assert [
        (segment.start_seconds, segment.end_seconds)
        for segment in prepared.segments
    ] == [(0.25, 0.75), (0.75, 1.5)]
    assert all(segment.reference_text is None for segment in prepared.segments)
    assert all(segment.audio_path.is_file() for segment in prepared.segments)


def test_manifest_serializes_reference_asr_and_segment_audio_paths(tmp_path):
    recording = MonoTEDAdapter(make_dataset(tmp_path)).load_recordings()[0]
    prepared = prepare_recording(
        recording,
        FakeWhisperModel(),
        tmp_path / "utterances",
        language="en",
    )
    output_path = tmp_path / "manifest.jsonl"

    write_manifest([prepared], output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["reference_transcript"] == "Reference text differs from ASR."
    assert row["asr_transcript"] == "ASR first. ASR second."
    assert row["segments"][0]["asr_text"] == "ASR first."
    assert row["segments"][0]["audio_path"].endswith("utt0000.wav")


def test_manifest_serializes_paths_inside_project_relative_to_project_root(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.prepare_monoted.PROJECT_ROOT", tmp_path)
    recording = MonoTEDAdapter(make_dataset(tmp_path)).load_recordings()[0]
    prepared = prepare_recording(
        recording,
        FakeWhisperModel(),
        tmp_path / "utterances",
        language="en",
    )
    output_path = tmp_path / "manifest.jsonl"

    write_manifest([prepared], output_path)

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["audio_path"] == "MonoTED/audio_wav/Talk_001.wav"
    assert row["transcript_path"] == "MonoTED/transcripts/Talk_001.txt"
    assert row["segments"][0]["audio_path"] == "utterances/Talk_001/utt0000.wav"


def test_prepare_recording_requires_overwrite_for_existing_cuts(tmp_path):
    recording = MonoTEDAdapter(make_dataset(tmp_path)).load_recordings()[0]
    utterance_dir = tmp_path / "utterances"
    prepare_recording(recording, FakeWhisperModel(), utterance_dir, "en")

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        prepare_recording(recording, FakeWhisperModel(), utterance_dir, "en")

    overwritten = prepare_recording(
        recording,
        FakeWhisperModel(),
        utterance_dir,
        "en",
        overwrite=True,
    )
    assert len(overwritten.segments) == 2


def test_prepare_recording_can_replace_cuts_from_an_incomplete_run(tmp_path):
    recording = MonoTEDAdapter(make_dataset(tmp_path)).load_recordings()[0]
    utterance_dir = tmp_path / "utterances"
    first = prepare_recording(recording, FakeWhisperModel(), utterance_dir, "en")

    resumed = prepare_recording(
        recording,
        FakeWhisperModel(),
        utterance_dir,
        "en",
        resume_incomplete=True,
    )

    assert [segment.audio_path for segment in resumed.segments] == [
        segment.audio_path for segment in first.segments
    ]


def test_prepare_recording_skips_segment_with_no_audio_at_eof(tmp_path):
    recording = MonoTEDAdapter(make_dataset(tmp_path)).load_recordings()[0]

    class ModelWithEofSegment:
        def transcribe(self, audio_path, **kwargs):
            return iter(
                [
                    SimpleNamespace(start=0.25, end=0.75, text="Audible."),
                    SimpleNamespace(start=2.0, end=2.1, text="No audio."),
                ]
            ), SimpleNamespace(language="en")

    with pytest.warns(RuntimeWarning, match="Skipping ASR segment"):
        prepared = prepare_recording(
            recording,
            ModelWithEofSegment(),
            tmp_path / "utterances",
            "en",
        )

    assert prepared.asr_transcript == "Audible. No audio."
    assert [segment.asr_text for segment in prepared.segments] == ["Audible."]
