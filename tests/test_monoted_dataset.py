import csv
import wave

import pytest

from src.datasets.monoted import MonoTEDAdapter


def write_test_wav(path):
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8)
        audio_file.writeframes(b"\x01\x00" * 8)


def make_dataset(tmp_path):
    root = tmp_path / "MonoTED"
    (root / "audio_wav").mkdir(parents=True)
    (root / "transcripts").mkdir()
    with (root / "names_and_labels.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["Name", "File", "Gender", "Language"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Name": "Speaker One",
                "File": "Talk_001",
                "Gender": "F",
                "Language": "US",
            }
        )
    write_test_wav(root / "audio_wav" / "Talk_001.wav")
    (root / "transcripts" / "Talk_001.txt").write_text(
        "Hello there. A short reply! Final sentence",
        encoding="utf-8",
    )
    return root


def test_monoted_adapter_loads_reference_without_segmenting_it(tmp_path):
    adapter = MonoTEDAdapter(make_dataset(tmp_path))

    recordings = adapter.load_recordings()

    assert len(recordings) == 1
    recording = recordings[0]
    assert recording.dataset_id == "monoted"
    assert recording.recording_id == "Talk_001"
    assert recording.speaker_id == "Speaker One"
    assert recording.reference_transcript == (
        "Hello there. A short reply! Final sentence"
    )
    assert recording.asr_transcript is None
    assert recording.segments == ()
    assert recording.metadata == {"Gender": "F", "Language": "US"}


def test_monoted_adapter_rejects_missing_audio(tmp_path):
    root = make_dataset(tmp_path)
    (root / "audio_wav" / "Talk_001.wav").unlink()

    with pytest.raises(FileNotFoundError, match="Audio for recording"):
        MonoTEDAdapter(root).load_recordings()


def test_monoted_adapter_rejects_duplicate_recording_ids(tmp_path):
    root = make_dataset(tmp_path)
    with (root / "names_and_labels.csv").open(
        "a", encoding="utf-8", newline=""
    ) as csv_file:
        csv_file.write("Speaker Two,Talk_001,M,US\n")

    with pytest.raises(ValueError, match="Duplicate MonoTED recording ID"):
        MonoTEDAdapter(root).load_recordings()


def test_monoted_adapter_rejects_recording_id_path_traversal(tmp_path):
    root = make_dataset(tmp_path)
    metadata_path = root / "names_and_labels.csv"
    metadata_path.write_text(
        "Name,File,Gender,Language\nSpeaker One,../outside,F,US\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid MonoTED recording ID"):
        MonoTEDAdapter(root).load_recordings()
