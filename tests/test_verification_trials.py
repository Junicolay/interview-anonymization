import pandas as pd
import pytest

from src.verification_trials import (
    build_verification_trials,
    select_eligible_speakers,
)


def sample_data():
    rows = []
    for speaker in ["s1", "s2", "s3"]:
        for video in range(3):
            rows.append(
                {
                    "speaker_id": speaker,
                    "video_id": f"{speaker}_v{video}",
                    "path": f"{speaker}/{video}.wav",
                }
            )
    return pd.DataFrame(rows)


def test_select_eligible_speakers():
    result = select_eligible_speakers(sample_data(), min_videos_per_speaker=3)
    assert result == ["s1", "s2", "s3"]


def test_trials_are_reproducible():
    data = sample_data()
    speakers = select_eligible_speakers(data, min_videos_per_speaker=3)

    first, _ = build_verification_trials(
        data, speakers, seed=42, n_speakers=3,
        positive_trials_per_speaker=2, negative_trials_per_speaker=2,
    )
    second, _ = build_verification_trials(
        data, speakers, seed=42, n_speakers=3,
        positive_trials_per_speaker=2, negative_trials_per_speaker=2,
    )

    pd.testing.assert_frame_equal(first, second)


def test_positive_and_negative_labels_are_consistent():
    data = sample_data()
    speakers = select_eligible_speakers(data, min_videos_per_speaker=3)
    trials, _ = build_verification_trials(
        data, speakers, seed=42, n_speakers=3,
        positive_trials_per_speaker=2, negative_trials_per_speaker=2,
    )

    assert (trials.loc[trials.label == 1, "speaker1"]
            == trials.loc[trials.label == 1, "speaker2"]).all()
    assert (trials.loc[trials.label == 0, "speaker1"]
            != trials.loc[trials.label == 0, "speaker2"]).all()


def test_too_few_speakers_raises():
    data = sample_data()
    speakers = select_eligible_speakers(data, min_videos_per_speaker=3)

    with pytest.raises(ValueError):
        build_verification_trials(data, speakers, n_speakers=10)
