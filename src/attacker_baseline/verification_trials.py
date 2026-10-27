"""Utilities for reproducible speaker-verification trials."""

import random
from pathlib import Path

import pandas as pd


def resolve_audio_path(path: str | Path, audio_root: str | Path) -> Path:
    """Resolve an audio path against the local VoxCeleb WAV directory."""
    path = Path(path).expanduser()
    audio_root = Path(audio_root).expanduser().resolve()

    if path.is_absolute() and path.is_file():
        return path.resolve()

    parts = path.parts
    if "wav" in parts:
        wav_index = parts.index("wav")
        candidate = audio_root.joinpath(*parts[wav_index + 1 :])
    else:
        candidate = audio_root / path

    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Audio file not found: {candidate}")

    return candidate


def select_eligible_speakers(
    utterances: pd.DataFrame,
    min_videos_per_speaker: int = 5,
) -> list:
    """Select speakers with at least the required number of distinct videos."""
    required = {"speaker_id", "video_id"}
    missing = required - set(utterances.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if min_videos_per_speaker < 1:
        raise ValueError("min_videos_per_speaker must be >= 1")

    # Keep the original notebook's pandas groupby ordering and index values.
    speaker_stats = (
        utterances.groupby("speaker_id")["video_id"]
        .nunique()
        .to_frame(name="n_videos")
    )
    return speaker_stats[
        speaker_stats["n_videos"] >= min_videos_per_speaker
    ].index.tolist()


def build_verification_trials(
    utterances: pd.DataFrame,
    eligible_speakers: list,
    seed: int = 42,
    n_speakers: int = 100,
    positive_trials_per_speaker: int = 10,
    negative_trials_per_speaker: int = 10,
) -> tuple[pd.DataFrame, list]:
    """Reproduce the original notebook's trial-generation procedure."""
    required = {"speaker_id", "video_id", "path"}
    missing = required - set(utterances.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if n_speakers < 2:
        raise ValueError("n_speakers must be >= 2")
    if positive_trials_per_speaker < 0 or negative_trials_per_speaker < 0:
        raise ValueError("Trial counts must be non-negative")
    if len(eligible_speakers) < n_speakers:
        raise ValueError(
            f"Need {n_speakers} eligible speakers; "
            f"found {len(eligible_speakers)}."
        )

    # IMPORTANT: preserve the same RNG and call order as the original notebook.
    rng = random.Random(seed)
    selected_speakers = rng.sample(eligible_speakers, n_speakers)

    positive_trials = []

    for speaker in selected_speakers:
        speaker_df = utterances[utterances["speaker_id"] == speaker]
        videos = speaker_df["video_id"].unique()

        if len(videos) < 2:
            raise ValueError(
                f"Speaker {speaker!r} has fewer than two distinct videos."
            )

        for _ in range(positive_trials_per_speaker):
            video1, video2 = rng.sample(list(videos), 2)

            utt1 = speaker_df[
                speaker_df["video_id"] == video1
            ].sample(
                1,
                random_state=rng.randint(0, 1_000_000),
            ).iloc[0]

            utt2 = speaker_df[
                speaker_df["video_id"] == video2
            ].sample(
                1,
                random_state=rng.randint(0, 1_000_000),
            ).iloc[0]

            positive_trials.append(
                {
                    "label": 1,
                    "path1": utt1["path"],
                    "path2": utt2["path"],
                    "speaker1": speaker,
                    "speaker2": speaker,
                }
            )

    negative_trials = []

    for _ in range(n_speakers * negative_trials_per_speaker):
        speaker1, speaker2 = rng.sample(selected_speakers, 2)

        utt1 = utterances[
            utterances["speaker_id"] == speaker1
        ].sample(
            1,
            random_state=rng.randint(0, 1_000_000),
        ).iloc[0]

        utt2 = utterances[
            utterances["speaker_id"] == speaker2
        ].sample(
            1,
            random_state=rng.randint(0, 1_000_000),
        ).iloc[0]

        negative_trials.append(
            {
                "label": 0,
                "path1": utt1["path"],
                "path2": utt2["path"],
                "speaker1": speaker1,
                "speaker2": speaker2,
            }
        )

    # Preserve the original concatenation and final shuffle.
    verification_trials = pd.DataFrame(
        positive_trials + negative_trials
    )
    verification_trials = verification_trials.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)

    return verification_trials, selected_speakers
