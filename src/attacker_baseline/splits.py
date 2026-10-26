
"""Utilities for creating reproducible dataset splits."""

import numpy as np
import pandas as pd


def create_speaker_video_splits(
    video_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Split each speaker's videos into train, validation, and test sets.

    Requires at least three videos per speaker.
    Target proportions are 70% train, 15% validation, and the remainder test.
    Each speaker receives at least one video in each split.
    """
    required_columns = {"speaker_id", "video_id"}
    missing = required_columns - set(video_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if video_df.duplicated(["speaker_id", "video_id"]).any():
        raise ValueError("video_df contains duplicate speaker-video pairs.")

    videos_per_speaker = video_df.groupby("speaker_id").size()
    if (videos_per_speaker < 3).any():
        raise ValueError(
            "Every speaker must have at least three videos to create "
            "non-empty train, validation, and test splits."
        )

    rng = np.random.default_rng(seed)
    split_rows = []

    for speaker_id, group in video_df.groupby("speaker_id"):
        videos = group["video_id"].tolist()
        rng.shuffle(videos)

        n = len(videos)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        n_test = n - n_train - n_val

        if n_val < 1:
            n_val = 1
        if n_test < 1:
            n_test = 1

        n_train = n - n_val - n_test

        assignments = (
            [("train", videos[:n_train])]
            + [("val", videos[n_train:n_train + n_val])]
            + [("test", videos[n_train + n_val:])]
        )

        for split_name, split_videos in assignments:
            for video_id in split_videos:
                split_rows.append({
                    "speaker_id": speaker_id,
                    "video_id": video_id,
                    "split": split_name,
                })

    return pd.DataFrame(
        split_rows,
        columns=["speaker_id", "video_id", "split"],
    )
