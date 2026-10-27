
"""Utilities for inspecting the VoxCeleb1 dataset."""

from pathlib import Path
import pandas as pd


def build_speaker_inventory(voxceleb_wav_dir: Path) -> pd.DataFrame:
    """Build an inventory of speakers, videos, and audio utterances.

    Parameters
    ----------
    voxceleb_wav_dir:
        Root directory containing one directory per speaker.

    Returns
    -------
    pd.DataFrame
        One row per speaker, with the number of videos and utterances.
    """
    if not voxceleb_wav_dir.is_dir():
        raise FileNotFoundError(
            f"VoxCeleb WAV directory not found: {voxceleb_wav_dir}"
        )

    rows = []

    for speaker_dir in voxceleb_wav_dir.iterdir():
        if not speaker_dir.is_dir():
            continue

        video_dirs = [
            path for path in speaker_dir.iterdir()
            if path.is_dir()
        ]
        wav_files = list(speaker_dir.rglob("*.wav"))

        rows.append({
            "speaker_id": speaker_dir.name,
            "n_videos": len(video_dirs),
            "n_utterances": len(wav_files),
        })

    return pd.DataFrame(
        rows,
        columns=["speaker_id", "n_videos", "n_utterances"],
    )


def build_video_inventory(voxceleb_wav_dir: Path) -> pd.DataFrame:
    """Build an inventory with one row per speaker-video pair.

    Counts WAV files directly inside each video directory.
    """
    if not voxceleb_wav_dir.is_dir():
        raise FileNotFoundError(
            f"VoxCeleb WAV directory not found: {voxceleb_wav_dir}"
        )

    rows = []

    for speaker_dir in voxceleb_wav_dir.iterdir():
        if not speaker_dir.is_dir():
            continue

        for video_dir in speaker_dir.iterdir():
            if not video_dir.is_dir():
                continue

            wav_files = list(video_dir.glob("*.wav"))

            rows.append({
                "speaker_id": speaker_dir.name,
                "video_id": video_dir.name,
                "n_utterances": len(wav_files),
            })

    return pd.DataFrame(
        rows,
        columns=["speaker_id", "video_id", "n_utterances"],
    )


def build_utterance_manifest(
    split_df: pd.DataFrame,
    voxceleb_wav_dir: Path,
) -> pd.DataFrame:
    """Build one manifest row per WAV utterance."""
    required_columns = {
        "speaker_id",
        "video_id",
        "split",
        "language",
        "language_confidence",
    }
    missing = required_columns - set(split_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []

    for row in split_df.itertuples(index=False):
        video_dir = voxceleb_wav_dir / row.speaker_id / row.video_id

        if not video_dir.is_dir():
            raise FileNotFoundError(
                f"Video directory not found: {video_dir}"
            )

        for wav_path in sorted(video_dir.glob("*.wav")):
            rows.append({
                "speaker_id": row.speaker_id,
                "video_id": row.video_id,
                "utterance_id": wav_path.stem,
                "path": wav_path.relative_to(voxceleb_wav_dir).as_posix(),
                "split": row.split,
                "language": row.language,
                "language_confidence": row.language_confidence,
            })

    return pd.DataFrame(
        rows,
        columns=[
            "speaker_id",
            "video_id",
            "utterance_id",
            "path",
            "split",
            "language",
            "language_confidence",
        ],
    )



def filter_eligible_english_videos(
    video_df: pd.DataFrame,
    min_videos_per_speaker: int = 3,
) -> pd.DataFrame:
    """Keep English videos from speakers with enough videos.

    Filtering is performed after language metadata has been merged
    into the video inventory.
    """
    required_columns = {"speaker_id", "language", "n_utterances"}
    missing = required_columns - set(video_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if min_videos_per_speaker < 1:
        raise ValueError("min_videos_per_speaker must be at least 1.")

    english_df = video_df.loc[
        video_df["language"] == "english"
    ].copy()

    videos_per_speaker = english_df.groupby("speaker_id").size()

    eligible_speakers = videos_per_speaker.loc[
        videos_per_speaker >= min_videos_per_speaker
    ].index

    return english_df.loc[
        english_df["speaker_id"].isin(eligible_speakers)
    ].copy()



def validate_utterance_manifest(
    utterance_df: pd.DataFrame,
    voxceleb_wav_dir: Path | None = None,
    check_paths: bool = True,
) -> None:
    """Validate the structure and integrity of an utterance manifest.

    Raises
    ------
    ValueError
        If required columns are missing, identifiers are duplicated,
        values are missing, or a video appears in multiple splits.
    FileNotFoundError
        If an audio file listed in the manifest does not exist.
    """
    required_columns = {
        "speaker_id",
        "video_id",
        "utterance_id",
        "path",
        "split",
        "language",
        "language_confidence",
    }

    missing_columns = required_columns - set(utterance_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if utterance_df.empty:
        raise ValueError("The utterance manifest is empty.")

    identifier_columns = [
        "speaker_id",
        "video_id",
        "utterance_id",
    ]

    if utterance_df[identifier_columns].isna().any().any():
        raise ValueError("Missing speaker, video, or utterance identifiers.")

    if utterance_df.duplicated(identifier_columns).any():
        raise ValueError("Duplicate utterances found in the manifest.")

    if utterance_df["path"].isna().any():
        raise ValueError("Missing audio paths found in the manifest.")

    if utterance_df["split"].isna().any():
        raise ValueError("Missing split assignments found in the manifest.")

    allowed_splits = {"train", "val", "test"}
    invalid_splits = set(utterance_df["split"].unique()) - allowed_splits

    if invalid_splits:
        raise ValueError(f"Invalid split names: {sorted(invalid_splits)}")

    # Each speaker-video pair must belong to exactly one split.
    splits_per_video = (
        utterance_df
        .groupby(["speaker_id", "video_id"])["split"]
        .nunique()
    )

    if (splits_per_video > 1).any():
        raise ValueError("A video appears in multiple splits.")

    if check_paths:
        if voxceleb_wav_dir is None:
            raise ValueError(
                "voxceleb_wav_dir must be provided when check_paths=True."
            )

        audio_paths = utterance_df["path"].map(
            lambda path: voxceleb_wav_dir / Path(path)
        )

        missing_paths = audio_paths.loc[
            ~audio_paths.map(Path.is_file)
        ]

        if not missing_paths.empty:
            examples = missing_paths.head(5).tolist()
            raise FileNotFoundError(
                f"{len(missing_paths)} audio files are missing. "
                f"Examples: {examples}"
            )
