"""Run the ECAPA speaker-verification experiment from the command line."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding_cache import SQLiteEmbeddingCache
from src.speaker_verification import compute_eer, cosine_similarity
from src.verification_trials import (
    build_verification_trials,
    resolve_audio_path,
    select_eligible_speakers,
)

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb:extract-v1"


def git_commit() -> str | None:
    """Return the current Git commit, if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256_file(path: Path) -> str:
    """Compute a file's SHA-256 hash."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ECAPA speaker-verification experiment."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data/metadata/voxceleb_utterances.parquet",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=PROJECT_ROOT / "voxceleb1/wav",
    )
    parser.add_argument(
        "--cache-db",
        type=Path,
        default=PROJECT_ROOT / "data/cache/ecapa_embeddings.sqlite",
    )
    parser.add_argument(
        "--trials-output",
        type=Path,
        default=PROJECT_ROOT
        / "data/trials/voxceleb_verification_trials_legacy_reproduction.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/speaker_verification",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-speakers", type=int, default=100)
    parser.add_argument("--positive-per-speaker", type=int, default=10)
    parser.add_argument("--negative-per-speaker", type=int, default=10)
    parser.add_argument("--min-videos-per-speaker", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    metadata_path = args.metadata.resolve()
    audio_root = args.audio_root.resolve()
    cache_path = args.cache_db.resolve()
    trials_path = args.trials_output.resolve()
    output_dir = args.output_dir.resolve()

    for path, description in [
        (metadata_path, "Metadata file"),
        (audio_root, "Audio directory"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    if args.n_speakers < 2:
        raise ValueError("--n-speakers must be at least 2.")
    if args.min_videos_per_speaker < 1:
        raise ValueError("--min-videos-per-speaker must be positive.")
    if args.positive_per_speaker < 0 or args.negative_per_speaker < 0:
        raise ValueError("Trial counts must be non-negative.")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Metadata: {metadata_path}")
    print(f"Audio root: {audio_root}")

    # 1. Load and validate metadata.
    utterances = pd.read_parquet(metadata_path)
    required = {"speaker_id", "video_id", "path"}
    missing = required - set(utterances.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    utterances = utterances.copy()
    tqdm.pandas(desc="Resolving audio paths")
    utterances["path"] = utterances["path"].progress_apply(
        lambda path: str(resolve_audio_path(path, audio_root))
    )

    print(f"Utterances: {len(utterances)}")
    print(f"Speakers: {utterances['speaker_id'].nunique()}")

    # 2. Generate reproducible trials.
    eligible = select_eligible_speakers(
        utterances,
        min_videos_per_speaker=args.min_videos_per_speaker,
    )
    trials, selected_speakers = build_verification_trials(
        utterances,
        eligible,
        seed=args.seed,
        n_speakers=args.n_speakers,
        positive_trials_per_speaker=args.positive_per_speaker,
        negative_trials_per_speaker=args.negative_per_speaker,
    )

    trials_path.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(trials_path, index=False)
    trials_hash = sha256_file(trials_path)

    print(f"Eligible speakers: {len(eligible)}")
    print(f"Selected speakers: {len(selected_speakers)}")
    print(f"Trials: {len(trials)}")
    print(f"Trials saved to: {trials_path}")

    # 3. Reuse cached embeddings; load ECAPA only if a cache miss occurs.
    cache = SQLiteEmbeddingCache(cache_path, model_id=MODEL_ID)
    unique_paths = sorted(set(trials["path1"]) | set(trials["path2"]))

    embeddings = {}
    cache_hits = 0
    cache_misses = 0
    classifier = None

    for audio_path in tqdm(unique_paths, desc="Loading embeddings"):
        embedding = cache.get(audio_path)

        if embedding is None:
            cache_misses += 1

            if classifier is None:
                from src.ecapa import extract_embedding, load_ecapa

                print("Cache miss detected: loading ECAPA model...")
                classifier = load_ecapa()

            embedding = extract_embedding(audio_path, classifier)
            embedding = np.asarray(embedding)
            cache.set(audio_path, embedding)
        else:
            cache_hits += 1
            embedding = np.asarray(embedding)

        embeddings[audio_path] = embedding

    print(f"Embeddings required: {len(unique_paths)}")
    print(f"Loaded from SQLite: {cache_hits}")
    print(f"Computed and cached: {cache_misses}")

    # 4. Score all trials and compute metrics.
    scores = []
    for row in tqdm(
        trials.itertuples(index=False),
        total=len(trials),
        desc="Scoring trials",
    ):
        scores.append(
            cosine_similarity(
                embeddings[row.path1],
                embeddings[row.path2],
            )
        )

    scored_trials = trials.copy()
    scored_trials["score"] = scores

    metrics = compute_eer(
        labels=scored_trials["label"].to_numpy(),
        scores=scored_trials["score"].to_numpy(),
    )

    # 5. Save scores and a machine-readable experiment report.
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scored_trials.csv"
    metrics_path = output_dir / "metrics.json"

    scored_trials.to_csv(scores_path, index=False)

    report = {
        "experiment": "ecapa_speaker_verification",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "git_commit": git_commit(),
        "config": {
            "seed": args.seed,
            "n_speakers": args.n_speakers,
            "positive_trials_per_speaker": args.positive_per_speaker,
            "negative_trials_per_speaker": args.negative_per_speaker,
            "min_videos_per_speaker": args.min_videos_per_speaker,
        },
        "inputs": {
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "audio_root": str(audio_root),
            "trials_file": str(trials_path),
            "trials_sha256": trials_hash,
        },
        "counts": {
            "utterances": len(utterances),
            "eligible_speakers": len(eligible),
            "selected_speakers": len(selected_speakers),
            "trials": len(scored_trials),
            "unique_audio_files": len(unique_paths),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        },
        "metrics": metrics,
        "outputs": {
            "scored_trials": str(scores_path),
            "metrics": str(metrics_path),
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }

    metrics_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== Speaker verification results ===")
    print(f"EER:       {metrics['eer']:.4%}")
    print(f"Threshold: {metrics['threshold']:.6f}")
    print(f"FAR:       {metrics['far']:.4%}")
    print(f"FRR:       {metrics['frr']:.4%}")
    print(f"\nScored trials: {scores_path}")
    print(f"Metrics report: {metrics_path}")
    print(f"Runtime: {report['runtime_seconds']} seconds")


if __name__ == "__main__":
    main()
