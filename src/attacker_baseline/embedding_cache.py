"""Persistent SQLite cache for audio embeddings."""

import hashlib
import sqlite3
from pathlib import Path

import numpy as np


class SQLiteEmbeddingCache:
    """Store embeddings on disk and invalidate entries when inputs change."""

    def __init__(self, db_path: str | Path, model_id: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    dtype TEXT NOT NULL,
                    shape TEXT NOT NULL
                )
                """
            )

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _metadata(self, audio_path: str | Path):
        path = Path(audio_path).resolve()
        stat = path.stat()
        identity = (
            f"{path}|{self.model_id}|{stat.st_size}|{stat.st_mtime_ns}"
        )
        key = hashlib.sha256(identity.encode()).hexdigest()
        return key, path, stat

    def get(self, audio_path: str | Path):
        key, _, _ = self._metadata(audio_path)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT embedding, dtype, shape
                FROM embeddings
                WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()

        if row is None:
            return None

        data, dtype, shape = row
        dimensions = tuple(int(value) for value in shape.split(",") if value)
        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(dimensions).copy()

    def set(self, audio_path: str | Path, embedding):
        key, path, stat = self._metadata(audio_path)
        array = np.asarray(embedding)
        if array.dtype == object:
            raise ValueError("Object-dtype embeddings are not supported.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (cache_key, path, model_id, file_size, mtime_ns,
                 embedding, dtype, shape)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    str(path),
                    self.model_id,
                    stat.st_size,
                    stat.st_mtime_ns,
                    array.tobytes(),
                    array.dtype.str,
                    ",".join(map(str, array.shape)),
                ),
            )

    def __len__(self):
        with self._connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_id = ?",
                (self.model_id,),
            ).fetchone()[0]
