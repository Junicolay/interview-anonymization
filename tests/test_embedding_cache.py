import numpy as np

from src.embedding_cache import SQLiteEmbeddingCache


def test_cache_round_trip(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake audio")
    cache = SQLiteEmbeddingCache(tmp_path / "cache.sqlite", "test-model")

    expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert cache.get(audio) is None

    cache.set(audio, expected)

    np.testing.assert_array_equal(cache.get(audio), expected)
    assert len(cache) == 1


def test_cache_invalidates_when_file_changes(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio-v1")
    cache = SQLiteEmbeddingCache(tmp_path / "cache.sqlite", "test-model")
    cache.set(audio, np.array([1.0], dtype=np.float32))

    audio.write_bytes(b"audio-version-two")

    assert cache.get(audio) is None
