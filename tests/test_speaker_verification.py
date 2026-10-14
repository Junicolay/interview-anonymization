import numpy as np
import pytest

from src.speaker_verification import compute_eer, cosine_similarity


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 2], [1, 2]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_zero_vector():
    with pytest.raises(ValueError, match="non-zero norm"):
        cosine_similarity([0, 0], [1, 2])


def test_cosine_similarity_rejects_different_shapes():
    with pytest.raises(ValueError, match="same shape"):
        cosine_similarity([1, 2], [1, 2, 3])


def test_compute_eer_returns_expected_fields():
    result = compute_eer(
        labels=[1, 1, 0, 0],
        scores=[0.9, 0.8, 0.2, 0.1],
    )

    assert set(result) == {"eer", "threshold", "far", "frr"}
    assert result["eer"] == pytest.approx(0.0)
    assert result["far"] == pytest.approx(0.0)
    assert result["frr"] == pytest.approx(0.0)


def test_compute_eer_rejects_single_class():
    with pytest.raises(ValueError, match="Both classes"):
        compute_eer([1, 1], [0.9, 0.8])


def test_compute_eer_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_eer([1, 0], [0.9])


def test_compute_eer_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        compute_eer([1, 0], [0.9, np.nan])
