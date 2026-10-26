"""Utilities for evaluating speaker verification systems."""

import numpy as np
from sklearn.metrics import roc_curve


def cosine_similarity(z1, z2) -> float:
    """Compute cosine similarity between two embedding vectors."""
    z1 = np.asarray(z1, dtype=float).reshape(-1)
    z2 = np.asarray(z2, dtype=float).reshape(-1)

    if z1.shape != z2.shape:
        raise ValueError("Embeddings must have the same shape.")

    if not np.all(np.isfinite(z1)) or not np.all(np.isfinite(z2)):
        raise ValueError("Embeddings must contain only finite values.")

    norm1 = np.linalg.norm(z1)
    norm2 = np.linalg.norm(z2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Embeddings must have a non-zero norm.")

    z1 = z1 / norm1
    z2 = z2 / norm2

    return float(np.dot(z1, z2))


def compute_eer(labels, scores) -> dict:
    """Compute EER and its associated threshold from verification scores."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional.")

    if len(labels) != len(scores):
        raise ValueError("Labels and scores must have the same length.")

    if len(labels) == 0:
        raise ValueError("Labels and scores must not be empty.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must contain only finite values.")

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be binary: 0 or 1.")

    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes (0 and 1) must be present.")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))

    return {
        "eer": float((fpr[eer_idx] + fnr[eer_idx]) / 2),
        "threshold": float(thresholds[eer_idx]),
        "far": float(fpr[eer_idx]),
        "frr": float(fnr[eer_idx]),
    }
