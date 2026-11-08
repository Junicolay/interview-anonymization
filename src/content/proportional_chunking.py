import math
import numpy as np
import pandas as pd


def build_proportional_chunks(
    original_units,
    anonymized_units,
    similarity_model,
    original_chunk_size=8,
):
    """
    Build proportional chunks and calculate semantic similarity.
    """

    if not original_units or not anonymized_units:
        raise ValueError(
            "Both original and anonymized documents must be non-empty."
        )

    if not all(isinstance(unit, str) for unit in original_units):
        raise TypeError("All original units must be strings.")

    if not all(isinstance(unit, str) for unit in anonymized_units):
        raise TypeError("All anonymized units must be strings.")

    if (
        not isinstance(original_chunk_size, int)
        or isinstance(original_chunk_size, bool)
        or original_chunk_size < 1
    ):
        raise ValueError(
            "original_chunk_size must be a positive integer."
        )

    n_original = len(original_units)
    n_anonymized = len(anonymized_units)

    chunks = []

    for original_start in range(0, n_original, original_chunk_size):

        original_end = min(
            original_start + original_chunk_size,
            n_original,
        )

        anonymized_start = round(
            original_start * n_anonymized / n_original
        )

        anonymized_end = round(
            original_end * n_anonymized / n_original
        )

        original_text = " ".join(original_units[original_start:original_end])
        anonymized_text = " ".join(anonymized_units[anonymized_start:anonymized_end])

        # Compute semantic similarity
        embeddings = similarity_model.encode(
            [original_text, anonymized_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarity = float(np.dot(embeddings[0], embeddings[1]))

        chunks.append({
            "chunk_id": len(chunks),
            "original_indices": list(range(original_start, original_end)),
            "anonymized_indices": list(range(anonymized_start, anonymized_end)),
            "original_text": original_text,
            "anonymized_text": anonymized_text,
            "original_size": original_end - original_start,
            "anonymized_size": anonymized_end - anonymized_start,
            "similarity": similarity,
        })

    return pd.DataFrame(chunks)



def optimize_proportional_boundaries(
    original_units,
    anonymized_units,
    similarity_model,
    n_chunks=15,
    window=2,
    min_chunk_size=2,
    n_iterations=2,
):
    """
    Proportional chunking with local semantic boundary optimization.

    Each internal boundary can move within a local window around
    its initial proportional position.

    The objective is to maximize the sum of semantic similarities
    between corresponding chunks.
    """

    n_original = len(original_units)
    n_anonymized = len(anonymized_units)

    if n_original == 0 or n_anonymized == 0:
        raise ValueError("Both documents must contain at least one unit.")

    if n_chunks < 1:
        raise ValueError("n_chunks must be at least 1.")

    if n_original < n_chunks * min_chunk_size:
        raise ValueError("Not enough original units for the requested chunks.")

    if n_anonymized < n_chunks * min_chunk_size:
        raise ValueError("Not enough anonymized units for the requested chunks.")

    # ---------------------------------------------------------
    # 1. Initial proportional boundaries
    # ---------------------------------------------------------

    original_initial = np.rint(
        np.linspace(0, n_original, n_chunks + 1)
    ).astype(int)

    anonymized_initial = np.rint(
        np.linspace(0, n_anonymized, n_chunks + 1)
    ).astype(int)

    original_boundaries = original_initial.copy()
    anonymized_boundaries = anonymized_initial.copy()

    # ---------------------------------------------------------
    # 2. Embedding cache
    # ---------------------------------------------------------

    embedding_cache = {}

    def get_embedding(units, start, end, document_type):
        key = (document_type, start, end)

        if key not in embedding_cache:
            text = " ".join(units[start:end])

            embedding = similarity_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            embedding_cache[key] = embedding

        return embedding_cache[key]

    def cosine_similarity(embedding_a, embedding_b):
        return float(np.dot(embedding_a, embedding_b))

    def chunk_similarity(o_start, o_end, a_start, a_end):
        original_embedding = get_embedding(
            original_units, o_start, o_end, "original"
        )

        anonymized_embedding = get_embedding(
            anonymized_units, a_start, a_end, "anonymized"
        )

        return cosine_similarity(
            original_embedding,
            anonymized_embedding,
        )

    # ---------------------------------------------------------
    # 3. Local boundary optimization
    # ---------------------------------------------------------

    for _ in range(n_iterations):

        for boundary_idx in range(1, n_chunks):

            # Boundaries surrounding the current boundary
            o_left = original_boundaries[boundary_idx - 1]
            o_right = original_boundaries[boundary_idx + 1]

            a_left = anonymized_boundaries[boundary_idx - 1]
            a_right = anonymized_boundaries[boundary_idx + 1]

            # Candidate positions around initial proportional boundaries
            o_center = original_initial[boundary_idx]
            a_center = anonymized_initial[boundary_idx]

            o_candidates = range(
                max(o_left + min_chunk_size, o_center - window),
                min(o_right - min_chunk_size, o_center + window) + 1,
            )

            a_candidates = range(
                max(a_left + min_chunk_size, a_center - window),
                min(a_right - min_chunk_size, a_center + window) + 1,
            )

            best_score = -np.inf
            best_original_boundary = original_boundaries[boundary_idx]
            best_anonymized_boundary = anonymized_boundaries[boundary_idx]

            # Test all combinations of candidate boundaries
            for o_boundary in o_candidates:
                for a_boundary in a_candidates:

                    # Similarity of the two chunks before the boundary
                    left_similarity = chunk_similarity(
                        o_left,
                        o_boundary,
                        a_left,
                        a_boundary,
                    )

                    # Similarity of the two chunks after the boundary
                    right_similarity = chunk_similarity(
                        o_boundary,
                        o_right,
                        a_boundary,
                        a_right,
                    )

                    score = left_similarity + right_similarity

                    if score > best_score:
                        best_score = score
                        best_original_boundary = o_boundary
                        best_anonymized_boundary = a_boundary

            original_boundaries[boundary_idx] = best_original_boundary
            anonymized_boundaries[boundary_idx] = best_anonymized_boundary

    # ---------------------------------------------------------
    # 4. Build final chunks
    # ---------------------------------------------------------

    results = []

    for i in range(n_chunks):

        o_start = original_boundaries[i]
        o_end = original_boundaries[i + 1]

        a_start = anonymized_boundaries[i]
        a_end = anonymized_boundaries[i + 1]

        similarity = chunk_similarity(
            o_start, o_end, a_start, a_end
        )

        results.append({
            "chunk_id": i,
            "original_indices": list(range(o_start, o_end)),
            "anonymized_indices": list(range(a_start, a_end)),
            "original_text": " ".join(original_units[o_start:o_end]),
            "anonymized_text": " ".join(anonymized_units[a_start:a_end]),
            "original_size": o_end - o_start,
            "anonymized_size": a_end - a_start,
            "similarity": similarity,
        })

    return pd.DataFrame(results)



def estimate_n_chunks(
    original_units: list[str],
    tokenizer,
    target_tokens_per_chunk: int = 150,
) -> dict:
    """
    Estimate the number of chunks based on the original text length.

    The target chunk size is measured using the supplied tokenizer.
    Special tokens are excluded from the count.
    """
    if target_tokens_per_chunk < 1:
        raise ValueError(
            "target_tokens_per_chunk must be positive."
        )

    if not original_units:
        raise ValueError("original_units cannot be empty.")

    full_original_text = " ".join(
        unit.strip()
        for unit in original_units
        if unit.strip()
    )

    if not full_original_text:
        raise ValueError("original_units contain no text.")

    total_tokens = len(
        tokenizer(
            full_original_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )

    n_chunks = max(
        1,
        math.ceil(total_tokens / target_tokens_per_chunk),
    )

    return {
        "total_tokens": total_tokens,
        "target_tokens_per_chunk": target_tokens_per_chunk,
        "n_chunks": n_chunks,
        "expected_average_tokens": total_tokens / n_chunks,
    }