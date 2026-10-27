import re

import pandas as pd
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoModel, AutoTokenizer


def load_embedding_model(model_name, device=None):
    """Load a tokenizer and model for contextual embeddings."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device



TERM_PATTERN = r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*"

def extract_terms(text):
    """Return lexical terms and their character spans."""
    return [
        {
            "term": m.group(),
            "start": m.start(),
            "end": m.end(),
        }
        for m in re.finditer(TERM_PATTERN, text)
    ]

def normalize_term(term):
    return term.lower()

def show_terms(text):
    return pd.DataFrame(extract_terms(text))

def term_embeddings(text, tokenizer, device, model):
    """
    Compute one contextual BERT embedding per lexical term.

    Returns
    -------
    terms : list of dict
        Terms with character spans.
    embeddings : np.ndarray, shape (n_terms, hidden_size)
        Mean-pooled contextual embedding for each term.
    """
    terms = extract_terms(text)

    if not terms:
        return terms, np.empty((0, model.config.hidden_size), dtype=np.float32)
    
    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.inference_mode():
        outputs = model(**encoded)

    hidden = outputs.hidden_states[-1][0].detach().cpu()

    term_vectors = []

    for term in terms:
        overlapping_token_ids = []

        for token_id, (tok_start, tok_end) in enumerate(offsets):
            # Special tokens generally have (0, 0).
            if tok_end <= tok_start:
                continue

            # Token overlaps the character span of the term.
            if tok_start < term["end"] and tok_end > term["start"]:
                overlapping_token_ids.append(token_id)

        if not overlapping_token_ids:
            raise ValueError(
                f"No tokenizer token found for term: {term['term']!r}"
            )

        vector = hidden[overlapping_token_ids].mean(dim=0)
        term_vectors.append(vector.numpy())

    return terms, np.vstack(term_vectors).astype(np.float32)

def cosine_similarity_matrix(A, B):
    """Pairwise cosine similarity between rows of A and B."""
    A_norm = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    B_norm = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return A_norm @ B_norm.T


def build_similarity_matrix(
    original_text,
    anonymized_text,
    tokenizer,
    model,
    device,
):
    """Build pairwise semantic similarities between lexical terms."""
    original_terms, original_embeddings = term_embeddings(
    original_text, tokenizer, device, model
    )
    anonymized_terms, anonymized_embeddings = term_embeddings(
    anonymized_text, tokenizer, device, model
    )

    if not original_terms or not anonymized_terms:
        similarity = np.empty(
            (len(original_terms), len(anonymized_terms)),
            dtype=np.float32,
        )
    else:
        similarity = cosine_similarity_matrix(
            original_embeddings,
            anonymized_embeddings,
        )

    return original_terms, anonymized_terms, similarity


def semantic_align(
    original_text,
    anonymized_text,
    tokenizer,
    model,
    device,
    similarity_threshold,
):
    """
    Align original and anonymized terms using contextual embeddings.

    Returns a dictionary containing:
      - original_terms
      - anonymized_terms
      - similarity_matrix
      - alignments
      - affected_original_indices
      - affected_original_terms
    """
    original_terms, anonymized_terms, similarity = build_similarity_matrix(
        original_text,
        anonymized_text,
        tokenizer,
        model,
        device,
    )

    n = len(original_terms)
    m = len(anonymized_terms)

    alignments = []
    matched_original = set()
    matched_anonymized = set()

    if n > 0 and m > 0:
        # linear_sum_assignment minimizes cost, so negate similarity.
        row_ind, col_ind = linear_sum_assignment(-similarity)

        for i, j in zip(row_ind, col_ind):
            score = float(similarity[i, j])

            if score >= similarity_threshold:
                matched_original.add(i)
                matched_anonymized.add(j)

                same_surface = (
                    normalize_term(original_terms[i]["term"])
                    == normalize_term(anonymized_terms[j]["term"])
                )

                status = "exact_match" if same_surface else "semantic_match"

                alignments.append({
                    "original_idx": i,
                    "original_term": original_terms[i]["term"],
                    "anonymized_idx": j,
                    "anonymized_term": anonymized_terms[j]["term"],
                    "similarity": score,
                    "status": status,
                })

    # Original terms that could not be confidently aligned
    # are considered affected.
    affected_original_indices = [
        i for i in range(n)
        if i not in matched_original
    ]

    # Sort alignments by original position.
    alignments = sorted(
        alignments,
        key=lambda x: x["original_idx"]
    )

    affected_original_terms = [
        original_terms[i]["term"]
        for i in affected_original_indices
    ]

    return {
        "original_terms": original_terms,
        "anonymized_terms": anonymized_terms,
        "similarity_matrix": similarity,
        "alignments": alignments,
        "affected_original_indices": affected_original_indices,
        "affected_original_terms": affected_original_terms,
        "unmatched_anonymized_indices": [
            j for j in range(m)
            if j not in matched_anonymized
        ],
    }

