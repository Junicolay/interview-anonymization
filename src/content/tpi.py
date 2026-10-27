
"""Utilities for estimating information content and Text Preserved Information (TPI)."""

import math
import re

import pandas as pd
import numpy as np
import torch


def whitespace_terms(text: str) -> list[dict]:
    """Extract alphanumeric terms with character spans, excluding punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    return [
        {
            "term": match.group(),
            "start": match.start(),
            "end": match.end(),
        }
        for match in re.finditer(
            r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*",
            text,
        )
    ]


def term_subtokens(term: str, tokenizer) -> list[str]:
    """Return the tokenizer's subword tokens for a term."""
    return tokenizer.tokenize(term)


def build_masked_text(
    text: str,
    target_indices: list[int] | set[int],
    tokenizer,
) -> tuple[str, dict[int, int]]:
    """Replace selected terms with the corresponding number of mask tokens."""
    terms = whitespace_terms(text)
    target_indices = set(target_indices)

    invalid_indices = target_indices - set(range(len(terms)))
    if invalid_indices:
        raise ValueError(f"Invalid term indices: {sorted(invalid_indices)}")

    pieces = []
    cursor = 0
    mask_lengths = {}

    for index, item in enumerate(terms):
        pieces.append(text[cursor:item["start"]])

        if index in target_indices:
            subtokens = term_subtokens(item["term"], tokenizer)

            if not subtokens:
                raise ValueError(
                    f"No subtokens produced for term {item['term']!r}."
                )

            pieces.append(
                " ".join([tokenizer.mask_token] * len(subtokens))
            )
            mask_lengths[index] = len(subtokens)
        else:
            pieces.append(text[item["start"]:item["end"]])

        cursor = item["end"]

    pieces.append(text[cursor:])

    return "".join(pieces), mask_lengths


def masked_term_probabilities(
    text: str,
    target_indices: list[int] | set[int],
    tokenizer,
    model,
    device: str | torch.device,
) -> dict[int, dict]:
    """Estimate contextual probabilities for selected original terms.

    All selected terms are masked simultaneously. For a multi-subtoken term,
    its probability is the minimum probability assigned to its original
    subtokens at their corresponding masked positions.
    """
    terms = whitespace_terms(text)
    target_indices = set(target_indices)

    invalid_indices = target_indices - set(range(len(terms)))
    if invalid_indices:
        raise ValueError(f"Invalid term indices: {sorted(invalid_indices)}")

    if not target_indices:
        return {}

    masked_text, mask_lengths = build_masked_text(
        text,
        target_indices,
        tokenizer,
    )

    inputs = tokenizer(masked_text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    input_ids = inputs["input_ids"][0]
    mask_positions = [
        position
        for position, token_id in enumerate(input_ids)
        if token_id.item() == tokenizer.mask_token_id
    ]

    results = {}
    cursor = 0

    for index in sorted(target_indices):
        term = terms[index]["term"]
        subtokens = term_subtokens(term, tokenizer)
        n_subtokens = mask_lengths[index]

        positions = mask_positions[cursor:cursor + n_subtokens]

        if len(positions) != n_subtokens:
            raise RuntimeError(
                f"Could not align masks for term {term!r}."
            )

        token_probabilities = []

        for position, subtoken in zip(positions, subtokens):
            token_id = tokenizer.convert_tokens_to_ids(subtoken)

            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"Could not map subtoken {subtoken!r} to a token ID."
                )

            probability = float(
                probabilities[position, token_id].item()
            )
            token_probabilities.append(probability)

        results[index] = {
            "term": term,
            "subtokens": subtokens,
            "subtoken_probabilities": token_probabilities,
            "probability": min(token_probabilities),
        }

        cursor += n_subtokens

    return results


def information_content(probability: float) -> float:
    """Calculate IC(t) = -log(P(t))."""
    probability = float(probability)

    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("probability must be finite and in (0, 1].")

    return -math.log(max(probability, 1e-24))


def estimate_all_term_ic(
    text: str,
    tokenizer,
    model,
    device: str | torch.device,
    strategy: str = "N-TA",
    N: int = 6,
) -> pd.DataFrame:
    """Estimate information content for every term using S-TA or N-TA.

    S-TA masks one term per forward pass.
    N-TA masks terms in N alternating groups.
    """
    terms = whitespace_terms(text)

    columns = ["index", "term", "probability", "IC"]
    n_terms = len(terms)

    if n_terms == 0:
        return pd.DataFrame(columns=columns)

    if strategy not in {"S-TA", "N-TA"}:
        raise ValueError("strategy must be 'S-TA' or 'N-TA'.")

    if strategy == "N-TA" and (
        not isinstance(N, int) or isinstance(N, bool) or N < 1
    ):
        raise ValueError("N must be a positive integer for N-TA.")

    if strategy == "S-TA":
        groups = [[index] for index in range(n_terms)]
    else:
        groups = [
            list(range(offset, n_terms, N))
            for offset in range(min(N, n_terms))
        ]

    records = []

    for group in groups:
        group_results = masked_term_probabilities(
            text=text,
            target_indices=group,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        for index in group:
            result = group_results[index]
            probability = result["probability"]

            records.append({
                "index": index,
                "term": result["term"],
                "probability": probability,
                "IC": information_content(probability),
            })

    return (
        pd.DataFrame(records, columns=columns)
        .sort_values("index")
        .reset_index(drop=True)
    )


def total_information_content(ic_table: pd.DataFrame) -> float:
    """Calculate TIC as the sum of term-level information content."""
    if "IC" not in ic_table.columns:
        raise ValueError("ic_table must contain an 'IC' column.")

    if not ic_table["IC"].map(
        lambda value: isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    ).all():
        raise ValueError("IC values must be finite and non-negative.")

    return float(ic_table["IC"].sum())


def information_loss_from_spans(
    ic_table: pd.DataFrame,
    affected_term_indices: list[int] | set[int],
) -> tuple[pd.DataFrame, float]:
    """Return affected original terms and their total information content."""
    required_columns = {"index", "term", "IC"}
    if not required_columns.issubset(ic_table.columns):
        raise ValueError(
            f"ic_table must contain {sorted(required_columns)}."
        )

    affected_term_indices = set(affected_term_indices)
    available_indices = set(ic_table["index"].tolist())

    invalid_indices = affected_term_indices - available_indices
    if invalid_indices:
        raise ValueError(
            f"Unknown term indices: {sorted(invalid_indices)}"
        )

    affected = ic_table[
        ic_table["index"].isin(affected_term_indices)
    ].copy()

    lost_information = total_information_content(affected)

    return affected, lost_information


def tpi_from_spans(
    ic_table: pd.DataFrame,
    affected_term_indices: list[int] | set[int],
) -> dict:
    """Calculate TPI from the indices of affected original terms.

    TPI = 1 - affected information content / original TIC.
    """
    tic = total_information_content(ic_table)

    if not math.isfinite(tic) or tic <= 0:
        raise ValueError("TIC must be strictly positive.")

    affected, lost_information = information_loss_from_spans(
        ic_table,
        affected_term_indices,
    )

    tpi = 1.0 - lost_information / tic

    return {
        "TIC_original": tic,
        "information_lost": lost_information,
        "TPI": tpi,
        "affected_terms": affected,
    }


def term_indices(
    text: str,
    selected_terms: list[str] | set[str],
) -> list[int]:
    """Return indices of terms whose lowercase form matches selected terms."""
    selected_terms = {str(term).lower() for term in selected_terms}

    return [
        index
        for index, item in enumerate(whitespace_terms(text))
        if item["term"].lower() in selected_terms
    ]


