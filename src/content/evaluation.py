
import pandas as pd

from content.semantic_alignment import semantic_align
from content.tpi import estimate_all_term_ic


def tpi_from_alignment(
    original_text,
    alignment_result,
    tokenizer,
    model,
    device,
    strategy="N-TA",
    N=6,
):
    """Compute TPI from semantic alignment and term information content."""
    ic_table = estimate_all_term_ic(
        original_text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        strategy=strategy,
        N=N,
    )

    affected_indices = alignment_result["affected_original_indices"]

    affected = ic_table[
        ic_table["index"].isin(affected_indices)
    ].copy()

    tic_original = float(ic_table["IC"].sum())
    information_lost = float(affected["IC"].sum())

    if tic_original <= 0:
        raise ValueError(
            "TIC(original) must be strictly positive."
        )

    tpi = 1.0 - information_lost / tic_original
    tpi = max(0.0, min(1.0, tpi))

    return {
        "TIC_original": tic_original,
        "information_lost": information_lost,
        "TPI": tpi,
        "L_TPI": 1.0 - tpi,
        "affected_terms": affected,
        "IC_table": ic_table,
    }


def evaluate_content_pair(
    original_text,
    anonymized_text,
    tokenizer,
    model,
    device,
    similarity_threshold=0.7,
    tpi_strategy="N-TA",
    N=6,
):
    """Evaluate semantic alignment and TPI for one text pair."""
    alignment_result = semantic_align(
        original_text,
        anonymized_text,
        tokenizer,
        model,
        device,
        similarity_threshold,
    )

    tpi_result = tpi_from_alignment(
        original_text,
        alignment_result,
        tokenizer=tokenizer,
        model=model,
        device=device,
        strategy=tpi_strategy,
        N=N,
    )

    return {
        "original_text": original_text,
        "anonymized_text": anonymized_text,
        "alignment": alignment_result,
        "tpi": tpi_result,
        "TPI": tpi_result["TPI"],
        "L_TPI": tpi_result["L_TPI"],
    }


def evaluate_dataset(
    df,
    tokenizer,
    model,
    device,
    original_column="original",
    anonymized_column="anonymized",
    similarity_threshold=0.7,
    tpi_strategy="N-TA",
    N=6,
):
    """Evaluate semantic alignment and TPI for every row in a dataframe."""
    rows = []

    for row_idx, row in df.iterrows():
        original = str(row[original_column])
        anonymized = str(row[anonymized_column])

        result = evaluate_content_pair(
            original,
            anonymized,
            tokenizer=tokenizer,
            model=model,
            device=device,
            similarity_threshold=similarity_threshold,
            tpi_strategy=tpi_strategy,
            N=N,
        )

        rows.append({
            "row_idx": row_idx,
            "TPI": result["TPI"],
            "L_TPI": result["L_TPI"],
            "TIC_original": result["tpi"]["TIC_original"],
            "information_lost": result["tpi"]["information_lost"],
            "n_affected_terms": len(
                result["alignment"]["affected_original_indices"]
            ),
            "affected_terms": result["alignment"][
                "affected_original_terms"
            ],
        })

    return pd.DataFrame(rows)
