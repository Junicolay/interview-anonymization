"""
Text diagnostics and document comparison utilities.
"""

import pandas as pd


def count_words(text: str) -> int:
    """Count whitespace-separated words."""
    return len(text.split())


def count_characters(text: str) -> int:
    """Count characters."""
    return len(text)


def document_statistics(
    text: str,
    units: list[str] | None = None,
) -> dict:
    """
    Compute basic statistics for a document.
    """
    words = count_words(text)

    stats = {
        "word_count": words,
        "character_count": count_characters(text),
    }

    if units is not None:
        unit_count = len(units)

        stats["unit_count"] = unit_count
        stats["average_words_per_unit"] = (
            words / unit_count if unit_count > 0 else 0.0
        )

    return stats


def compare_documents(
    original_document: str,
    anonymized_document: str,
    original_units: list[str],
    anonymized_units: list[str],
) -> pd.DataFrame:
    """
    Compare basic statistics between original and anonymized documents.
    """
    original_stats = document_statistics(
        original_document,
        original_units,
    )

    anonymized_stats = document_statistics(
        anonymized_document,
        anonymized_units,
    )

    metrics = [
        ("Number of words", "word_count"),
        ("Number of utterances", "unit_count"),
        ("Average words per utterance", "average_words_per_unit"),
        ("Document length (characters)", "character_count"),
    ]

    rows = []

    for label, key in metrics:
        original_value = original_stats[key]
        anonymized_value = anonymized_stats[key]

        if original_value != 0:
            change = (
                (anonymized_value - original_value)
                / original_value
                * 100
            )
        else:
            change = float("nan")

        rows.append({
            "Metric": label,
            "Original": original_value,
            "Anonymized": anonymized_value,
            "Change (%)": round(change, 2),
        })

    return pd.DataFrame(rows)


def word_retention(
    original_document: str,
    anonymized_document: str,
) -> dict:
    """
    Compute word retention and word reduction.
    """
    original_words = count_words(original_document)
    anonymized_words = count_words(anonymized_document)

    if original_words == 0:
        raise ValueError("Original document contains no words.")

    retention = anonymized_words / original_words
    reduction = 1.0 - retention

    return {
        "original_word_count": original_words,
        "anonymized_word_count": anonymized_words,
        "word_retention": retention,
        "word_reduction": reduction,
    }


def token_length_audit(
    texts: dict[str, str],
    tokenizers: dict,
    add_special_tokens: bool = True,
) -> pd.DataFrame:
    """
    Audit token counts for several texts and tokenizers.

    Parameters
    ----------
    texts:
        Mapping of text labels to text content.
    tokenizers:
        Mapping of tokenizer labels to tokenizer objects.
    """
    rows = []

    for text_name, text in texts.items():
        row = {"Text": text_name}

        for tokenizer_name, tokenizer in tokenizers.items():
            token_ids = tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )["input_ids"]

            row[f"{tokenizer_name} tokens"] = len(token_ids)

        rows.append(row)

    return pd.DataFrame(rows)