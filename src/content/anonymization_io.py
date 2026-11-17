"""
Input/output utilities for text anonymization experiments.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union


def load_jsonl(path: Union[str, Path]) -> list[dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}."
                ) from exc

    if not rows:
        raise ValueError(f"No records found in {path}.")

    return rows


def extract_response_content(
    response_row: dict[str, Any],
    row_index: Optional[int] = None,
) -> str:
    """
    Extract generated text from an OpenAI-compatible response row.
    """
    try:
        content = (
            response_row["response"]["body"]["choices"][0]
            ["message"]["content"]
        )

    except (KeyError, IndexError, TypeError) as exc:
        label = (
            f"response {row_index}"
            if row_index is not None
            else "response"
        )

        raise ValueError(
            f"Could not extract generated content from {label}."
        ) from exc

    if not isinstance(content, str):
        raise TypeError("Generated response content must be a string.")

    return content.strip()


def validate_response_rows(
    response_rows: list[dict[str, Any]],
    expected_status_code: int = 200,
) -> None:
    """
    Check that all inference requests completed successfully.
    """
    failed_indices = []

    for index, row in enumerate(response_rows):
        status_code = (
            row.get("response", {}).get("status_code")
        )

        if status_code != expected_status_code:
            failed_indices.append(index)

    if failed_indices:
        raise RuntimeError(
            f"{len(failed_indices)} inference requests failed. "
            f"Failed row indices: {failed_indices}"
        )


def extract_anonymized_segments(
    response_rows: list[dict[str, Any]],
    validate: bool = True,
) -> list[str]:
    """
    Extract generated segments while preserving their original order.
    """
    if not response_rows:
        raise ValueError("response_rows cannot be empty.")

    if validate:
        validate_response_rows(response_rows)

    return [
        extract_response_content(row, row_index=index)
        for index, row in enumerate(response_rows)
    ]


def reconstruct_utterances(
    anonymized_segments: list[str],
    separator: str = "##",
) -> list[str]:
    """
    Reconstruct anonymized utterances from generated segments.
    """
    if not separator:
        raise ValueError("separator cannot be empty.")

    anonymized_utterances = []

    for segment in anonymized_segments:
        segment_utterances = [
            utterance.strip()
            for utterance in segment.split(separator)
            if utterance.strip()
        ]

        anonymized_utterances.extend(segment_utterances)

    return anonymized_utterances


def reconstruct_document(
    utterances: list[str],
    separator: str = "\n",
) -> str:
    """
    Reconstruct a document from a list of utterances.
    """
    return separator.join(
        utterance.strip()
        for utterance in utterances
        if utterance.strip()
    )


def save_text(
    text: str,
    path: Union[str, Path],
) -> Path:
    """
    Save a text document.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(text, encoding="utf-8")

    return path


def save_json(
    data: Any,
    path: Union[str, Path],
) -> Path:
    """
    Save data as a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path


def build_utterance_dataset(
    utterances: list[str],
    call_id: str,
    speaker_id: str,
    speaker_gender: str,
) -> dict[str, Any]:
    """
    Build the utterance JSON structure expected by the original pipeline.
    """
    if speaker_gender not in {"m", "f"}:
        raise ValueError("speaker_gender must be 'm' or 'f'.")

    return {
        call_id: {
            speaker_id: {
                "gender": speaker_gender,
                "text": utterances,
            }
        }
    }