"""
Transcript preprocessing utilities.
"""

import re


def clean_transcript(
    raw_transcript: str,
    min_words: int = 3,
) -> str:
    """
    Remove timestamps and stage directions, then normalize whitespace.
    """
    if not isinstance(raw_transcript, str):
        raise TypeError("raw_transcript must be a string.")

    text = raw_transcript

    # Remove timestamps (e.g. 00:28, 1:02:15)
    text = re.sub(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
        "",
        text,
    )

    # Remove common stage directions
    stage_directions = r"(?:laughter|applause|music|cheering)"

    text = re.sub(
        rf"\({stage_directions}[^)]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        rf"\[{stage_directions}[^\]]*\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_into_utterances(
    text: str,
    min_words: int = 3,
) -> list[str]:
    """
    Split text into sentence-like utterances while preserving punctuation.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    parts = re.split(r'([.!?]["”’]?\s+)', text.strip())

    utterances = []

    for i in range(0, len(parts), 2):
        sentence = parts[i].strip()

        if i + 1 < len(parts):
            sentence += parts[i + 1].strip()

        if len(sentence.split()) >= min_words:
            utterances.append(sentence)

    return utterances


def prepare_transcript(
    raw_transcript: str,
    min_words: int = 3,
) -> tuple[str, list[str]]:
    """
    Clean a raw transcript and split it into utterances.

    Returns
    -------
    clean_text:
        Cleaned transcript.
    utterances:
        Sentence-level utterances.
    """
    clean_text = clean_transcript(raw_transcript)

    utterances = split_into_utterances(
        clean_text,
        min_words=min_words,
    )

    return clean_text, utterances


def prepare_sentence_units(
    text: str,
    language: str = "english",
) -> list[str]:
    """
    Split a document into sentence-level units using NLTK.
    """
    import nltk
    from nltk.tokenize import sent_tokenize

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    try:
        return sent_tokenize(text, language=language)

    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        return sent_tokenize(text, language=language)