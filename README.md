# Interview Anonymization

Research code for studying privacy and content preservation in long-form speech.
The project is in progress; it is not a finished anonymization system.

## Overview

Speech anonymization aims to reduce the information in a recording that reveals a speaker's identity while preserving the information conveyed by the speech.

This project investigates the trade-off between **speaker privacy** and **semantic content preservation** in interview recordings.

An anonymization method may make a speaker harder to identify while unintentionally changing words, removing important details, or altering meaning. Evaluating both objectives is therefore essential.

The project combines speech processing, natural language processing, and
quantitative evaluation to investigate this trade-off.

## Research direction: Linguistic and stylometric style transfer

### Linguistic and stylometric style transfer for long-form speech anonymization

In long-form recordings, an adversary may re-identify a speaker by analyzing persistent linguistic and stylistic characteristics, such as accent, dialectal variations, syntax, vocabulary, and individual phrasing habits.

Traditional speech anonymization methods primarily focus on modifying voice characteristics or removing explicit Personally Identifiable Information (PII), such as names and locations. However, they generally do not address persistent linguistic patterns that can reveal a speaker's identity across extended recordings.

The objective of this research project is to investigate **linguistic and stylometric anonymization** through sequence-to-sequence style-transfer models. We aim to develop an AI framework capable of transforming or obfuscating individual speaking styles, syntactic patterns, and dialectal markers while preserving the semantic content of the original speech.

The planned research focuses primarily on:

- French and Italian.
- German, including Swiss German dialects.

Evaluation will involve human participants and real-world datasets, in collaboration with **Radio Télévision Suisse (RTS)**.

The central research challenge is to reduce speaker-identifying linguistic
information while maintaining the meaning and coherence of long-form speech.
The style-transfer anonymizer itself is not implemented in this repository yet.

## Current implementation

The repository currently provides:

- Content-preservation utilities for semantic alignment, term information
  content, and transcript diagnostics.
- ASR/audio helpers and a MonoTED adapter that prepares timestamped utterance
  audio.
- Research notebooks for selected content-preservation experiments.
- A retained Gemma-based experiment notebook (10) that requires local inputs
  and separately maintained code under `external/`.

These are research components, not an end-to-end anonymization pipeline.

## Content preservation

Experiments compare original and transformed transcripts to investigate
whether anonymization changes the information conveyed.

| Method | Purpose |
| --- | --- |
| Automatic speech recognition (ASR) | Transcribe original and anonymized recordings using a consistent setup. |
| Semantic alignment | Identify words or terms that are missing or altered. |
| Term Preservation Index (TPI) | Quantify preserved information using term-level information content. |
| Semantic similarity | Compare contextual text representations. |
| Natural Language Inference (NLI) | Investigate semantic consistency between original and anonymized transcripts. |

These methods capture different aspects of content preservation. Their
agreement and disagreement are part of the research.

## Repository structure

```text
.
├── data/                   # Data instructions; datasets stay local
├── external/                # Temporary, separate long-form experiment code
├── notebooks/              # Selected research experiments
├── notes/                   # Research notes
├── scripts/                # Experiment entry points
├── src/
│   ├── content/             # Content-preservation evaluation
│   ├── datasets/            # Dataset adapters and shared recording types
│   ├── asr.py               # Speech transcription utilities
│   ├── audio.py             # WAV utilities
│   └── paths.py             # Shared project paths
├── tests/                   # Unit tests for maintained utilities
└── pyproject.toml           # Dependencies, extras, and package metadata
```

**Design principle:** notebooks support exploration and interpretation; reusable functionality belongs in `src/`, and repeatable command-line experiments belong in `scripts/`.

### Preparing MonoTED utterances

The MonoTED adapter loads each talk, its reference transcript, and metadata.
Preparation then transcribes the audio with Faster-Whisper (`medium` by
default), uses its native timestamped segments as variable-duration utterance
units, and extracts a WAV for each segment. It does not align the reference
transcript to audio or split ASR segments into fixed-length chunks. The manifest
stores both the complete reference and ASR transcripts, per-talk normalized
WER, and each utterance's ASR text and timestamps.

```bash
python scripts/prepare_monoted.py
```

By default, the script reads `data/raw/MonoTED/` and writes
`data/processed/monoted/manifest.jsonl` and extracted audio under
`data/processed/monoted/utterance_audio/`. Use `--limit 1` for a one-talk
smoke test; `--model-size`, `--device`, `--compute-type`, and `--language`
configure Faster-Whisper. If an utterance WAV already exists, the script stops
instead of risking a mismatch with newly generated ASR timestamps; use
`--overwrite` to regenerate the cuts.
The terminal shows a progress bar for the current talk (based on ASR segment
timestamps) and an overall counter for completed talks. Model loading is
announced separately.
The extracted WAVs duplicate the source audio samples, so a full run requires
substantial additional disk space.

The ASR segments are the cut boundaries; the reference transcript is only a
transcription-quality check and does not affect segmentation. Native
Faster-Whisper segments may not correspond exactly to human-defined utterances,
and ASR errors can affect both their text and boundaries. This preparation
stage does not yet wire in Aggazzotti prompt/inference or privacy-evaluation
stages. Since MonoTED has one talk per speaker, segment-based privacy results
must be described as within-talk evaluation, not cross-recording
speaker re-identification.

## Getting started

### Requirements

Python 3.9 or newer is supported; Python 3.11 is recommended. Some
experiments require pretrained models and local datasets that are not included
in this repository.

Clone the repository:

```bash
git clone https://github.com/Junicolay/interview-anonymization.git
cd interview-anonymization
```

Create an isolated environment and install the project plus the experiment and
development extras:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[asr,content,dev,notebooks]"
```

For the CUDA 12.1 PyTorch build used on the EPFL GPU environment:

```bash
module load python/3.11
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  -e ".[asr,content,dev,gpu,notebooks]"
```

Some notebooks may require additional model downloads or experiment-specific setup. Check the relevant notebook before running it.

## Data and reproducibility

Datasets and raw audio are not distributed with this repository. Use external datasets in accordance with their applicable access and usage conditions.

Large datasets, model weights, local caches, and generated artifacts should generally remain outside version control.

For reproducible experiments, record where applicable:

- Dataset and metadata versions.
- Model identifiers and configurations.
- Random seeds and experiment parameters.
- Input data and trial-file hashes.
- Evaluation results and the Git commit used.

These records improve traceability but do not guarantee identical results
across environments or hardware.

See [`data/README.md`](data/README.md) for additional information.

## Development

Run the tests with:

```bash
python -m pytest -q
```

Run lint checks with:

```bash
python -m ruff check scripts src tests
```

## Scope and Limitations

The experiments are part of an ongoing research project and do not constitute a
finished, end-to-end speech anonymization system.

- **NLI reliability:** Initial experiments have revealed inconsistencies in the Natural Language Inference (NLI) evaluation. In some cases, identical original and anonymized sentences are classified as contradictory. We are investigating the causes of these unexpected predictions to understand the reliability and limitations of NLI for evaluating content preservation.

- **ASR:** ASR-based content evaluation depends on transcription quality. Transcription errors may affect the measured content preservation and can be mistaken for changes introduced by anonymization.

- **Content-preservation metrics:** Semantic similarity, NLI, and term-preservation metrics capture different aspects of content preservation. Their results are complementary and should not be interpreted as interchangeable measures of information loss.

- **Anonymizer scope:** The planned linguistic/style-transfer anonymizer is not
  yet implemented. Existing notebooks and utilities are exploratory and should
  not be read as evidence of a complete anonymization system.

The research aims to develop reliable evaluation methods and better understand the trade-off between speaker privacy and semantic content preservation.
