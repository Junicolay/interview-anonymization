# Interview Anonymization: Identity Protection and Content Preservation

Research project at the **EPFL MINTS Lab**, supervised by Prof. Andrea Cavallaro.

## Overview

Speech anonymization aims to reduce the information in a recording that reveals a speaker's identity while preserving the information conveyed by the speech.

This project investigates the trade-off between **speaker privacy** and **semantic content preservation** in interview recordings.

An anonymization method may make a speaker harder to identify while unintentionally changing words, removing important details, or altering meaning. Evaluating both objectives is therefore essential.

The project combines speaker-identification experiments, speech processing, natural language processing, and quantitative evaluation to investigate this trade-off.

## Research objectives

The project addresses three complementary questions:

1. **Identity protection:** How much speaker-identifying information remains in an anonymized recording?
2. **Content preservation:** How much of the original meaning and information survives anonymization?
3. **Evaluation methodology:** Which metrics best capture content loss, and where do their conclusions agree or disagree?

## Methodology

### 1. Speaker identifiability

The repository implements an ECAPA-TDNN-based speaker-verification baseline using pretrained SpeechBrain embeddings.

The baseline includes:

- Generation of positive and negative speaker-verification trials.
- Extraction and caching of speaker embeddings.
- Cosine-similarity scoring.
- Equal Error Rate (EER) and associated decision threshold.
- False Acceptance Rate (FAR) and False Rejection Rate (FRR).

This is an evaluation baseline, not a complete anonymization system.

### 2. Content preservation

Content-preservation experiments compare original and anonymized speech, using transcripts to investigate whether anonymization changes the information conveyed.

| Method | Purpose |
| --- | --- |
| Automatic speech recognition (ASR) | Transcribe original and anonymized recordings using a consistent setup. |
| Semantic alignment | Identify words or terms that are missing or altered. |
| Term Preservation Index (TPI) | Quantify preserved information using term-level information content. |
| Semantic similarity | Compare contextual text representations. |
| Natural Language Inference (NLI) | Investigate semantic consistency between original and anonymized transcripts. |

These methods capture different aspects of content preservation. Their agreement and disagreements are part of the research.

### 3. Experimental analysis

Research notebooks document experiments involving speaker representations, content representations, metric validation, semantic alignment, and original/anonymized speech pairs.

The objective is to understand not only whether information appears to be lost, but also which parts of the content are affected and why different metrics may reach different conclusions.

## Repository structure

```text
.
├── data/                   # Dataset metadata and experiment artifacts
├── notebooks/              # Research experiments and analyses
│   └── archive/             # Archived notebooks
├── scripts/                # Experiment entry points
├── src/
│   ├── attacker_baseline/   # Speaker-verification baseline
│   ├── content/             # Content-preservation evaluation
│   ├── asr.py               # Speech transcription utilities
│   ├── paths.py             # Shared project paths
│   └── utils.py             # Shared utilities
├── tests/                   # Tests for selected modules
├── requirements.txt
└── requirements-dev.txt
```

**Design principle:** notebooks support exploration and interpretation; reusable functionality belongs in `src/`, and repeatable command-line experiments belong in `scripts/`.

## Getting started

### Requirements

Python 3.11 is recommended. Some experiments require pretrained models and local datasets that are not included in this repository.

Clone the repository:

```bash
git clone https://github.com/Junicolay/interview-anonymization.git
cd interview-anonymization
```

Create an isolated environment and install development dependencies:

```bash
conda create -n interview-anon python=3.11
conda activate interview-anon

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Some notebooks may require additional model downloads or experiment-specific setup. Check the relevant notebook before running it.

## Running the speaker-verification baseline

From the repository root:

```bash
python scripts/run_speaker_verification.py
```

By default, the script expects:

- Metadata at `data/metadata/voxceleb_utterances.parquet`
- VoxCeleb audio under `voxceleb1/wav/`

The metadata must contain the columns `speaker_id`, `video_id`, and `path`.

You can override the data paths and experiment configuration:

```bash
python scripts/run_speaker_verification.py \
  --metadata /path/to/voxceleb_utterances.parquet \
  --audio-root /path/to/voxceleb1/wav \
  --seed 42 \
  --n-speakers 100
```

Inspect all supported arguments with:

```bash
python scripts/run_speaker_verification.py --help
```

The script generates verification trials, extracts or reuses cached embeddings, computes evaluation metrics, and writes experiment reports.

Check output paths before running experiments, particularly when reproducing earlier results.

## Data and reproducibility

Datasets and raw audio are not distributed with this repository. Use external datasets in accordance with their applicable access and usage conditions.

Large datasets, model weights, local caches, and generated artifacts should generally remain outside version control.

For reproducible experiments, record where applicable:

- Dataset and metadata versions.
- Model identifiers and configurations.
- Random seeds and experiment parameters.
- Input data and trial-file hashes.
- Evaluation results and the Git commit used.

The speaker-verification pipeline records experiment configuration, input hashes, metrics, runtime, and Git revision when available.

These records improve traceability but do not guarantee identical results across environments or hardware.

See [`data/README.md`](data/README.md) for additional information.

## Development

Run the tests with:

```bash
python -m pytest -q
```

If Ruff is installed, run lint checks with:

```bash
ruff check scripts src tests
```

## Scope and limitations

The repository contains research experiments, evaluation methods, and a speaker-verification baseline. It is not a finished, end-to-end speech anonymization product.

- **Speaker identifiability:** The current speaker verification baseline evaluates speaker identity using individual utterances. However, interview anonymization must account for multiple utterances from the same speaker, as identity-related information accumulated across an interview may enable identification even when individual utterances provide limited evidence. The current evaluation therefore does not fully capture the risk of speaker re-identification across an entire interview.
- **ASR:** ASR-based content evaluation depends on transcription quality and may introduce errors unrelated to anonymization.
- Semantic similarity, NLI, and term-preservation metrics measure different properties and are not interchangeable.
- Results depend on the selected data, models, thresholds, and experimental configuration.

The research aims to establish reliable evaluation methods and better understand the trade-off between speaker privacy and semantic content preservation.

## Acknowledgements

This work is conducted at the EPFL MINTS Lab under the supervision of Prof. Andrea Cavallaro.

