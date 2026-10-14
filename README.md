# Interview Anonymization

Research project on reducing speaker-identifying information in interviews
while preserving their semantic content.

**Supervisor:** Prof. Andrea Cavallaro, EPFL MINTS Lab.

## Research objective

The project investigates how speech anonymization can reduce the information
that reveals a speaker's identity without unnecessarily altering what the
speaker says.

The research considers speaker identity, speech characteristics and semantic
content. The current repository contains exploratory notebooks and an
engineering baseline for evaluating speaker identifiability.

## Current implementation

The implemented speaker-verification pipeline uses pretrained ECAPA-TDNN
embeddings from SpeechBrain.

It supports:

- Reproducible generation of positive and negative verification trials.
- Audio embedding extraction with a local SQLite cache.
- Cosine-similarity scoring.
- Equal Error Rate (EER), its associated threshold, False Acceptance Rate
  (FAR) and False Rejection Rate (FRR).
- Machine-readable experiment reports containing configuration, input hashes,
  Git commit information, counts and runtime.
- Automated tests and lint checks through pytest, Ruff and GitHub Actions.

This is an evaluation baseline, not a complete speech anonymization system.
The effectiveness of anonymization methods and their impact on semantic
content require separate experiments.

## Repository structure

```text
.
├── .github/workflows/   # Continuous integration
├── data/                # Dataset metadata and trial files
├── notebooks/           # Research exploration and experiments
├── reports/             # Generated experiment reports
├── scripts/             # Command-line experiment entry points
├── src/                 # Reusable Python modules
└── tests/               # Automated tests
```

Notebooks preserve the research workflow. Reusable functionality and the
speaker-verification experiment entry point are maintained separately in
`src/` and `scripts/`.

## Requirements

Use Python 3.11, matching the version configured in continuous integration.

Create and activate an isolated environment, then install the development
dependencies:

```bash
conda create -n interview-anon python=3.11
conda activate interview-anon

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The requirements include the experiment dependencies as well as pytest and
Ruff.

## Data prerequisites

The speaker-verification script expects local VoxCeleb audio files and a
Parquet metadata file. These datasets are not included in this repository.

By default, it expects:

- Metadata: `data/metadata/voxceleb_utterances.parquet`
- Audio: `voxceleb1/wav/`

The metadata must contain the columns `speaker_id`, `video_id` and `path`.
Audio paths must resolve to existing local files.

If your data is stored elsewhere, provide the paths using the command-line
arguments documented below.

## Run speaker verification

From the repository root:

```bash
python scripts/run_speaker_verification.py
```

The script uses a deterministic random seed and, by default, selects
100 eligible speakers with 10 positive and 10 negative trials per speaker.

You can override the data locations and experiment configuration:

```bash
python scripts/run_speaker_verification.py \
  --metadata /path/to/voxceleb_utterances.parquet \
  --audio-root /path/to/voxceleb1/wav \
  --seed 42 \
  --n-speakers 100
```

The script generates verification trials, computes or retrieves embeddings,
scores the trials and writes a report.

**Important:** the default output paths include
`data/trials/voxceleb_verification_trials_legacy_reproduction.csv`,
`data/cache/ecapa_embeddings.sqlite` and `reports/speaker_verification/`.
The script writes to these locations, so check existing files before running
it. Use `--trials-output`, `--cache-db` and `--output-dir` to choose alternative
paths when necessary.

The pretrained model may need to be downloaded on the first cache miss.
Runtime depends on the dataset, hardware and number of uncached audio files.

## Development checks

Run the automated tests:

```bash
python -m pytest -q
```

Run Ruff:

```bash
ruff check scripts src tests
```

GitHub Actions runs these checks on pushes and pull requests.

## Reproducibility

For a reproducible experiment, record the dataset and metadata version,
experiment parameters, random seed, model version and Git commit.

The speaker-verification report records the Git commit when available,
metadata and trial-file SHA-256 hashes, configuration, counts, metrics and
runtime. These records support traceability but do not by themselves guarantee
bitwise-identical results across different environments or hardware.

## Research status

The current focus is establishing reliable speaker-identifiability baselines
and improving the software structure around the experiments.

Future work includes evaluating anonymization methods and investigating the
trade-off between reducing speaker-identifying information and preserving
semantic content.
