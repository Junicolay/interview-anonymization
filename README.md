# Interview Anonymization

Research code for speech privacy and content-preservation experiments. This
repository is not a finished, end-to-end anonymization system.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/Junicolay/interview-anonymization.git
cd interview-anonymization
make setup
```

`make setup` creates a project-local environment using Python 3.9 or newer,
installs the CPU PyTorch build, notebook/research dependencies, and developer
tools. Python 3.11 is recommended. To install only dependencies for the
MonoTED preparation script, use `make setup-minimal`.

## Prepare MonoTED audio

MonoTED is not included. Obtain it under its applicable access terms and place
it in the layout described in [`data/README.md`](data/README.md). Then run a
small CPU smoke test:

```bash
make prepare-monoted ARGS="--root data/raw/MonoTED --limit 1 --model-size tiny --device cpu --compute-type int8"
```

For a full run, omit `--limit 1` and optionally choose a larger model such as
`--model-size medium`. The default output is
`data/processed/monoted/manifest.jsonl`, with extracted WAV segments in
`data/processed/monoted/utterance_audio/`. Existing manifest entries are
skipped; `--overwrite` reprocesses recordings. See
`make help` or `make prepare-monoted ARGS="--help"` for options.

The script uses Faster-Whisper timestamps as segment boundaries. ASR segment
boundaries may not match human-defined utterances, and ASR errors can affect
both segment text and timing.

## CPU and GPU environments

CPU is the default and is the most portable option:

```bash
make setup
```

For CUDA 12.1 PyTorch wheels:

```bash
make setup-gpu
```

The GPU setup selects the PyTorch CUDA 12.1 index. Faster-Whisper uses
CTranslate2 separately; GPU transcription also requires a compatible CUDA and
cuDNN runtime on the host. See the
[Faster-Whisper GPU requirements](https://opennmt.net/CTranslate2/installation.html#gpu).
Use CPU flags (`--device cpu --compute-type int8`) when those runtime libraries
are unavailable.

The `cpu` and `gpu` extras are alternatives; do not install both into the same
environment.

## Notebooks and development

Start JupyterLab with the project dependencies:

```bash
make notebooks
```

Run tests and lint checks locally:

```bash
make test
make lint
```

The notebooks are exploratory research records. Some require additional local
datasets, pretrained models, or code under `external/`; see
[`data/README.md`](data/README.md) and each notebook before running it. Notebook
outputs are retained as research artifacts and may contain results or paths
from the original author’s machine.

The root `uv.lock` covers the maintained project and its CPU/GPU, notebook,
and research extras. The separate code under `external/` has its own
`requirements.txt`; the complete voice-anonymization and Fisher trial
workflows may need those additional dependencies and licensed datasets.

## Repository map

- `src/`: reusable ASR, audio, dataset, and content-evaluation utilities.
- `scripts/prepare_monoted.py`: MonoTED transcription and timestamped audio
  extraction command.
- `notebooks/`: research experiments, including speaker-verification and
  content-preservation analyses.
- `external/long-form-speech-anonymization/`: separately maintained research
  code used by selected experiments.
- `data/README.md`: required datasets, expected layout, and local output paths.

## Data and reproducibility

Raw speech datasets, model weights, caches, and generated outputs should remain
local. Verify redistribution terms before publishing or sharing experiment
inputs or derived outputs. Dataset access and licensing are the user's
responsibility.

For reproducible runs, record the dataset version, model identifier, command
line, relevant parameters, and Git commit. Dependencies are declared in
`pyproject.toml` and pinned by the committed `uv.lock`.
