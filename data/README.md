# Data and local configuration

Raw datasets, generated audio, model weights, and experiment outputs are not
downloaded automatically. Obtain each dataset under its access and
redistribution terms. Local raw data and processed audio are ignored by Git;
the curated MonoTED manifest is the exception.

## MonoTED preparation

The maintained command is `make prepare-monoted`. By default it expects:

```text
data/raw/MonoTED/
├── audio_wav/
│   └── <recording-id>.wav
├── transcripts/
│   └── <recording-id>.txt
└── names_and_labels.csv
```

The CSV must have `Name` and `File` columns. `File` is the recording ID used
for matching the WAV and transcript files. Override the input directory with
`--root`; override outputs with `--output-dir`. The default output is
`data/processed/monoted/`, containing a JSONL manifest and extracted
timestamped WAV segments. The manifest is tracked with paths relative to the
repository root; the WAV segments remain local and are not included.

Example:

```bash
make prepare-monoted ARGS="--root /path/to/MonoTED --limit 1 --model-size tiny --device cpu --compute-type int8"
```

## Research notebook inputs

- **Notebook 08:** requires local VoxCeleb WAV audio and a locally provided
  `data/metadata/voxceleb_utterances.parquet` metadata file. Neither file is
  included in the repository. Prepare the metadata from VoxCeleb data you are
  authorized to use before running the notebook.
- **Notebook 09:** expects the VPC subset and anonymized WAV outputs under
  `data/vpc2026/`. Those inputs are not included.
- **Notebook 10:** requires experiment inputs under `data/aggazzotti_test/`
  and helper scripts under `external/long-form-speech-anonymization/`. The
  experiment inputs are not included in the public repository. Gemma inference
  uses `google/gemma-3-4b-it`; model access and adequate memory are required.
- Other notebooks may use additional VoxCeleb metadata, checkpoints, and
  locally generated outputs. Inspect their configuration cells before running.

For the extended Fisher workflow in `external/`, obtain the licensed Fisher
corpora and follow its `DATA.md`. Set the relevant `speech_attribution_dir` or
`ldc_trials_dir`, `trials_info_dir`, and `utterance_audio_dir` values in
`external/long-form-speech-anonymization/config.yaml`. That workflow has
separate dependencies documented in its own `requirements.txt`.

Check dataset and transcript licenses before using or publishing derived
outputs.

## Reproducibility and Git

Keep downloaded corpora and generated audio out of version control. The
MonoTED manifest is a curated exception. Before adding any other data file,
confirm its provenance, license, and necessity; stage specific files rather
than using `git add data/`.
