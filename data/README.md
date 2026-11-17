# Local data

Datasets, source recordings, generated transcripts, model outputs, and caches
are kept out of version control. Check each dataset's access and redistribution
terms before downloading or sharing it.

## MonoTED

Place an authorized MonoTED copy under `data/raw/MonoTED/`. The preparation
script writes its manifest and extracted utterance audio to
`data/processed/monoted/`; that generated directory is ignored by Git.

## Research notebooks

Notebook 08 expects local VoxCeleb audio and the versioned metadata file
`data/metadata/voxceleb_utterances.parquet`.

Notebook 10 expects local experiment inputs under `data/aggazzotti_test/`.
Those files are not distributed by this repository. Use data you are permitted
to access and process.

Do not commit raw recordings, transcripts, model responses, generated audio,
or caches unless their licenses explicitly permit redistribution and the
project maintainers have approved the addition.
