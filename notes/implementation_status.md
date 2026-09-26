# Implementation Status

## Representation status

| Representation | Candidate model | Status | Role |
|---|---|---|---|
| `Z_id` | ECAPA-TDNN | Implemented | Global speaker reference / speaker attacker input |
| `Z_accent` | GenAID / AccentBox | To implement | Accent-related speaker information |
| `Z_prosody` | W-PRO | To implement | Prosodic / non-timbral speaker information |
| `Z_timbre` | Gengembre et al. framework | To implement | Candidate timbre-related speaker information |
| `Z_style` | CISR | To implement | Lexical / linguistic speaker information |
| `Z_content` | Sentence-BERT-style encoder | To implement | Semantic content to preserve |

## Model vs. project implementation

For the current ECAPA baseline:

`X -> pretrained ECAPA -> Z_id -> downstream speaker attacker -> S_hat`

- **Pretrained model:** ECAPA-TDNN from the literature.
- **Project implementation:** model loading, embedding extraction, data selection, trial construction, downstream speaker attacker, and evaluation.
- **Logistic regression:** a downstream classifier trained on ECAPA embeddings; it is not part of ECAPA-TDNN.
- **1-NN / cosine verification:** uses ECAPA embedding similarity and does not train a new neural speaker encoder.

## Current experimental baseline

- Speaker identification: logistic regression and 1-NN.
- Speaker verification: cosine similarity.
- Verification metric: EER.
- Current reference result: approximately 1% EER on the custom VoxCeleb verification protocol.

## Dataset strategy

VoxCeleb is currently used for speaker-identity experiments.

A different dataset may be selected when a representation has specific data requirements. This is particularly relevant for accent representation.

The project focuses on English speech.

## Anonymization direction

The main direction is direct / targeted speech anonymization:

`X -> T -> X'`

ASR--TTS remains a possible comparison baseline:

`X -> ASR -> W -> TTS -> X'_TTS`

## Evaluation goal

For each anonymization method, compare speaker identifiability before and after anonymization while measuring semantic-content preservation.
