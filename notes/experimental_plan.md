# Experimental Plan

## 1. Experimental objective

The experiments aim to compare anonymization methods according to two questions:

1. How much speaker-identifying information remains after anonymization?
2. How well is the original interview content preserved?

The analysis will be performed on English speech only.

---

## 2. General workflow

For each anonymization method `T_m`:

`X -> T_m -> X'_m`

the same evaluation pipeline will be applied to the original and anonymized speech.

### Original speech

`X -> representation extraction -> speaker prediction`

### Anonymized speech

`X'_m -> representation extraction -> speaker prediction`

The difference between the two will be used to quantify the effect of anonymization.

---

## 3. Dataset strategy and split

The dataset will be selected according to the requirements of each representation. VoxCeleb is currently used for speaker-identity experiments, while representation-specific datasets may be used when necessary, for example for accent representation.

The selected dataset will be divided into training, validation, and test sets.

The split must avoid leakage between segments coming from the same recording.

When possible:

- training and test utterances should come from different recordings,
- the same speakers may appear in both sets for speaker-identification experiments,
- anonymization evaluation should always be performed on held-out data.

The exact split strategy will depend on the selected dataset.

---

## 4. Baseline 1 — Original speaker identifiability

Before anonymization, speaker-identification models will be trained on each selected representation.

For a representation `Z_j`:

`f_j(Z_j) -> S_hat`

The main output will be speaker-identification performance on held-out data.

For `Z_id`, the representation is obtained using the pretrained ECAPA-TDNN speaker encoder:

`X -> ECAPA -> Z_id -> speaker attacker -> S_hat`

ECAPA-TDNN itself is not trained as part of this project. The downstream speaker attacker and evaluation protocol are implemented for this project.

This establishes the reference level of identifiability before anonymization.

---

## 5. Main anonymization direction — Targeted speech anonymization

The main direction is direct / targeted speech anonymization:

`X -> T -> X'`

The anonymization method will modify selected speaker-related characteristics while attempting to preserve semantic content.

Possible targets include:

- timbre,
- accent,
- prosody,
- lexical / linguistic style.

The first targeted method will be selected from the literature based on implementation availability and compatibility with the selected datasets.

---

## 6. Baseline 2 — ASR–TTS anonymization

ASR–TTS remains a possible strong comparison baseline:

`X -> ASR -> W -> TTS -> X'_TTS`

The reconstructed speech will be evaluated using the same privacy and utility protocol as the targeted anonymization methods.

This baseline is useful because it is expected to strongly modify acoustic speaker characteristics.

---

## 7. Privacy evaluation

For each anonymization method `T_m` and each selected representation `Z_j`, compute:

`Z_j(X'_m)`

and predict the original speaker:

`f_j(Z_j(X'_m)) -> S_hat`

Privacy will be assessed by comparing speaker-identification performance before and after anonymization.

Example:

`Acc_original,j`

versus

`Acc_anonymized,m,j`

A larger decrease indicates stronger privacy protection.

Speaker verification may be used as an additional evaluation, building on the current ECAPA verification baseline.

---

## 8. Utility evaluation

Utility will primarily measure semantic preservation between the original and anonymized utterances.

For each pair `(X, X'_m)`, compute:

`L_content(X, X'_m)`

using the content-preservation definition selected for the project.

The main comparison will be:

- privacy gain,
- semantic-content loss.

Additional utility measures may be added later if required.

---

## 9. Comparison between methods

Each anonymization method will be summarized using a common evaluation table.

Example:

| Method | Speaker accuracy | Privacy gain | Content loss |
|---|---:|---:|---:|
| Original | TBD | 0 | 0 |
| ASR–TTS | TBD | TBD | TBD |
| Targeted method 1 | TBD | TBD | TBD |
| Targeted method 2 | TBD | TBD | TBD |

The goal is to compare privacy protection and semantic preservation across methods.

---

## 10. Initial implementation order

1. Set up the EPFL/HPC storage.
2. Load and inspect the selected datasets.
3. Create train / validation / test splits.
4. Implement the remaining representation extraction pipelines.
5. Train the original speaker-identification baselines.
6. Evaluate speaker information contained in each representation.
7. Implement the first targeted anonymization method.
8. Re-run the privacy and utility evaluation.
9. Implement ASR–TTS as a comparison baseline if useful.
10. Compare methods using the same evaluation protocol.
11. Add additional anonymization methods only if time permits.

---

## 11. Reproducibility

For every experiment, record:

- dataset version,
- split definition,
- pretrained model version,
- model parameters,
- random seed,
- evaluation metrics,
- output results.

All experiments should be reproducible from the repository.
