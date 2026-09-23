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

## 3. Dataset split strategy

The dataset will be divided into training, validation, and test sets.

The split must avoid leakage between segments coming from the same recording.

When possible:

- training and test utterances should come from different recordings,
- the same speakers may appear in both sets for speaker identification experiments,
- anonymization evaluation should always be performed on held-out data.

The exact split strategy will depend on the selected dataset.

---

## 4. Baseline 1 — Original speaker identifiability

Before anonymization, speaker-identification models will be trained on each selected representation.

For a representation `Z_j`:

`f_j(Z_j) -> S_hat`

The main output will be speaker-identification performance on held-out data.

This establishes the reference level of identifiability before anonymization.

---

## 5. Baseline 2 — ASR–TTS anonymization

A strong anonymization baseline will use:

`X -> ASR -> W -> TTS -> X'_TTS`

The reconstructed speech will then be evaluated using the same privacy and utility protocol as all other methods.

This baseline is useful because it is expected to strongly remove acoustic speaker characteristics.

---

## 6. Targeted anonymization methods

One or more targeted anonymization methods will be selected from the literature.

Each method will modify only a subset of speaker-related characteristics.

Possible categories include:

- timbre modification,
- accent modification,
- prosody modification,
- lexical-style modification.

The exact methods will be selected after reviewing implementation availability and compatibility with the chosen datasets.

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

Speaker verification may be added as an additional evaluation if needed.

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
| Method 1 | TBD | TBD | TBD |
| Method 2 | TBD | TBD | TBD |

The goal is to identify the methods that provide the best privacy–utility trade-off.

---

## 10. Initial implementation order

1. Load and inspect the selected dataset.
2. Create train / validation / test splits.
3. Implement representation extraction.
4. Train the original speaker-identification baseline.
5. Implement the ASR–TTS baseline.
6. Re-run the privacy and utility evaluation.
7. Add the first targeted anonymization method.
8. Compare methods using the same evaluation protocol.
9. Add additional methods only if time permits.

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
