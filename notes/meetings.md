# Meeting Notes

## Meeting 1 — Project introduction

### Main discussion
- Prof. Cavallaro presented two possible research projects.
- I chose the interview anonymization project.
- The general objective is to reduce speaker-identifying information in an interview while preserving the semantic content.
- Potential identity-related factors discussed include voice characteristics, accent, prosody, and lexical habits.
- The project should include both an analysis of where identity information is present and methods to reduce it.

### Main questions raised
- What exactly should be anonymized?
- What information should be preserved?
- What threat model should be considered?
- How should privacy and utility be evaluated?
- What would be an appropriate scope for an 8 ECTS project?

### Follow-up
- Review related literature on speaker anonymization and privacy-preserving representations.
- Think about possible mathematical representations of speaker identity, accent, prosody, lexical style, and semantic content.
- Start defining a possible research direction.


## Meeting 2 — Representation framework and project formalization

### Material presented
- Presented the first project document containing candidate representations for:
  - global speaker identity,
  - accent,
  - prosody,
  - timbre,
  - lexical / linguistic style,
  - semantic content.

### Feedback / decisions
- Need to find suitable datasets independently.
- Need to define the content-preservation objective mathematically.
- Representations such as `Z_accent`, `Z_prosody`, and `Z_content` should be justified using existing papers.
- The representations should be treated as candidate views of the signal rather than perfectly disentangled components.
- The research question still needs to be defined more precisely.

### Next steps
- Select suitable datasets.
- Finalize literature-based definitions of the main representations.
- Define `L_content` mathematically.
- Decide which information should be removed and which information should be preserved.
- Build the first experimental baseline for speaker identifiability.


## Meeting 3 — ECAPA baseline and anonymization direction

### Material presented

- Presented the implementation of the first speaker-identity baseline using ECAPA-TDNN.
- Clarified that ECAPA-TDNN is a pretrained speaker encoder from the literature, not a model developed from scratch for this project.
- Used ECAPA embeddings as the basis for a speaker attacker.
- Evaluated speaker identifiability using:
  - multinomial logistic regression,
  - 1-nearest-neighbour classification with cosine similarity,
  - speaker verification using cosine similarity and EER.
- The speaker-identification experiments achieved 100% accuracy and 1.00 macro-F1 on the tested subset.
- The speaker-verification baseline achieved an EER of approximately 1%, providing a reference level of speaker identifiability before anonymization.

### Important implementation distinction

The ECAPA pipeline consists of two separate components:

`X -> pretrained ECAPA -> Z_id -> downstream speaker attacker -> S_hat`

- **ECAPA-TDNN:** pretrained model imported from the literature.
- **ECAPA embedding extraction:** implemented as part of the project.
- **Speaker attacker:** downstream models and evaluation protocols implemented for the project.
- **Logistic regression:** trained on the extracted ECAPA embeddings; it is not part of ECAPA itself.
- **1-NN / cosine verification:** uses the geometry of the ECAPA embeddings without training a new neural speaker encoder.

### Feedback / decisions

- The main anonymization direction should be the **second approach: direct / targeted speech anonymization**.
- The ASR--TTS approach remains interesting and may be used as a strong baseline for comparison.
- ECAPA is now clearly positioned as the **speaker attacker / evaluation model** rather than as a model developed within the project.
- The next stage should focus on implementing and evaluating the other candidate representations.
- Different datasets may be used when necessary for a specific representation. In particular, a dedicated dataset may be required for accent representation.
- EPFL/HPC storage will be provided to handle the larger datasets and models required for the project.

### Current status

The first speaker-identifiability baseline is now implemented.

The current experimental logic is:

`X -> representation -> speaker attacker`

and, after anonymization:

`X' = T(X) -> representation -> same speaker attacker`

The goal is to compare speaker identifiability before and after anonymization while preserving semantic content.

### Next steps

1. Set up the EPFL/HPC storage.
2. Implement the remaining candidate representations:
   - `Z_accent`,
   - `Z_prosody`,
   - `Z_timbre`,
   - `Z_style`,
   - `Z_content`.
3. Use representation-specific datasets when required.
4. Evaluate speaker information contained in each representation.
5. Start implementing the direct / targeted anonymization approach.
6. Keep ASR--TTS as a possible baseline for comparison.

## Meeting 4 — Speaker attacker, privacy–utility trade-off, and content loss

### Material presented

- Presented the first speaker-attacker baseline based on a pretrained ECAPA-TDNN speaker encoder.
- Used ECAPA embeddings for speaker verification and evaluated the baseline using EER.
- Verified the baseline on non-anonymized speech, obtaining results consistent with the expected results from the literature.
- Presented the current content-preservation evaluation framework based on semantic similarity and NLI.
- Discussed the difficulty of quantifying information loss when part of the original content disappears, e.g.
  `W_o = A + B -> W_a = A`.

### Feedback / decisions

#### Long-form speaker attacker

- The speaker attacker should use **all utterances from an interview/video**, rather than evaluating isolated utterance pairs only.
- This better reflects the actual interview setting, where an attacker can exploit information accumulated across multiple utterances.
- The approach used by Aggazzotti is particularly relevant and should be studied in more detail.

#### Privacy–utility trade-off

- Potentially sensitive information may need to be generalized rather than simply removed.
- For example:

  `I live in Lausanne -> I live in Switzerland`

- This can be viewed as a form of generalization of potentially identifying information.
- However, whether a piece of information is compromising depends strongly on the context.
- For example, mentioning Lausanne may be problematic in one setting but not in another.
- Therefore, the privacy–utility trade-off depends on the speaker, context, population, and attacker.

#### Assumption on compromising information

- For the current scope of the project, the assumption that compromising information does not need to be explicitly removed can be kept.
- However, this assumption should be **implemented as a parameter rather than hard-coded**.
- The parameter can initially be fixed to zero.
- If time permits, the parameter can later be changed to explicitly account for compromising information and study its effect on the privacy–utility trade-off.

#### Content loss

- Semantic similarity and contradiction detection are already considered.
- The remaining difficulty is to quantify **information loss**.
- In particular, the case

  `W_o = A + B -> W_a = A`

  should receive an appropriate penalty even when no contradiction is detected.
- Need to investigate how this information loss can be incorporated into the overall content-preservation loss.

### Research direction

- The general reasoning and current framework were considered appropriate.
- Aggazzotti should be reviewed more carefully because the paper addresses several aspects that overlap with the project.
- A potential future direction is to study **linguistic-style anonymization while preserving semantic content**.

In particular, Aggazzotti highlights the trade-off between:

`Stylistic anonymization <-> Semantic fidelity`

- This could become a more specific research direction later in the project, especially if time permits.

### Infrastructure / project organization

- Prof. Cavallaro will provide access to the **EPFL HPC infrastructure**.
- Storage will therefore no longer be a major constraint, making it possible to consider **additional datasets beyond VoxCeleb** in the future.
- GitHub access was provided to Prof. Cavallaro.
- The repository should now be cleaned and organized.
- A summary of the work completed so far should be added to the repository so that the current state of the project is easy to understand and follow at any time.

### Next steps

1. Improve the speaker attacker and adapt it to the **long-form interview setting**.
2. Study and reproduce the relevant evaluation methodology from Aggazzotti.
3. Re-read Aggazzotti in detail and identify the parts most relevant to the current project.
4. Find a way to incorporate **information loss** into `L_content`.
5. Keep the compromising-information parameter fixed to zero for now.
6. If time permits, vary this parameter and study its effect on the privacy–utility trade-off.
7. Obtain access to the EPFL HPC infrastructure.
8. Consider additional datasets beyond VoxCeleb once the HPC setup is available.
9. Clean and organize the GitHub repository.
10. Add a concise summary of the project, experiments, decisions, and current status to the repository.
11. Consider **stylistic anonymization vs. semantic fidelity** as a possible later research direction.
