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

## Meeting 4 — Long-form evaluation and privacy–utility trade-off

### Main discussion

The discussion focused on how to extend the current evaluation framework from the initial speaker baseline to the actual interview anonymization problem.

### Decisions

#### 1. Long-form speaker attack

- The speaker attacker should use **all utterances available in an interview/video**, rather than considering isolated utterances only.
- This reflects the realistic setting where an attacker can accumulate information across the interview.
- Aggazzotti's evaluation protocol is particularly relevant for this extension.

#### 2. Privacy–utility trade-off

- Potentially identifying information does not necessarily need to be removed completely.
- A possible anonymization strategy is to **generalize information**:

  `I live in Lausanne -> I live in Switzerland`

- Whether an information is compromising depends on the context.
- The current assumption is therefore to **not explicitly model compromising information for now**.
- This assumption should nevertheless be implemented as a **parameter**, initially fixed to zero, so that it can be relaxed later if time permits.

#### 3. Content loss

- Semantic similarity and contradiction detection provide two components of the current content evaluation.
- The main unresolved issue is **information loss**.

Example:

`W_o = A + B -> W_a = A`

- Part of the original information disappears without necessarily producing a contradiction.
- Need to determine how to quantify and integrate this loss into `L_content`.

#### 4. Possible future direction

Aggazzotti is particularly relevant to the project and should be studied further.

A possible later direction is:

`Stylistic anonymization -> Semantic fidelity`

- Modify linguistic style while preserving the underlying semantic content.
- This directly raises the trade-off between **stylistic anonymization and semantic fidelity** discussed in Aggazzotti.

### Infrastructure

- Prof. Cavallaro will provide access to the **EPFL HPC**.
- Storage is therefore no longer a major constraint.
- This opens the possibility of using **datasets beyond VoxCeleb** when required by the different representations.

### Project organization

- GitHub access was provided to Prof. Cavallaro.
- The repository should now be cleaned and organized.
- A concise project summary should be added so that the current state, previous work, decisions, and next steps are easily accessible.

### Next steps

1. Adapt the speaker attacker to the **long-form interview setting**.
2. Re-read Aggazzotti and study its evaluation methodology.
3. Define how to incorporate **information loss** into `L_content`.
4. Keep the compromising-information parameter at zero for now.
5. Set up the EPFL HPC environment.
6. Clean the GitHub repository and add a project summary.
7. Consider additional datasets when required.
8. Explore **stylistic anonymization vs. semantic fidelity** if time permits.
