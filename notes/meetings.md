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
