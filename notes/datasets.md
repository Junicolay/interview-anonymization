# Candidate Datasets

## Language scope

The project will focus on English speech only.

Restricting the project to English simplifies dataset selection, representation extraction, ASR evaluation, and comparison across speakers. Multilingual variation will not be considered in the initial experimental setup.

---

## Selection criteria

A suitable dataset should ideally provide:

- English speech only,
- multiple speakers,
- multiple utterances per speaker,
- speaker identity labels,
- transcriptions,
- sufficient audio duration,
- possibly accent metadata,
- a license compatible with research use.

Because the project studies interview anonymization, conversational or interview-style speech is preferred when possible.

---

## VoxCeleb

### Strengths
- Large number of speakers.
- Multiple utterances per speaker.
- Natural speech collected from interviews and similar media.
- Well suited for speaker identification and speaker verification.
- Particularly relevant for evaluating global speaker recognizability.

### Limitations
- Transcriptions are not the main focus of the dataset.
- Accent labels are not systematically available.
- Less directly suitable for lexical-style analysis unless transcripts are generated separately.
- Additional preprocessing may be required to isolate the English-only subset.

### Possible use in the project
- Global speaker identity baseline.
- Evaluation of `Z_id`.
- Speaker identification.
- Speaker verification.
- Re-identification experiments before and after anonymization.

---

## Common Voice — English subset

### Strengths
- English audio with corresponding text transcriptions.
- Large number of speakers.
- Useful for experiments involving linguistic content.
- Some speaker metadata may be available.
- Potentially useful for accent-related experiments depending on the available metadata.

### Limitations
- Speaker metadata is not complete for every sample.
- Number of recordings per speaker can be highly unbalanced.
- Speech is generally prompted/read rather than natural interview speech.
- Accent labels may be incomplete or noisy.

### Possible use in the project
- Evaluation of `Z_accent`.
- Evaluation of `Z_content`.
- Semantic-content preservation experiments.
- ASR-related evaluation.
- Potential comparison of speaker information across accents.

---

## LibriSpeech

### Strengths
- English-only speech.
- Reliable transcriptions.
- Multiple utterances per speaker.
- Standard benchmark for speech and ASR research.
- Relatively clean audio.

### Limitations
- Read speech rather than conversational or interview speech.
- Less representative of natural lexical habits.
- Less suitable for linguistic-style analysis.
- Accent diversity and metadata are limited compared with other datasets.

### Possible use in the project
- Content-preservation experiments.
- ASR baseline.
- Evaluation of semantic or linguistic-content representations.
- Controlled experiments where clean transcripts are important.

---

## Current direction

The project may require more than one dataset because the different representations have different data requirements.

A possible setup is:

- `VoxCeleb` for speaker identity, speaker verification, and privacy evaluation;
- `Common Voice (English)` for accent and content-related experiments;
- `LibriSpeech` as a clean English ASR/content-preservation benchmark if needed.

The final dataset selection should be based on the representations and anonymization methods retained in the experimental setup.

---

## Open questions

- Is one dataset sufficient, or should different datasets be used for different parts of the project?
- Which dataset provides enough repeated recordings per speaker for reliable train/test splits?
- Should transcriptions be taken from the dataset or generated using ASR?
- Which dataset best matches the final anonymization scenario of real interviews?
- Are the selected datasets compatible with the pretrained models used for the different representations?
