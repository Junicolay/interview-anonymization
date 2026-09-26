# Representations

This document summarizes the candidate representations currently considered in the project.

The goal is not to obtain a perfectly disentangled decomposition of speech. Instead, the representations are used as complementary views of the interview in order to study which types of information contribute to speaker identity and how anonymization methods affect them.

---

## 1. Global Speaker Identity Representation — `Z_id`

### Definition

Let `X` denote an audio utterance.

We define:

$Z_{id}(X) = E_{ECAPA}(X)$

where `E_ECAPA` is a pretrained ECAPA-TDNN speaker encoder.

### Model

ECAPA-TDNN.

### Paper

Desplanques et al. (2020),
*ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.*

### Representation

The encoder produces frame-level representations:

$H(X) = (h_1, ..., h_T)$

which are aggregated using attentive statistics pooling.

The pooled statistics are projected into the final speaker embedding:

$Z_{id} = P_{\theta}([\mu; \sigma])$

### Role in the project

`Z_id` is the global reference representation for speaker recognizability.

Because ECAPA-TDNN is trained for speaker discrimination, it is expected that:

$Z_{id} \rightarrow \hat{S}$

where `S` denotes speaker identity.

The representation will be used as a reference to measure how identifiable a speaker remains before and after anonymization.

---

## 2. Accent Representation — `Z_accent`

### Definition

Let `X` denote an audio utterance and `A` its accent label.

A pretrained XLSR encoder produces frame-level representations:

$H(X) = (h_1, ..., h_T)$

which are aggregated using mean pooling:

$\bar{h}(X) = \frac{1}{T} \sum_t h_t$

The pooled representation is passed through an information bottleneck `B_phi`.

We define:

$Z_{accent}(X) = B_{\phi}(\bar{h}(X))$

with:

$Z_{accent}(X) \in \mathbb{R}^{64}$

### Model

GenAID.

### Paper

Zhong et al. (2025),
*AccentBox: Towards High-Fidelity Zero-Shot Accent Generation.*

### Training objective

The representation is trained to remain informative about accent while reducing speaker-specific information.

The loss is:

$L = L_{accent} + \alpha L_{spk}$

with:

$L_{spk} = MSE(p(S \mid Z_{accent}), U)$

where `U` is the uniform distribution over speakers.

### Role in the project

`Z_accent` is a candidate representation of accent information.

The project will test whether speaker identity can still be recovered from it:

$Z_{accent} \rightarrow \hat{S}$

If speaker identity remains predictable, this indicates residual identity leakage through accent-related information.

---

## 3. Prosody Representation — `Z_prosody`

### Definition

Prosody includes temporal and melodic properties of speech such as:

* rhythm,
* intonation,
* stress,
* pitch variations,
* intensity contours.

The representation is based on the W-PRO model.

A pretrained WavLM encoder produces:

$H(X) = (h_1, ..., h_T)$

Statistics pooling is then applied:

$\mu(X) = \frac{1}{T} \sum_t h_t$

and:

$\sigma^2(X) = \frac{1}{T} \sum_t (h_t - \mu(X))^2$

The pooled representation is processed by fully connected layers.

We define:

$Z_{prosody}(X) = E_{W-PRO}(X)$

with:

$Z_{prosody}(X) \in \mathbb{R}^{250}$

### Model

W-PRO.

### Paper

Gengembre et al. (2024),
*Disentangling Prosody and Timbre Embeddings via Voice Conversion.*

### Training idea

Voice conversion is used to modify mainly the timbre while preserving much of the original prosodic structure.

The source speaker identity is then predicted from the converted speech.

This encourages the model to capture speaker-dependent information that survives timbre conversion.

### Role in the project

`Z_prosody` is treated as a candidate representation of prosodic and other
non-timbral speaker information that survives voice conversion.

The project will evaluate:

$Z_{prosody} \rightarrow \hat{S}$

---

## 4. Timbre Representation — `Z_timbre`

### Definition

Timbre refers to relatively stable acoustic characteristics of a speaker's voice that help distinguish speakers independently of linguistic content.

We define:

$Z_{timbre}(X) = E_{TBR}(X)$

where `E_TBR` is an encoder intended to capture speaker-related information complementary to the prosody representation.

### Model

Timbre encoder from the framework of Gengembre et al.

### Paper

Gengembre et al. (2024),
*Disentangling Prosody and Timbre Embeddings via Voice Conversion.*

### Training idea

The prosody representation is first learned and kept fixed.

A second encoder is then trained so that:

$[Z_{prosody}; Z_{timbre}] \rightarrow \hat{S}$

remains predictive of speaker identity.

### Role in the project

`Z_timbre` is treated as a candidate timbre representation.

It should not be interpreted as a perfectly pure timbre representation, because the training objective only encourages it to capture speaker information that complements `Z_prosody`.

The distinction with `Z_id` is important:

* `Z_id` = global speaker representation,
* `Z_timbre` = candidate representation of one more specific source of speaker-related information.

---

## 5. Lexical and Linguistic Style Representation — `Z_style`

### Definition

Let:

$W = ASR(X)$

denote the transcription of the interview.

The representation is based on a content-independent style encoder.

A RoBERTa encoder produces token-level representations:

$H(W) = (h_1, ..., h_n)$

which are aggregated using mean pooling:

$\bar{h}(W) = \frac{1}{n} \sum_i h_i$

We define:

$Z_{style}(X) = E_{CISR}(ASR(X))$

with:

$Z_{style}(X) \in \mathbb{R}^{768}$

### Model

Content-Independent Style Representation (CISR).

### Paper

Wegmann et al. (2022),
*Same Author or Just Same Topic? Towards Content-Independent Style Representations.*

### Training objective

The model uses contrastive authorship verification.

For an anchor text `W_A`, a positive text from the same author `W_A+`, and a negative text from another author `W_B-`, the representation is trained so that:

$d(E_{CISR}(W_A), E_{CISR}(W_{A+})) < d(E_{CISR}(W_A), E_{CISR}(W_{B-}))$

with a triplet loss:

$L_{style} = \max(d(z_A, z_{A+}) - d(z_A, z_{B-}) + m, 0)$

### Role in the project

`Z_style` captures candidate speaker-dependent linguistic habits such as:

* word choice,
* fillers,
* discourse markers,
* contractions,
* syntactic habits.

The project will evaluate:

$Z_{style} \rightarrow \hat{S}$

Strong speaker predictability would indicate that a speaker can still be identified from linguistic behavior even when acoustic information is removed.

---

## 6. Semantic Content Representation — `Z_content`

### Definition

Let:

$W = ASR(X)$

denote the transcription of the original interview.

We define:

$Z_{content}(X) = E_{sem}(ASR(X))$

where `E_sem` is a Sentence-BERT-style semantic encoder.

The candidate representation has dimension:

$Z_{content}(X) \in \mathbb{R}^{d}$

### Model

Sentence-BERT-style semantic encoder.

### Paper

Reimers and Gurevych (2019),
*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*

### Role in the project

Unlike the previous representations, `Z_content` should be preserved during anonymization.

Let:

$X' = T(X)$

denote the anonymized utterance.

The desired behavior is:

$Z_{content}(X') \approx Z_{content}(X)$

Semantic preservation is measured using cosine similarity:

$Sim_{content}(X, X') =
\frac{Z_{content}(X)^T Z_{content}(X')}
{|Z_{content}(X)| |Z_{content}(X')|}$

The corresponding content loss is:

$L_{content} = 1 - Sim_{content}(X, X')$

The anonymization system should minimize $L_{content}$.

---

## 7. Important interpretation

The representations are not assumed to be statistically independent.

Accent, timbre, prosody, linguistic style, and semantic content may overlap and share information.

Therefore, speaker prediction performance across representations should not be interpreted as a direct ranking of perfectly independent sources of identity.

Instead, each representation should be interpreted according to its own training objective.

In particular:

* high speaker predictability from `Z_timbre` is partly expected by construction;
* speaker predictability from `Z_accent` represents residual identity leakage, since speaker-specific information is explicitly discouraged during training;
* `Z_id` acts as the global reference for speaker recognizability;
* `Z_content` represents information that should be preserved rather than removed.
