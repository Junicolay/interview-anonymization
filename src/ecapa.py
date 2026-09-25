from pathlib import Path

import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier


PROJECT_DIR = Path.home() / "Documents" / "interview-anonymization"


def load_ecapa():
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(
            PROJECT_DIR
            / "pretrained_models"
            / "spkrec-ecapa-voxceleb"
        )
    )

    return classifier


def extract_embedding(path, classifier):
    audio, sample_rate = sf.read(path, dtype="float32")

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16 kHz, got {sample_rate} Hz for {path}"
        )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    waveform = torch.from_numpy(audio).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)

    return embedding.squeeze().cpu()