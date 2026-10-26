from pathlib import Path

import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = (
    PROJECT_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"
)


def load_ecapa(model_dir: str | Path = DEFAULT_MODEL_DIR):
    """Load the pretrained ECAPA-TDNN speaker encoder."""
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(model_dir),
    )
    return classifier


def extract_embedding(
    path: str | Path,
    classifier: EncoderClassifier,
) -> torch.Tensor:
    """Extract a speaker embedding from a mono or stereo 16 kHz audio file."""
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
