import numpy as np
import pytest
import torch

from src.attacker_baseline import ecapa


def test_load_ecapa_uses_configured_model_directory(monkeypatch, tmp_path):
    expected_classifier = object()
    captured = {}

    def fake_from_hparams(**kwargs):
        captured.update(kwargs)
        return expected_classifier

    monkeypatch.setattr(
        ecapa.EncoderClassifier,
        "from_hparams",
        fake_from_hparams,
    )

    result = ecapa.load_ecapa(model_dir=tmp_path)

    assert result is expected_classifier
    assert captured["source"] == "speechbrain/spkrec-ecapa-voxceleb"
    assert captured["savedir"] == str(tmp_path)


def test_extract_embedding_from_mono_audio(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    expected_embedding = torch.tensor([[[1.0, 2.0, 3.0]]])

    monkeypatch.setattr(
        ecapa.sf,
        "read",
        lambda *args, **kwargs: (audio, 16000),
    )

    class FakeClassifier:
        def encode_batch(self, waveform):
            assert waveform.shape == (1, 3)
            return expected_embedding

    result = ecapa.extract_embedding(audio_path, FakeClassifier())

    assert torch.equal(result, expected_embedding.squeeze())


def test_extract_embedding_converts_stereo_to_mono(monkeypatch, tmp_path):
    audio_path = tmp_path / "stereo.wav"
    audio = np.array(
        [[0.2, 0.4], [0.6, 0.8]],
        dtype=np.float32,
    )
    captured = {}

    monkeypatch.setattr(
        ecapa.sf,
        "read",
        lambda *args, **kwargs: (audio, 16000),
    )

    class FakeClassifier:
        def encode_batch(self, waveform):
            captured["waveform"] = waveform
            return torch.tensor([[[1.0, 2.0]]])

    ecapa.extract_embedding(audio_path, FakeClassifier())

    expected_waveform = torch.tensor([[0.3, 0.7]])
    assert torch.allclose(captured["waveform"], expected_waveform)


def test_extract_embedding_rejects_wrong_sample_rate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ecapa.sf,
        "read",
        lambda *args, **kwargs: (
            np.array([0.1, 0.2], dtype=np.float32),
            44100,
        ),
    )

    with pytest.raises(ValueError, match="Expected 16 kHz"):
        ecapa.extract_embedding(tmp_path / "audio.wav", object())
