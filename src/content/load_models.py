import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer

from content.nli import load_nli_model as _load_nli_model


def load_tpi_model(
    model_name="bert-base-uncased",
    device=None,
):
    """Load the tokenizer and masked language model required for TPI."""

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        output_hidden_states=True,
    ).to(device)

    model.eval()

    return tokenizer, model, device


def load_similarity_model(
    model_name="sentence-transformers/paraphrase-mpnet-base-v2",
    device=None,
):
    """Load the SentenceTransformer model used for semantic similarity."""

    model = SentenceTransformer(
        model_name,
        device=str(device) if device is not None else None,
    )

    return model


def load_nli_model(
    model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    device=None,
):
    """Load the NLI tokenizer, model, device and label mapping."""

    return _load_nli_model(
        model_name=model_name,
        device=device,
    )