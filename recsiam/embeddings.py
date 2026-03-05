import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoModel
from dotenv import load_dotenv

load_dotenv()
_HF_TOKEN = os.getenv("HF_TOKEN")

# Public DINOv2-small model (non-gated, matches embedding_preprocess.py)
_DEFAULT_MODEL = "facebook/dinov2-small"


class DINOEmbedding(nn.Module):
    """
    DINO feature extractor wrapper.
    Returns CLS embedding of shape (B, D).
    """

    _shared_model = None  # shared per-process instance

    def __init__(
        self,
        model_name=_DEFAULT_MODEL,
        pretrained=True,
        freeze=True,
    ):
        super().__init__()

        if DINOEmbedding._shared_model is None:
            DINOEmbedding._shared_model = AutoModel.from_pretrained(
                model_name, token=_HF_TOKEN
            )

        self.model = DINOEmbedding._shared_model

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.model.eval()

        # ImageNet normalization (DINO uses this)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, x):
        """
        x: torch.Tensor (B, 3, H, W) in [0,1]
        Images are resized to 224x224 to guarantee a fixed number of patches.
        """

        if x.ndim != 4:
            raise ValueError("Input must be (B, C, H, W)")

        # Resize to the patch size expected by DINO (224x224)
        if x.shape[-2:] != (224, 224):
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

        x = self.normalize(x)

        outputs = self.model(pixel_values=x)

        # CLS token
        embedding = outputs.last_hidden_state[:, 0]

        return embedding


def no_embedding(*args, **kwargs):
    return nn.Identity()


_EMBEDDINGS = {
    "dinov3": DINOEmbedding,
    "none": no_embedding,
}


def get_embedding(name):
    if name not in _EMBEDDINGS:
        raise ValueError(f"Valid embeddings: {list(_EMBEDDINGS.keys())}")
    return _EMBEDDINGS[name]