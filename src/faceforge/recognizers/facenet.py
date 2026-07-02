"""Facenet embedding wrapper (placeholder).
Implements the interface used by FaceForge. Real models should be downloaded separately.
"""
from typing import Optional
import numpy as np

try:
    from facenet_pytorch import InceptionResnetV1
except Exception:
    InceptionResnetV1 = None

class FaceNetRecognizer:
    def __init__(self, device: str = "cpu"):
        if InceptionResnetV1 is None:
            raise RuntimeError("facenet-pytorch is required for FaceNetRecognizer")
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.device = device

    def embed(self, face: "np.ndarray") -> np.ndarray:
        # Accepts cropped aligned face as numpy array HWC uint8
        # Convert to tensor in actual implementation; simplified here
        import torch
        from torchvision import transforms
        img = transforms.ToTensor()(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(img)
        return emb.cpu().numpy().ravel()

    def identify(self, embedding: np.ndarray):
        # Identification logic belongs to higher-level classifier/FAISS index
        return None
