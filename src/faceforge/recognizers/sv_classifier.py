"""Recognizers: embedding extraction and simple classifier/verifier."""
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Recognizer:
    def embed(self, face: "np.ndarray") -> np.ndarray:
        raise NotImplementedError

    def identify(self, embedding: np.ndarray) -> Optional[str]:
        raise NotImplementedError

class SimpleVerifier(Recognizer):
    """A toy verifier that computes a normalized pixel vector as embedding and does nearest neighbor on a small store."""
    def __init__(self):
        self.store = {}

    def embed(self, face: "np.ndarray") -> np.ndarray:
        # very small deterministic embedding for tests/demos
        vec = face.mean(axis=(0,1)) if face is not None else 0.0
        emb = np.array(vec).astype('float32').ravel()
        if emb.ndim == 0:
            emb = np.array([float(emb)])
        # L2 normalize
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm

    def enroll(self, name: str, embedding: np.ndarray):
        self.store[name] = embedding

    def identify(self, embedding: np.ndarray) -> Optional[str]:
        if not self.store:
            return None
        best = None
        best_score = float('inf')
        for name, emb in self.store.items():
            dist = float(np.linalg.norm(emb - embedding))
            if dist < best_score:
                best_score = dist
                best = name
        # Threshold for demo
        if best_score < 0.6:
            return best
        return None
