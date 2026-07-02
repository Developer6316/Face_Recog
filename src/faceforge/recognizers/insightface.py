"""InsightFace wrapper placeholder."""
from typing import Optional
import numpy as np

class InsightFaceRecognizer:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def embed(self, face: "np.ndarray") -> np.ndarray:
        # Production code would run an insightface model
        return np.zeros(512, dtype='float32')

    def identify(self, embedding: np.ndarray):
        return None
