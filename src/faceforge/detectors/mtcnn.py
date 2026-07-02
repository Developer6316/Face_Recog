"""Lightweight MTCNN wrapper using facenet-pytorch."""
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None

class MTCNNDetector:
    def __init__(self, device: str = "cpu"):
        if MTCNN is None:
            raise RuntimeError("facenet-pytorch is required for MTCNNDetector")
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, image: "np.ndarray") -> List[Dict]:
        boxes, probs = self.mtcnn.detect(image)
        results = []
        if boxes is None:
            return results
        for box, p in zip(boxes, probs):
            x1, y1, x2, y2 = map(int, box.tolist())
            results.append({"box": [x1, y1, x2, y2], "score": float(p)})
        return results
