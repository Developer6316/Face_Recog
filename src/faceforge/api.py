"""
High-level pipeline: detection -> alignment -> embedding -> recognition
"""
from typing import Optional, List, Tuple
import numpy as np
import logging

from .detectors import Detector
from .recognizers import Recognizer
from .utils.transforms import align_face

logger = logging.getLogger(__name__)

class FaceForge:
    def __init__(self, detector: Detector, recognizer: Recognizer, device: Optional[str] = None):
        """Create a FaceForge pipeline.

        Args:
            detector: detector instance implementing detect(image) -> list of boxes
            recognizer: recognizer instance implementing embed(face) and identify(embedding)
            device: optional device string like 'cpu' or 'cuda'
        """
        self.detector = detector
        self.recognizer = recognizer
        self.device = device

    def process_image(self, image: "np.ndarray") -> List[dict]:
        """Run full pipeline on a single image.

        Returns a list of dicts with keys: box, score, embedding, identity (optional).
        """
        detections = self.detector.detect(image)
        results = []
        for det in detections:
            box = det.get("box")
            score = det.get("score")
            x1, y1, x2, y2 = box
            face = image[y1:y2, x1:x2]
            face_aligned = align_face(face)
            embedding = self.recognizer.embed(face_aligned)
            identity = self.recognizer.identify(embedding)
            results.append({"box": box, "score": score, "embedding": embedding, "identity": identity})
        return results
