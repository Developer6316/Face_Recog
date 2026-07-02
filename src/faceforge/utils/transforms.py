"""Image transforms and alignment helpers."""
from typing import Tuple
import numpy as np
import cv2

def align_face(face: "np.ndarray") -> "np.ndarray":
    # Placeholder: a real aligner uses landmarks. Here we simply resize.
    if face is None or face.size == 0:
        return face
    aligned = cv2.resize(face, (160, 160))
    return aligned
