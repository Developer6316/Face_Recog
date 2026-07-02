from faceforge.recognizers.facenet import FaceNetRecognizer
import pytest
import numpy as np

def test_facenet_missing_dependency():
    # If facenet-pytorch not installed, construction should raise
    with pytest.raises(RuntimeError):
        FaceNetRecognizer()
