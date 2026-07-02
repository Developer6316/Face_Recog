# Minimal tests for detectors/recognizers
import numpy as np
from faceforge.detectors.opencv_dnn import OpenCVDNNDetector
from faceforge.recognizers.sv_classifier import SimpleVerifier

def test_simple_detector_init():
    det = OpenCVDNNDetector()
    assert det is not None

def test_simple_recognizer_embed_identify():
    recog = SimpleVerifier()
    fake_face = np.zeros((160,160,3), dtype='uint8')
    emb = recog.embed(fake_face)
    assert emb is not None
    recog.enroll('alice', emb)
    identity = recog.identify(emb)
    assert identity == 'alice'
