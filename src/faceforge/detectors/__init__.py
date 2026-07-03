"""
Face detectors module - provides base Detector class and detector implementations
"""
from .opencv_dnn import Detector, OpenCVDNNDetector

__all__ = ["Detector", "OpenCVDNNDetector"]
