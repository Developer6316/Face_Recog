"""CLI entrypoints for FaceForge."""
import argparse
import logging
from .__version__ import __version__
from .detectors.opencv_dnn import OpenCVDNNDetector
from .recognizers.sv_classifier import SimpleVerifier
from .api import FaceForge

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(prog="faceforge")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--webcam", action="store_true", help="Run webcam demo")
    args = parser.parse_args()

    detector = OpenCVDNNDetector(device=args.device)
    recognizer = SimpleVerifier()
    pipeline = FaceForge(detector, recognizer, device=args.device)

    if args.webcam:
        from ..scripts.demo_webcam import run_webcam
        run_webcam(pipeline)

if __name__ == "__main__":
    main()
