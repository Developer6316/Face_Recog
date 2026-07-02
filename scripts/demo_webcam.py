#!/usr/bin/env python3
"""Run a threaded webcam demo using a FaceForge pipeline."""
import cv2
import time
import argparse
import logging

from faceforge.utils.video import WebcamStream, draw_detections

logger = logging.getLogger(__name__)

def run_webcam(pipeline):
    ws = WebcamStream().start()
    time.sleep(0.5)

    try:
        while True:
            frame = ws.read()
            if frame is None:
                continue
            results = pipeline.process_image(frame)
            out = draw_detections(frame, results)
            cv2.imshow('FaceForge', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        ws.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    # Minimal local demo: use simple detector/recognizer
    from faceforge.detectors.opencv_dnn import OpenCVDNNDetector
    from faceforge.recognizers.sv_classifier import SimpleVerifier
    from faceforge.api import FaceForge

    detector = OpenCVDNNDetector()
    recognizer = SimpleVerifier()
    pipeline = FaceForge(detector, recognizer)
    run_webcam(pipeline)
