#!/usr/bin/env python3
"""Batch process a video and save detections per-frame to a simple JSONL."""
import cv2
import json
import argparse
from faceforge.detectors.opencv_dnn import OpenCVDNNDetector
from faceforge.recognizers.sv_classifier import SimpleVerifier
from faceforge.api import FaceForge

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('--output', default='detections.jsonl')
args = parser.parse_args()

cap = cv2.VideoCapture(args.input)
detector = OpenCVDNNDetector()
recognizer = SimpleVerifier()
pipeline = FaceForge(detector, recognizer)

out = open(args.output, 'w')
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    detections = pipeline.process_image(frame)
    out.write(json.dumps({"frame": frame_idx, "detections": detections}) + "\n")
    frame_idx += 1
out.close()
