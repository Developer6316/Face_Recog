#!/usr/bin/env bash
# Download common models used by FaceForge. This script is safe to edit to add mirrors.
set -e
mkdir -p models
cd models
# Example placeholder URLs - replace with official model download links
# wget -c https://example.com/models/resnet50.caffemodel -O resnet50.caffemodel
# wget -c https://example.com/models/face_recognition.pth -O face_recognition.pth
echo "No models configured. Edit this script with valid download URLs."
