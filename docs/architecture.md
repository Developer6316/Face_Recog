# Architecture overview

This document explains the high-level architecture of FaceForge.

- detectors/: detector implementations (OpenCV DNN, MTCNN, RetinaFace)
- recognizers/: embedding extractors and recognition backends (FaceNet, InsightFace, FAISS)
- utils/: helpers for video, transforms, devices, and metrics
- scripts/: ready-to-run CLI scripts

Design goals: modular, testable, Colab-friendly, and GPU-aware.
