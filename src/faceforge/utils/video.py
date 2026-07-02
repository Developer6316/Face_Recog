"""Video and webcam helpers."""
from typing import Callable
import cv2
import threading
import time
import numpy as np

class WebcamStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()

def draw_detections(frame: "np.ndarray", detections: list, color=(0,255,0)):
    for det in detections:
        x1,y1,x2,y2 = det.get('box')
        label = det.get('identity') or ''
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
