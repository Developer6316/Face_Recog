"""Evaluation metrics for recognition/detection."""
from typing import List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def compute_roc(labels: List[int], scores: List[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, thresh = roc_curve(labels, scores)
    return fpr, tpr, auc(fpr, tpr)

def precision_recall(labels: List[int], scores: List[float]):
    p, r, t = precision_recall_curve(labels, scores)
    return p, r
