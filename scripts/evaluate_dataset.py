#!/usr/bin/env python3
"""Evaluate a dataset using the metrics module. This is a small driver for demo purposes."""
import argparse
import json
from faceforge.utils.metrics import compute_roc

parser = argparse.ArgumentParser()
parser.add_argument('--preds', required=True, help='JSONL with records {label:int, score:float}')
args = parser.parse_args()

labels = []
scores = []
with open(args.preds) as f:
    for line in f:
        rec = json.loads(line)
        labels.append(rec['label'])
        scores.append(rec['score'])

fpr, tpr, auc_score = compute_roc(labels, scores)
print('AUC:', auc_score)
