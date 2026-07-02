from faceforge.utils.metrics import compute_roc

def test_metrics_roc():
    labels = [0,1,1,0,1]
    scores = [0.1,0.9,0.8,0.2,0.7]
    fpr, tpr, auc_score = compute_roc(labels, scores)
    assert auc_score > 0.5
