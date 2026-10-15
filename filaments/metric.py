"""Panoptic Quality as the organisers compute it: every annotator record is a separate ground truth.

Within one record, a (ground truth, prediction) pair with mask IoU > 0.5 is a true positive; unmatched predictions are
false positives and unmatched ground-truth filaments false negatives. The counts are pooled over all records:
PQ = sum of matched IoUs / (TP + FP / 2 + FN / 2).
"""
import numpy as np
from pycocotools import mask as cocomask


def image_counts(image_records, predictions):
    """[sum of matched IoU, TP, FP, FN] of one image's predicted RLEs against each of its annotator records."""
    out = np.zeros(4)
    for record in image_records:
        truth = record["rles"]
        if not truth or not predictions:
            out += [0, 0, len(predictions), len(truth)]
            continue
        iou = cocomask.iou(truth, predictions, [0] * len(predictions))
        hit = iou > 0.5
        out += [iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return out


def panoptic_quality(counts):
    iou_sum, tp, fp, fn = map(float, counts)
    denominator = tp + (fp + fn) / 2
    return {"PQ": iou_sum / denominator if denominator else 0.0, "SQ": iou_sum / tp if tp else 0.0,
            "RQ": tp / denominator if denominator else 0.0, "TP": int(tp), "FP": int(fp), "FN": int(fn)}
