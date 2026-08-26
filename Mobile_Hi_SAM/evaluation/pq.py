"""
Panoptic Quality, as defined in Kirillov et al., "Panoptic Segmentation" (CVPR 2019).

    PQ = SQ * RQ
    SQ = sum(IoU over true positives) / |TP|
    RQ = |TP| / (|TP| + 0.5*|FP| + 0.5*|FN|)

The previous version used RQ = |TP| / (|pred| + |gt| - |TP|), a Jaccard-style
ratio over instance counts. The two coincide only when the matching is perfect;
otherwise the Jaccard form is systematically lower. Worked example - one merged
prediction against two ground-truth instances (60 px + 40 px):

    matched = 1, sum_iou = 0.6, |pred| = 1, |gt| = 2
    Jaccard RQ = 1 / (1 + 2 - 1) = 0.500  ->  PQ = 0.300
    panoptic RQ = 1 / (1 + 0 + 0.5) = 0.667  ->  PQ = 0.400

At the standard threshold of 0.5 a prediction can match at most one ground-truth
instance, so the greedy assignment below is optimal. Below 0.5 it is not, and
neither is the metric well defined.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    inter = np.logical_and(mask1, mask2).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def match_instances(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedily match predictions to ground truth above ``iou_threshold``.

    Returns (matches, unmatched_pred_idx, unmatched_gt_idx) where each match is
    (pred_idx, gt_idx, iou).
    """
    matches: List[Tuple[int, int, float]] = []
    used_gt = set()

    for p_idx, pred in enumerate(pred_masks):
        best_iou, best_gt = 0.0, None
        for g_idx, gt in enumerate(gt_masks):
            if g_idx in used_gt:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou, best_gt = score, g_idx
        if best_gt is not None and best_iou > iou_threshold:
            used_gt.add(best_gt)
            matches.append((p_idx, best_gt, best_iou))

    matched_pred = {m[0] for m in matches}
    unmatched_pred = [i for i in range(len(pred_masks)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_masks)) if i not in used_gt]
    return matches, unmatched_pred, unmatched_gt


def compute_pq(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Return (PQ, SQ, RQ)."""
    matches, unmatched_pred, unmatched_gt = match_instances(
        pred_masks, gt_masks, iou_threshold
    )

    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    if tp == 0:
        # No true positives: SQ is undefined, RQ is zero, so PQ is zero.
        return 0.0, 0.0, 0.0

    sq = sum(m[2] for m in matches) / tp
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    return sq * rq, sq, rq


def compute_pq_detailed(
    pred_masks: Sequence[np.ndarray],
    gt_masks: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """PQ plus the counts behind it, for error analysis."""
    matches, unmatched_pred, unmatched_gt = match_instances(
        pred_masks, gt_masks, iou_threshold
    )
    tp, fp, fn = len(matches), len(unmatched_pred), len(unmatched_gt)
    sq = (sum(m[2] for m in matches) / tp) if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp or fp or fn) else 0.0
    return {
        "PQ": sq * rq, "SQ": sq, "RQ": rq,
        "TP": float(tp), "FP": float(fp), "FN": float(fn),
        "n_pred": float(len(pred_masks)), "n_gt": float(len(gt_masks)),
    }
