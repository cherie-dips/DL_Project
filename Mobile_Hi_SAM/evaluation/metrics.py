"""
Segmentation metrics for Mobile-Hi-SAM.

The previous implementation reported recall under the name ``fgIOU``: it clipped
the prediction to the GT foreground before taking the union, so the union could
never exceed the GT area. In the 50-epoch run this is visible directly - fgIOU
and R agreed to eight decimal places.

Names here mean what they say:
  * ``recall`` / ``precision`` / ``f_score`` - pixel-level, over binary masks
  * ``iou``                                  - true intersection over union
  * ``fg_iou``                               - pixel-level IoU over the union of
                                               all foreground, Hi-SAM's shared
                                               pixel-level quantity (see NOTE)
  * ``panoptic_quality``                     - instance-matched, via evaluation/pq.py

Hi-SAM parity, VERIFIED against ymy-k/Hi-SAM (eval_img.py):

  * fgIOU is a single STROKE-level number, not a per-hierarchy-level one. The
    published table repeats it unchanged on the Word and Text-line rows (both
    74.86) because it is literally the same measurement. A per-level "fgIOU" is
    not comparable to that column no matter how correctly it is computed.
  * It is aggregated over the whole split as sum(I)/sum(U), not as a mean of
    per-image IoUs. Use DatasetIoUAccumulator, not MetricAccumulator, for it.
  * F-score, by contrast, IS per-image: precision and recall are computed per
    image and then averaged.
  * Reproducing fgIOU requires the S-Decoder plus Hi-SAM's contributed
    stroke-level annotations for HierText, which are a separate download.
"""

from typing import Dict, List, Sequence

import numpy as np
import torch

from .pq import compute_pq

try:
    from scipy.ndimage import label as _cc_label
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ----------------------------------------------------------------------
# Pixel-level
# ----------------------------------------------------------------------
def binarize(logits: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Threshold logits. SAM's mask_threshold is 0.0 in logit space."""
    return (logits > threshold).float()


def pixel_stats(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """Precision / recall / F / IoU over a binary prediction and target."""
    pred = pred.flatten()
    gt = gt.flatten()
    tp = float((pred * gt).sum())
    fp = float((pred * (1 - gt)).sum())
    fn = float(((1 - pred) * gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    union = tp + fp + fn
    iou = tp / union if union > 0 else (1.0 if tp == 0 and union == 0 else 0.0)

    return {"P": precision, "R": recall, "F": f, "IoU": iou}


def iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """True IoU: intersection / (|pred| + |gt| - intersection)."""
    pred = pred.flatten()
    gt = gt.flatten()
    inter = float((pred * gt).sum())
    union = float(pred.sum()) + float(gt.sum()) - inter
    if union <= 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def recall(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """What the old ``compute_fg_iou`` actually computed. Kept, correctly named."""
    gt_sum = float(gt.sum())
    if gt_sum == 0:
        return 1.0 if float(pred.sum()) == 0 else 0.0
    return float((pred * gt).sum()) / gt_sum


# ----------------------------------------------------------------------
# Instance-level
# ----------------------------------------------------------------------
def to_instances(binary: torch.Tensor, min_area: int = 4) -> List[np.ndarray]:
    """Split a binary mask into connected components."""
    arr = binary.detach().cpu().numpy().astype(bool)
    if arr.ndim > 2:
        arr = arr.squeeze()
    if not _HAS_SCIPY:
        return [arr] if arr.any() else []
    labelled, n = _cc_label(arr)
    out = []
    for i in range(1, n + 1):
        component = labelled == i
        if component.sum() >= min_area:
            out.append(component)
    return out


def panoptic_quality(
    pred_instances: Sequence[np.ndarray],
    gt_instances: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Instance-matched PQ. Degenerate cases are defined explicitly."""
    if not pred_instances and not gt_instances:
        return {"PQ": 1.0, "SQ": 1.0, "RQ": 1.0}
    if not pred_instances or not gt_instances:
        return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0}
    pq, sq, rq = compute_pq(list(pred_instances), list(gt_instances), iou_threshold)
    return {"PQ": pq, "SQ": sq, "RQ": rq}


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
class DatasetIoUAccumulator:
    """fgIOU the way Hi-SAM computes it: sum(intersection) / sum(union) over the
    whole split, not the mean of per-image IoUs.

    The two are different statistics - a dataset-aggregated ratio is dominated by
    large-area images, a per-image mean is not - so a per-image mean cannot be
    compared against Hi-SAM's published fgIOU column even when both are correctly
    computed IoUs.

    Note also that Hi-SAM's fgIOU is a single STROKE-level number, reported
    unchanged on the Word and Text-line rows of its table (both 74.86). It is not
    a per-hierarchy-level metric, and reproducing it requires the S-Decoder plus
    Hi-SAM's contributed stroke annotations.
    """

    def __init__(self):
        self.intersection = 0.0
        self.union = 0.0

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        pred = pred.flatten()
        gt = gt.flatten()
        inter = float((pred * gt).sum())
        self.intersection += inter
        self.union += float(pred.sum()) + float(gt.sum()) - inter

    def value(self, as_percent: bool = True) -> float:
        if self.union <= 0:
            return 0.0
        return self.intersection / self.union * (100.0 if as_percent else 1.0)


class MetricAccumulator:
    """Collects per-sample metrics and reports the mean as a percentage."""

    def __init__(self, keys: Sequence[str]):
        self.keys = list(keys)
        self.values: Dict[str, List[float]] = {k: [] for k in self.keys}

    def update(self, sample: Dict[str, float]):
        for k in self.keys:
            if k in sample:
                self.values[k].append(float(sample[k]))

    def mean(self, as_percent: bool = True) -> Dict[str, float]:
        scale = 100.0 if as_percent else 1.0
        return {
            k: (float(np.mean(v)) * scale if v else 0.0)
            for k, v in self.values.items()
        }

    def __len__(self) -> int:
        return len(self.values[self.keys[0]]) if self.keys else 0
