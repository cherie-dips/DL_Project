import numpy as np

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0


def compute_pq(pred_masks, gt_masks, iou_threshold=0.5):
    matched = 0
    sum_iou = 0

    used = set()
    for pm in pred_masks:
        best_iou = 0
        best_idx = None

        for idx, gm in enumerate(gt_masks):
            if idx in used:
                continue
            i = iou(pm, gm)
            if i > best_iou:
                best_iou = i
                best_idx = idx

        if best_iou > iou_threshold:
            matched += 1
            used.add(best_idx)
            sum_iou += best_iou

    SQ = sum_iou / matched if matched > 0 else 0
    RQ = matched / (len(pred_masks) + len(gt_masks) - matched)
    PQ = SQ * RQ

    return PQ, SQ, RQ

