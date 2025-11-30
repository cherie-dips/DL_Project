import torch
import numpy as np

def sigmoid_to_binary(x, thr=0.5):
    return (torch.sigmoid(x) > thr).float()

def compute_iou_batch(preds_logits, masks, thr=0.5):
    preds = sigmoid_to_binary(preds_logits, thr)
    masks = (masks > 0.5).float()
    inter = (preds * masks).sum(dim=[1,2,3])
    union = (preds + masks - preds*masks).sum(dim=[1,2,3]) + 1e-6
    iou = (inter / union).cpu().numpy()
    return float(np.nanmean(iou))

def compute_dice_batch(preds_logits, masks, thr=0.5):
    preds = sigmoid_to_binary(preds_logits, thr)
    masks = (masks > 0.5).float()
    inter = (preds * masks).sum(dim=[1,2,3])
    denom = preds.sum(dim=[1,2,3]) + masks.sum(dim=[1,2,3]) + 1e-6
    dice = (2 * inter / denom).cpu().numpy()
    return float(np.nanmean(dice))

def compute_pixel_acc(preds_logits, masks, thr=0.5):
    preds = sigmoid_to_binary(preds_logits, thr)
    masks = (masks > 0.5).float()
    correct = (preds == masks).float().sum().item()
    total = torch.numel(preds)
    return correct / total

def compute_simple_pq(preds_logits, masks, thr=0.5):
    # simplified single-mask PQ: for each sample, compute IoU; if IoU>0 -> SQ=IoU, TP = 1 if IoU>0.5 else 0
    preds = sigmoid_to_binary(preds_logits, thr).cpu().numpy()
    masks = (masks > 0.5).cpu().numpy()

    pq_scores = []
    N = preds.shape[0]
    for i in range(N):
        p = preds[i,0]
        g = masks[i,0]
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union == 0:
            continue
        iou = inter / (union + 1e-6)
        sq = iou
        tp = 1 if iou > 0.5 else 0
        fp = 0 if tp else 1
        fn = 0 if tp else 1
        rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
        pq = sq * rq
        pq_scores.append(pq)
    if len(pq_scores) == 0:
        return 0.0
    return float(np.mean(pq_scores))

