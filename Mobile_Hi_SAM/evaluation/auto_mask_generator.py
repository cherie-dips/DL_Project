"""
Hi-SAM's prompt-free inference, adapted to MobileHiSAM.

Their protocol is NOT a prompt grid. It is:

  1. ModalAligner -> S-Decoder -> predicted text foreground mask;
  2. sample up to ``fg_points_num`` (600) points UNIFORMLY AT RANDOM FROM THAT
     FOREGROUND - so every prompt lands on predicted text;
  3. run the H-Decoder on those points in batches of ``batch_points_num`` (100);
  4. drop predictions whose line score is below ``score_thresh`` (0.5);
  5. Matrix NMS (SOLOv2) on the line masks, keep ``updated_score > nms_thresh``
     (0.5);
  6. group surviving lines into paragraphs by pairwise IoU of their predicted
     paragraph masks.

A grid instead puts most prompts on background, where the decoder still emits a
mask, which is why grid-prompted numbers understate the model badly. Reproducing
their numbers requires reproducing this.

``matrix_nms`` and ``get_para_iou`` are copied verbatim from
hi_sam/modeling/auto_mask_generator.py.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch


# ----------------------------------------------------------------------
# Verbatim from Hi-SAM's auto_mask_generator.py
# ----------------------------------------------------------------------
def matrix_nms(seg_masks, scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS from SOLOv2

    Args:
        seg_masks (Tensor): shape (n, h, w)
        scores (Tensor): shape (n)
        kernel (str): 'linear' or 'gaussian'
        sigma (float): std in gaussian method
        sum_masks (Tensor): the sum of seg_masks
    """
    n_samples = len(seg_masks)
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    del seg_masks
    # union
    sum_masks = sum_masks.expand(n_samples, n_samples)
    # iou
    iou_matrix = (inter_matrix / (sum_masks + sum_masks.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # IOU compensation
    compensate_iou, _ = iou_matrix.max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)
    # IOU decay
    decay_iou = iou_matrix  # no label matrix because there is only one foreground class

    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coef, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coef, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError
    updated_score = scores * decay_coef
    return updated_score


def get_para_iou(para_masks):
    """
        Args:
            para_masks (Tensor): shape (n, h, w)
        """
    n_samples = len(para_masks)
    sum_masks = para_masks.sum((1, 2)).float()
    para_masks = para_masks.reshape(n_samples, -1).float()
    inter_matrix = torch.mm(para_masks, para_masks.transpose(1, 0))
    # del para_masks
    sum_masks = sum_masks.expand(n_samples, n_samples)
    iou_matrix = (inter_matrix / (sum_masks + sum_masks.transpose(1, 0) - inter_matrix))

    return iou_matrix


# ----------------------------------------------------------------------
def group_by_affinity(affinity: torch.Tensor, threshold: float = 0.5) -> List[List[int]]:
    """Union-find over the paragraph-IoU affinity matrix."""
    n = affinity.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    aff = affinity.detach().cpu().numpy()
    for i in range(n):
        for j in range(i + 1, n):
            if aff[i, j] > threshold:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


class MobileHiSAMAutoMaskGenerator:
    """Prompt-free hierarchical inference following Hi-SAM's protocol."""

    def __init__(self, model):
        if model.hi_decoder is None:
            raise RuntimeError("hierarchical decoding required")
        if not model.enable_s_decoder:
            raise RuntimeError(
                "Prompt-free inference needs the S-Decoder: prompts are sampled "
                "from its predicted text foreground. Train with the S-Decoder "
                "enabled, or evaluate with ground-truth prompts instead."
            )
        self.model = model
        self.features = None
        self.input_size = None
        self.original_size = None

    @torch.no_grad()
    def set_image(self, image: torch.Tensor, input_size, original_size):
        """image: (1, 3, H, W) in [0, 1], already resized and padded."""
        self.features = self.model.encode(image)
        self.input_size = input_size
        self.original_size = original_size

    @torch.no_grad()
    def forward_foreground_points(self, from_low_res: bool = False,
                                  fg_points_num: int = 600) -> torch.Tensor:
        sparse_emb = self.model.modal_aligner(self.features)
        low_res_mask, high_res_mask, _, _ = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            multimask_output=False,
        )
        if from_low_res:
            fg_mask = (low_res_mask > self.model.mask_threshold).squeeze(1)[0]
        else:
            fg_mask = (high_res_mask > self.model.mask_threshold).squeeze(1)[0]

        y_idx, x_idx = torch.where(fg_mask > 0)
        p_n = x_idx.size(0)
        if p_n == 0:
            return torch.zeros((0, 1, 2), device=self.features.device)
        perm = torch.randperm(p_n, device=x_idx.device)
        idx = perm[: min(p_n, fg_points_num)]
        y_idx, x_idx = y_idx[idx][:, None], x_idx[idx][:, None]
        fg_points = torch.cat((x_idx, y_idx), dim=1)[:, None, :].float()
        if from_low_res:
            fg_points = fg_points * 4      # 256 -> 1024
        return fg_points

    @torch.no_grad()
    def forward_hi_decoder(self, point_coords, point_labels):
        point_embeddings, _ = self.model.prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None
        )
        return self.model.hi_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=point_embeddings,
            multimask_output=True,
        )

    @torch.no_grad()
    def predict(self, from_low_res: bool = False, fg_points_num: int = 600,
                batch_points_num: int = 100, score_thresh: float = 0.5,
                nms_thresh: float = 0.5, para_thresh: float = 0.5):
        """Returns (word_masks, line_masks, para_groups) or (None, None, None).

        word/line masks are boolean at their decoder resolutions; para_groups
        lists, per paragraph, the indices of the lines belonging to it.
        """
        fg_points = self.forward_foreground_points(from_low_res, fg_points_num)
        n_points = fg_points.shape[0]
        if n_points == 0:
            return None, None, None

        masks, scores, word_masks = [], [], []
        for start in range(0, n_points, batch_points_num):
            end = min(start + batch_points_num, n_points)
            hi_masks, hi_iou, word_logits = self.forward_hi_decoder(
                fg_points[start:end],
                torch.ones((end - start, 1), device=fg_points.device),
            )
            # multimask_output=True already sliced [1:], leaving word/line/para.
            masks.append(hi_masks)
            scores.append(hi_iou)
            word_masks.append(word_logits)

        masks = torch.cat(masks, dim=0)              # (N, 3, 256, 256)
        scores = torch.cat(scores, dim=0)            # (N, 3)
        word_masks = torch.cat(word_masks, dim=0)    # (N, 1, 384, 384)

        keep = scores[:, 1] > score_thresh           # index 1 is the line token
        if keep.sum() == 0:
            return None, None, None
        masks, scores, word_masks = masks[keep], scores[keep], word_masks[keep]

        # Matrix NMS on the LINE masks (index -2 of word/line/para).
        updated = matrix_nms(
            seg_masks=(masks[:, -2, :, :] > self.model.mask_threshold),
            scores=scores[:, 1],
        )
        keep = updated > nms_thresh
        if keep.sum() == 0:
            return None, None, None
        masks, word_masks = masks[keep], word_masks[keep]

        affinity = get_para_iou(
            para_masks=(masks[:, -1, :, :] > self.model.mask_threshold)
        )
        groups = group_by_affinity(affinity, para_thresh)

        line_out = (masks[:, -2, :, :] > self.model.mask_threshold)
        word_out = (word_masks[:, 0, :, :] > self.model.mask_threshold)
        return word_out, line_out, groups
