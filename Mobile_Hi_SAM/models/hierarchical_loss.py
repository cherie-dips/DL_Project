"""
Hierarchical loss for Mobile-Hi-SAM.

Each level is supervised at the resolution its decoder branch actually produces,
against ground truth rasterised there. Nothing is upsampled or downsampled
before the comparison.

Tversky (C2) and the containment penalty (C3) are implemented but default to off
so that turning them on is a config change, not a code change - the plan calls
for them to be evaluated one at a time against the validation curve.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Dice over logits. Symmetric in false positives and false negatives."""
    pred = torch.sigmoid(pred).flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return 1 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Dice generalised so false negatives can be punished harder than false
    positives. beta > alpha targets a recall-bound failure mode."""
    pred = torch.sigmoid(pred).flatten(1)
    target = target.flatten(1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    return 1 - ((tp + smooth) / (tp + alpha * fp + beta * fn + smooth)).mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss. Requires binary targets - p_t stops being a probability if the
    target is fractional, which is why masks are rasterised rather than resized."""
    pred = pred.flatten(1)
    target = target.flatten(1)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p = torch.sigmoid(pred)
    p_t = p * target + (1 - p) * (1 - target)
    loss = bce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return loss.mean()


def iou_prediction_loss(
    pred_iou: torch.Tensor,
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    """MSE between the predicted quality score and the mask's realised IoU."""
    with torch.no_grad():
        binary = (torch.sigmoid(pred_masks) > 0.5).float().flatten(2)
        target = target_masks.flatten(2)
        inter = (binary * target).sum(dim=2)
        union = binary.sum(dim=2) + target.sum(dim=2) - inter
        actual = inter / (union + 1e-6)
    return F.mse_loss(pred_iou, actual)


def containment_penalty(
    word: torch.Tensor,
    line: torch.Tensor,
    para: torch.Tensor,
) -> torch.Tensor:
    """Penalise probability mass that escapes its parent.

    word, line and para must be at the same resolution. Zero parameters; this is
    what makes a 'hierarchical' loss actually hierarchical.
    """
    w, l, p = torch.sigmoid(word), torch.sigmoid(line), torch.sigmoid(para)
    return F.relu(w - l).mean() + F.relu(l - p).mean()


class HierarchicalLoss(nn.Module):
    """Weighted word/line/paragraph loss over the H-Decoder's outputs."""

    def __init__(
        self,
        weight_word: float = 1.0,
        weight_line: float = 1.0,
        weight_para: float = 1.0,
        weight_dice: float = 1.0,
        weight_focal: float = 20.0,
        weight_iou: float = 1.0,
        weight_containment: float = 0.0,   # C3: off by default
        use_tversky: bool = False,         # C2: off by default
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        supervise_word_hr: bool = True,
        weight_pixel: float = 1.0,
    ):
        super().__init__()
        self.weight_word = weight_word
        self.weight_line = weight_line
        self.weight_para = weight_para
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_iou = weight_iou
        self.weight_containment = weight_containment
        self.use_tversky = use_tversky
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.supervise_word_hr = supervise_word_hr
        self.weight_pixel = weight_pixel

    def _region_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_tversky:
            return tversky_loss(pred, target, self.tversky_alpha, self.tversky_beta)
        return dice_loss(pred, target)

    def _level(self, pred, target) -> Tuple[torch.Tensor, Dict[str, float]]:
        region = self._region_loss(pred, target)
        focal = focal_loss(pred, target, self.focal_alpha, self.focal_gamma)
        total = self.weight_dice * region + self.weight_focal * focal
        return total, {"region": region.item(), "focal": focal.item()}

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: from ``MobileHiSAM.forward_hierarchical``.
            batch:   collated batch carrying the gt_* masks.
        """
        logs: Dict[str, float] = {}

        # Word is supervised on the refined 384^2 branch when available, which is
        # the branch Hi-SAM added specifically for small text.
        if self.supervise_word_hr and "word_hr" in outputs:
            word_pred, word_gt = outputs["word_hr"], batch["gt_word_mask"]
        else:
            word_pred, word_gt = outputs["word"], batch["gt_word_mask_lr"]

        loss_word, l_word = self._level(word_pred, word_gt)
        loss_line, l_line = self._level(outputs["line"], batch["gt_line_mask"])
        loss_para, l_para = self._level(outputs["para"], batch["gt_para_mask"])

        for name, parts in (("word", l_word), ("line", l_line), ("para", l_para)):
            for k, v in parts.items():
                logs[f"{name}_{k}"] = v

        total = (
            self.weight_word * loss_word
            + self.weight_line * loss_line
            + self.weight_para * loss_para
        )

        # The IoU head predicts one score per level; supervise it against the
        # realised IoU at the shared 256^2 resolution so the three line up.
        if self.weight_iou > 0 and "iou" in outputs:
            stacked_pred = torch.cat(
                [outputs["word"], outputs["line"], outputs["para"]], dim=1
            )
            stacked_gt = torch.cat(
                [batch["gt_word_mask_lr"], batch["gt_line_mask"], batch["gt_para_mask"]],
                dim=1,
            )
            loss_iou = iou_prediction_loss(outputs["iou"], stacked_pred, stacked_gt)
            total = total + self.weight_iou * loss_iou
            logs["iou"] = loss_iou.item()

        # S-Decoder pixel-level branch: whole-image text foreground, prompted by
        # the ModalAligner. This is what puts the aligner in the gradient path.
        if self.weight_pixel > 0 and "pixel_hr" in outputs and "gt_text_mask" in batch:
            loss_hr, _ = self._level(outputs["pixel_hr"], batch["gt_text_mask"])
            loss_pixel = loss_hr
            logs["pixel_hr"] = loss_hr.item()
            if "pixel" in outputs and "gt_text_mask_lr" in batch:
                loss_lr, _ = self._level(outputs["pixel"], batch["gt_text_mask_lr"])
                loss_pixel = loss_pixel + loss_lr
                logs["pixel_lr"] = loss_lr.item()
            # Quality heads for both S-Decoder branches, same treatment as the
            # H-Decoder's: predict the mask's realised IoU.
            if self.weight_iou > 0:
                if "pixel_iou" in outputs:
                    l = iou_prediction_loss(
                        outputs["pixel_iou"], outputs["pixel_hr"], batch["gt_text_mask"]
                    )
                    loss_pixel = loss_pixel + self.weight_iou * l
                    logs["pixel_iou"] = l.item()
                if "pixel_iou_lr" in outputs and "pixel" in outputs:
                    l = iou_prediction_loss(
                        outputs["pixel_iou_lr"], outputs["pixel"], batch["gt_text_mask_lr"]
                    )
                    loss_pixel = loss_pixel + self.weight_iou * l
                    logs["pixel_iou_lr"] = l.item()

            total = total + self.weight_pixel * loss_pixel
            logs["pixel"] = loss_pixel.item()

        if self.weight_containment > 0:
            pen = containment_penalty(
                outputs["word"], outputs["line"], outputs["para"]
            )
            total = total + self.weight_containment * pen
            logs["containment"] = pen.item()

        logs["total"] = total.item()
        logs["word"] = loss_word.item()
        logs["line"] = loss_line.item()
        logs["para"] = loss_para.item()
        return total, logs
