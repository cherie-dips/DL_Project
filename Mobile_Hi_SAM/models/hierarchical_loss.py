"""
Hierarchical Loss Functions for Mobile-Hi-SAM
Implements weighted multi-level loss for paragraph, line, and word segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """
    Compute Dice loss for binary segmentation.
    
    Args:
        pred: (B, N, H, W) - Predicted masks (logits)
        target: (B, N, H, W) - Ground truth masks (binary)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Scalar Dice loss
    """
    pred = torch.sigmoid(pred)
    pred = pred.flatten(1)
    target = target.flatten(1)
    
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Compute Focal loss for handling class imbalance.
    
    Args:
        pred: (B, N, H, W) - Predicted masks (logits)
        target: (B, N, H, W) - Ground truth masks (binary)
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
        
    Returns:
        Scalar Focal loss
    """
    pred = pred.flatten(1)
    target = target.flatten(1)
    
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pred_prob = torch.sigmoid(pred)
    p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
    
    loss = bce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    
    return loss.mean()


def iou_loss(
    pred_iou: torch.Tensor,
    pred_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Compute IoU prediction loss (MSE between predicted and actual IoU).
    
    Args:
        pred_iou: (B, N) - Predicted IoU scores
        pred_masks: (B, N, H, W) - Predicted masks (logits)
        target_masks: (B, N, H, W) - Ground truth masks
        
    Returns:
        Scalar MSE loss for IoU prediction
    """
    with torch.no_grad():
        # Compute actual IoU
        pred_masks_binary = (torch.sigmoid(pred_masks) > 0.5).float()
        pred_masks_binary = pred_masks_binary.flatten(2)
        target_masks_flat = target_masks.flatten(2)
        
        intersection = (pred_masks_binary * target_masks_flat).sum(dim=2)
        union = pred_masks_binary.sum(dim=2) + target_masks_flat.sum(dim=2) - intersection
        actual_iou = intersection / (union + 1e-6)
    
    # MSE loss between predicted and actual IoU
    loss = F.mse_loss(pred_iou, actual_iou)
    return loss


class HierarchicalLoss(nn.Module):
    """
    Multi-level hierarchical loss for paragraph, line, and word segmentation.
    
    Total Loss = α * L_para + β * L_line + γ * L_word
    
    Where each level loss is:
        L_level = λ_dice * Dice + λ_focal * Focal + λ_iou * IoU_MSE
    """
    
    def __init__(
        self,
        # Hierarchy level weights
        weight_para: float = 1.0,
        weight_line: float = 1.0,
        weight_word: float = 1.0,
        # Loss component weights
        weight_dice: float = 1.0,
        weight_focal: float = 20.0,
        weight_iou: float = 1.0,
        # Focal loss params
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        
        self.weight_para = weight_para
        self.weight_line = weight_line
        self.weight_word = weight_word
        
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_iou = weight_iou
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(
        self,
        # Paragraph predictions
        pred_para_masks: torch.Tensor,
        pred_para_iou: torch.Tensor,
        target_para_masks: torch.Tensor,
        # Line predictions
        pred_line_masks: torch.Tensor,
        pred_line_iou: torch.Tensor,
        target_line_masks: torch.Tensor,
        # Word predictions
        pred_word_masks: torch.Tensor,
        pred_word_iou: torch.Tensor,
        target_word_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute hierarchical loss.
        
        Args:
            pred_*_masks: (B, N, H, W) - Predicted masks (logits)
            pred_*_iou: (B, N) - Predicted IoU scores
            target_*_masks: (B, N, H, W) - Ground truth masks
            
        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary with individual loss components
        """
        
        # Paragraph loss
        loss_para_dice = dice_loss(pred_para_masks, target_para_masks)
        loss_para_focal = focal_loss(
            pred_para_masks, target_para_masks,
            alpha=self.focal_alpha, gamma=self.focal_gamma
        )
        loss_para_iou = iou_loss(pred_para_iou, pred_para_masks, target_para_masks)
        loss_para = (
            self.weight_dice * loss_para_dice +
            self.weight_focal * loss_para_focal +
            self.weight_iou * loss_para_iou
        )
        
        # Line loss
        loss_line_dice = dice_loss(pred_line_masks, target_line_masks)
        loss_line_focal = focal_loss(
            pred_line_masks, target_line_masks,
            alpha=self.focal_alpha, gamma=self.focal_gamma
        )
        loss_line_iou = iou_loss(pred_line_iou, pred_line_masks, target_line_masks)
        loss_line = (
            self.weight_dice * loss_line_dice +
            self.weight_focal * loss_line_focal +
            self.weight_iou * loss_line_iou
        )
        
        # Word loss
        loss_word_dice = dice_loss(pred_word_masks, target_word_masks)
        loss_word_focal = focal_loss(
            pred_word_masks, target_word_masks,
            alpha=self.focal_alpha, gamma=self.focal_gamma
        )
        loss_word_iou = iou_loss(pred_word_iou, pred_word_masks, target_word_masks)
        loss_word = (
            self.weight_dice * loss_word_dice +
            self.weight_focal * loss_word_focal +
            self.weight_iou * loss_word_iou
        )
        
        # Total hierarchical loss
        total_loss = (
            self.weight_para * loss_para +
            self.weight_line * loss_line +
            self.weight_word * loss_word
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'para_total': loss_para.item(),
            'para_dice': loss_para_dice.item(),
            'para_focal': loss_para_focal.item(),
            'para_iou': loss_para_iou.item(),
            'line_total': loss_line.item(),
            'line_dice': loss_line_dice.item(),
            'line_focal': loss_line_focal.item(),
            'line_iou': loss_line_iou.item(),
            'word_total': loss_word.item(),
            'word_dice': loss_word_dice.item(),
            'word_focal': loss_word_focal.item(),
            'word_iou': loss_word_iou.item(),
        }
        
        return total_loss, loss_dict


class SimplifiedHierarchicalLoss(nn.Module):
    """
    Simplified version using only Dice loss (faster training, good baseline).
    """
    
    def __init__(
        self,
        weight_para: float = 1.0,
        weight_line: float = 1.0,
        weight_word: float = 1.0,
    ):
        super().__init__()
        self.weight_para = weight_para
        self.weight_line = weight_line
        self.weight_word = weight_word
    
    def forward(
        self,
        pred_para_masks: torch.Tensor,
        target_para_masks: torch.Tensor,
        pred_line_masks: torch.Tensor,
        target_line_masks: torch.Tensor,
        pred_word_masks: torch.Tensor,
        target_word_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute simplified hierarchical loss (Dice only)"""
        
        loss_para = dice_loss(pred_para_masks, target_para_masks)
        loss_line = dice_loss(pred_line_masks, target_line_masks)
        loss_word = dice_loss(pred_word_masks, target_word_masks)
        
        total_loss = (
            self.weight_para * loss_para +
            self.weight_line * loss_line +
            self.weight_word * loss_word
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'para': loss_para.item(),
            'line': loss_line.item(),
            'word': loss_word.item(),
        }
        
        return total_loss, loss_dict
