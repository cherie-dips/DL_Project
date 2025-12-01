#!/usr/bin/env python3
"""
Evaluate Mobile-HiSAM on HierText validation set.
- Uses SAFE JSONL parsing (line-by-line)
- Avoids JSONDecodeError
- Computes IoU and Dice between predicted masks and GT masks
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
import sys

PROJECT_ROOT = "/scratch/hpc/visitor/px151.visitor/DL/DL_Project/MOBILE-HI-SAM"
sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM


# ============================================================
# JSONL loader (same as training)
# ============================================================
def load_jsonl(path, max_items=None):
    items = []
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                items.append(json.loads(line))
                count += 1
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed JSON in {path}")
                continue

            if max_items and count >= max_items:
                break

    return items


# ============================================================
# Util: Convert vertices → binary mask
# ============================================================
def polygon_to_mask(size, vertices):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    try:
        if isinstance(vertices[0], list):
            pts = [(v[0], v[1]) for v in vertices]
        else:
            pts = [(vertices[i], vertices[i+1]) for i in range(0, len(vertices), 2)]

        if len(pts) >= 3:
            draw.polygon(pts, fill=1)

    except Exception:
        pass

    return torch.tensor(np.array(mask), dtype=torch.float32)


# ============================================================
# Simple IoU and Dice metrics
# ============================================================
def compute_iou(pred, gt):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter / (union + 1e-6)).item()


def compute_dice(pred, gt):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * gt).sum()
    total = pred.sum() + gt.sum()
    return (2 * inter / (total + 1e-6)).item()


# ============================================================
# Main evaluation
# ============================================================
def main():

    # -----------------------------------------------
    # CONFIG
    # -----------------------------------------------
    hiertext_root = "/scratch/hpc/visitor/px151.visitor/DL/DL_Project/hiertext"

    val_jsonl = os.path.join(hiertext_root, "gt", "validation.jsonl")
    val_images = os.path.join(hiertext_root, "validation")

    checkpoint_path = "/scratch/hpc/visitor/px151.visitor/DL/DL_Project/MOBILE-HI-SAM/Mobile_Hi_SAM/train/hiertext_epoch5.pth"
    max_samples = 50    # small subset for fast evaluation
    img_size = 1024

    # -----------------------------------------------
    # Load model
    # -----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    print("Loading Mobile-HiSAM model...")
    model = MobileHiSAM(checkpoint_path=None)        # init architecture
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print("Model loaded.")

    # -----------------------------------------------
    # Load validation GT
    # -----------------------------------------------
    print(f"Loading GT: {val_jsonl}")
    gt_records = load_jsonl(val_jsonl, max_items=max_samples)
    print(f"Loaded {len(gt_records)} validation samples.")

    # -----------------------------------------------
    # Transform
    # -----------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # -----------------------------------------------
    # Evaluation Loop
    # -----------------------------------------------
    total_iou = 0.0
    total_dice = 0.0
    count = 0

    for rec in gt_records:
        count += 1
        if count % 10 == 0:
            print(f"Evaluating sample {count}/{len(gt_records)}...")

        # ---------------------------
        # Find image_id
        # ---------------------------
        img_id = rec.get("image_id")

        if img_id is None:
            print(f"[WARN] No image_id in record, skipping")
            continue

        # Build possible image path
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG"]:
            test = os.path.join(val_images, f"{img_id}{ext}")
            if os.path.exists(test):
                img_path = test
                break

        if img_path is None:
            print(f"[WARN] Missing image file for {img_id}, skipping")
            continue

        # Load & transform
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        img_t = transform(img).unsqueeze(0).to(device)

        # ---------------------------------------
        # Get GT polygon (paragraph-level only)
        # ---------------------------------------
        vertices = None

        if "paragraphs" in rec:
            for para in rec["paragraphs"]:
                if "vertices" in para:
                    vertices = para["vertices"]
                    break

        if vertices is None:
            print(f"[WARN] No vertices for {img_id}, skipping")
            continue

        # GT mask
        gt_mask = polygon_to_mask((W, H), vertices)
        gt_mask_resized = transforms.Resize((img_size, img_size))(gt_mask.unsqueeze(0)).squeeze(0)
        gt_mask_resized = gt_mask_resized.unsqueeze(0).unsqueeze(0).to(device)

        # ---------------------------------------
        # PROMPT: polygon centroid
        # ---------------------------------------
        verts_np = np.array(vertices)
        cx = verts_np[:, 0].mean() / W
        cy = verts_np[:, 1].mean() / H

        point_coords = torch.tensor([[[cx * img_size, cy * img_size]]], dtype=torch.float32).to(device)
        point_labels = torch.tensor([[1]], dtype=torch.long).to(device)

        batch = [{
            "image": img_t,
            "original_size": (img_size, img_size),
            "point_coords": point_coords,
            "point_labels": point_labels,
        }]

        # ---------------------------------------
        # Run model
        # ---------------------------------------
        with torch.no_grad():
            up_masks_logits, up_masks, _, hr_logits, hr_masks, _ = model(batch)

        pred_logits = up_masks_logits[:, 0:1]  # take first mask

        # ---------------------------------------
        # Metrics
        # ---------------------------------------
        iou = compute_iou(pred_logits, gt_mask_resized)
        dice = compute_dice(pred_logits, gt_mask_resized)

        total_iou += iou
        total_dice += dice

    # -----------------------------------------------
    # Final results
    # -----------------------------------------------
    print("\n=======================")
    print("Validation Results")
    print("=======================")
    print(f"Samples evaluated: {count}")
    print(f"Mean IoU:  {total_iou / count:.4f}")
    print(f"Mean Dice: {total_dice / count:.4f}")


if __name__ == "__main__":
    main()

