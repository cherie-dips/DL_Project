#!/usr/bin/env python3
"""
Train Mobile-HiSAM on HierText dataset.
Handles pretty-printed JSON format.
"""

import argparse
import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

PROJECT_ROOT = "/scratch/hpc/visitor/px151.visitor/DL/DL_Project/MOBILE-HI-SAM"
sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM


class HierTextDataset(Dataset):
    """HierText dataset loader - handles pretty-printed JSON"""
    def __init__(self, root, split="train", max_items=200, img_size=1024):
        self.root = root
        self.split = split
        self.max_items = max_items
        self.img_size = img_size

        self.jsonl_path = os.path.join(root, "gt", f"{split}.jsonl")
        self.img_folder = os.path.join(root, split)

        print(f"[HierText] Loading annotations from: {self.jsonl_path}")
        print(f"[HierText] Images folder: {self.img_folder}")
        
        # Load the entire JSON file (it's pretty-printed, not line-by-line)
        print("[HierText] Parsing JSON file (this may take a moment)...")
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load entire file as one JSON object/array
        
        # Extract annotations based on structure
        if isinstance(data, dict):
            # If it's a dict, look for annotations key
            if "annotations" in data:
                self.records = data["annotations"]
            elif "images" in data:
                self.records = data["images"]
            else:
                # Take all dict values that look like lists
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        self.records = value
                        break
        elif isinstance(data, list):
            self.records = data
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")

        print(f"[HierText] Found {len(self.records)} total annotations")

        # Take subset
        if len(self.records) > max_items:
            self.records = random.sample(self.records, max_items)
        
        print(f"[HierText] Using {len(self.records)} samples for training")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def polygon_to_mask(self, size, vertices):
        """Convert polygon vertices to binary mask"""
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        
        try:
            if isinstance(vertices[0], list):
                pts = [(v[0], v[1]) for v in vertices]
            else:
                pts = [(vertices[i], vertices[i+1]) for i in range(0, len(vertices), 2)]
            
            if len(pts) >= 3:
                draw.polygon(pts, fill=1)
        except Exception as e:
            pass
        
        return torch.tensor(np.array(mask), dtype=torch.float32)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Get image ID
        img_id = None
        if "image_id" in rec:
            img_id = rec["image_id"]
        elif "info" in rec and "image_id" in rec["info"]:
            img_id = rec["info"]["image_id"]
        elif "image_path" in rec:
            img_id = os.path.splitext(os.path.basename(rec["image_path"]))[0]
        
        if img_id is None:
            img_id = f"img_{idx}"

        # Find image
        img_path = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
            test_path = os.path.join(self.img_folder, f"{img_id}{ext}")
            if os.path.exists(test_path):
                img_path = test_path
                break

        if img_path is None or not os.path.exists(img_path):
            # Dummy image
            img = Image.new("RGB", (self.img_size, self.img_size), color=(128, 128, 128))
            W, H = self.img_size, self.img_size
        else:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size

        # Extract vertices
        vertices = None
        if "paragraphs" in rec:
            for para in rec["paragraphs"]:
                if "vertices" in para and len(para["vertices"]) > 0:
                    vertices = para["vertices"]
                    break
                for line in para.get("lines", []):
                    if "vertices" in line and len(line["vertices"]) > 0:
                        vertices = line["vertices"]
                        break
                if vertices:
                    break
        elif "annotations" in rec:
            for ann in rec["annotations"]:
                if "vertices" in ann and len(ann["vertices"]) > 0:
                    vertices = ann["vertices"]
                    break
        elif "vertices" in rec:
            vertices = rec["vertices"]

        # Fallback
        if vertices is None or len(vertices) < 3:
            cx, cy = W / 2, H / 2
            size = min(W, H) / 4
            vertices = [
                [cx - size, cy - size],
                [cx + size, cy - size],
                [cx + size, cy + size],
                [cx - size, cy + size]
            ]

        # Create mask
        mask = self.polygon_to_mask((W, H), vertices)
        mask_resized = transforms.Resize((self.img_size, self.img_size))(mask.unsqueeze(0)).squeeze(0)

        # Create prompt
        try:
            if isinstance(vertices[0], list):
                verts_np = np.array(vertices)
            else:
                verts_np = np.array(vertices).reshape(-1, 2)
            
            cx = np.clip(verts_np[:, 0].mean() / W, 0, 1)
            cy = np.clip(verts_np[:, 1].mean() / H, 0, 1)
        except:
            cx, cy = 0.5, 0.5

        point_coords = torch.tensor([[[cx * self.img_size, cy * self.img_size]]], dtype=torch.float32)
        point_labels = torch.tensor([[1]], dtype=torch.long)

        img_t = self.transform(img)

        return {
            "image": img_t,
            "original_size": (self.img_size, self.img_size),
            "point_coords": point_coords,
            "point_labels": point_labels,
            "gt_mask": mask_resized.unsqueeze(0)
        }

    def __len__(self):
        return len(self.records)


def collate_fn(batch):
    return batch


def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        gt_masks = []
        for s in batch:
            s["image"] = s["image"].to(device)
            s["point_coords"] = s["point_coords"].to(device)
            s["point_labels"] = s["point_labels"].to(device)
            gt_masks.append(s["gt_mask"].to(device))

        gt_masks = torch.stack(gt_masks)
        outputs = model(batch)
        up_masks_logits = outputs[0]
        loss = dice_loss(up_masks_logits, gt_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 10 == 0:
            print(f"  Batch {num_batches}/{len(loader)} | Loss: {loss.item():.6f}")

    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--checkpoint_encoder", type=str,
                        default="../../MobileSAM/weights/mobile_sam.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading dataset...")
    dataset = HierTextDataset(args.root, split="train", max_items=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                       collate_fn=collate_fn, num_workers=0)

    print("\nLoading model...")
    model = MobileHiSAM(checkpoint_path=args.checkpoint_encoder)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    print(f"Trainable parameters: {sum(p.numel() for p in params):,}")

    print(f"\nTraining for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        avg_loss = train_epoch(model, loader, optimizer, device)
        
        print(f"\nEpoch {epoch} Summary: Average Loss = {avg_loss:.6f}")

        ckpt_path = f"hiertext_epoch{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, ckpt_path)
        print(f"Saved: {ckpt_path}\n")

    print("="*60)
    print("✓ Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()

