#!/usr/bin/env python3
"""
Train script for Mobile-HiSAM (minimal example).
Run from project root (Mobile-Hi-SAM/):

python3 Mobile_Hi_SAM/train/train_mobile_hisam.py --images Mobile_Hi_SAM/sample_images --epochs 1

This script is deliberately minimal — it demonstrates training loop, optimizer,
saving checkpoint and how to load MobileSAM encoder checkpoint.
Adapt dataset & loss to your HierText dataset later.
"""

import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Import integrated model (works when run from project root)
from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM


class SimpleImageDataset(Dataset):
    """Very small dataset wrapper to test training loop (images only)."""
    def __init__(self, image_dir, transform=None):
        files = []
        for root, _, fnames in os.walk(image_dir):
            for f in fnames:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    files.append(os.path.join(root, f))
        self.files = sorted(files)
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return max(1, len(self.files))

    def __getitem__(self, idx):
        if len(self.files) == 0:
            img = Image.new("RGB", (1024, 1024), color=(127, 127, 127))
        else:
            img = Image.open(self.files[idx % len(self.files)]).convert("RGB")
        img_t = self.transform(img)
        sample = {
            "image": img_t,
            "original_size": (img_t.shape[-2], img_t.shape[-1]),
            "point_coords": torch.tensor([[[0.5, 0.5]]], dtype=torch.float32),
            "point_labels": torch.tensor([[1]], dtype=torch.int64),
            "boxes": None,
            "mask_inputs": None,
        }
        print("DEBUG PROMPTS:", sample["point_coords"], sample["point_labels"])
        return sample


def collate_fn(batch):
    return batch


def train_epoch(model, loader, optimizer, device, iters_print=10):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(loader):
        # move images to device
        for s in batch:
            s["image"] = s["image"].to(device)

        outputs = model(batch)  # returns up_masks_logits, up_masks, iou, hr_logits, hr_masks, iou_hr

        # dummy loss for sanity: mean absolute of logits
        up_masks_logits = outputs[0]  # shape: B x C x H x W
        loss = up_masks_logits.abs().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % iters_print == 0:
            print(f"  iter {i+1} loss {loss.item():.6f}")

    return running_loss / (i + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="Mobile_Hi_SAM/sample_images")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_encoder", type=str, default=None)
    parser.add_argument("--save", type=str, default="mobile_hisam_train_ckpt.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MobileHiSAM(checkpoint_path=args.checkpoint_encoder)
    model.to(device)

    # Only train parameters with requires_grad=True (adapter + decoder)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    dataset = SimpleImageDataset(args.images)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch} avg loss {avg_loss:.6f}")

    # Save checkpoint (trainable params + state)
    out = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(out, args.save)
    print("Saved checkpoint to", args.save)


if __name__ == "__main__":
    main()

