#!/usr/bin/env python3
"""
Fixed training script for Mobile-HiSAM
"""
import argparse
import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM


class SimpleImageDataset(Dataset):
    """Dataset with proper prompt shapes"""
    def __init__(self, image_dir, img_size=1024, transform=None):
        files = []
        for root, _, fnames in os.walk(image_dir):
            for f in fnames:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    files.append(os.path.join(root, f))
        self.files = sorted(files)
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return max(1, len(self.files))

    def __getitem__(self, idx):
        if len(self.files) == 0:
            img = Image.new("RGB", (self.img_size, self.img_size), color=(127, 127, 127))
        else:
            img = Image.open(self.files[idx % len(self.files)]).convert("RGB")
        
        img_t = self.transform(img)
        
        # FIXED: Proper prompt shapes
        # point_coords: [1, N, 2] where N is number of points
        # point_labels: [1, N]
        center_x = self.img_size / 2.0
        center_y = self.img_size / 2.0
        
        sample = {
            "image": img_t,
            "original_size": (self.img_size, self.img_size),
            "point_coords": torch.tensor([[[center_x, center_y]]], dtype=torch.float32),  # [1, 1, 2]
            "point_labels": torch.tensor([[1]], dtype=torch.long),  # [1, 1]
            "boxes": None,
            "mask_inputs": None,
        }
        return sample


def collate_fn(batch):
    """Keep batch as list of dicts"""
    return batch


def train_epoch(model, loader, optimizer, device, iters_print=10):
    model.train()
    running_loss = 0.0
    
    for i, batch in enumerate(loader):
        # Move images and prompts to device
        for s in batch:
            s["image"] = s["image"].to(device)
            s["point_coords"] = s["point_coords"].to(device)
            s["point_labels"] = s["point_labels"].to(device)

        # Forward pass
        outputs = model(batch)
        
        # Unpack outputs (up_masks_logits, up_masks, iou, hr_logits, hr_masks, iou_hr)
        up_masks_logits = outputs[0]  # [B, num_masks, H, W]
        
        # Dummy loss: mean absolute value (just for testing)
        loss = up_masks_logits.abs().mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % iters_print == 0:
            print(f"  Iter {i+1}/{len(loader)} | Loss: {loss.item():.6f}")

    avg_loss = running_loss / max(1, len(loader))
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="../sample_images")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_encoder", type=str, default="../../MobileSAM/weights/mobile_sam.pt")
    parser.add_argument("--save", type=str, default="mobile_hisam_checkpoint.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load model
    print("\nLoading model...")
    model = MobileHiSAM(checkpoint_path=args.checkpoint_encoder)
    model.to(device)

    # Count trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in trainable_params)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {train_params:,} ({100*train_params/total_params:.1f}%)")

    # Optimizer
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Dataset
    print(f"\nLoading dataset from: {args.images}")
    dataset = SimpleImageDataset(args.images)
    print(f"Dataset size: {len(dataset)} images")
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        avg_loss = train_epoch(model, loader, optimizer, device, iters_print=5)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint after each epoch
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        save_path = f"{args.save.replace('.pth', '')}_epoch{epoch+1}.pth"
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint saved: {save_path}")

    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
