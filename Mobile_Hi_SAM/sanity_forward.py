#!/usr/bin/env python3
"""
Simple sanity test: run one forward pass and save a low-res mask image.
Run from project root:

python Mobile_Hi_SAM/sanity_forward.py
"""

import sys
import os
import torch
from PIL import Image
from torchvision import transforms

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM

class DummyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # Correct MobileSAM output shape for img_size=1024
        return torch.randn(x.size(0), 256, 64, 64)

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img)

def main():
    # run from project root so imports work
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MobileHiSAM()
    model.encoder = DummyEncoder()
    model.to(device)
    model.eval()

    img_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if len(files) == 0:
        print("No sample images found in Mobile_Hi_SAM/sample_images. Place a test image and retry.")
        return
    
    print(f"Loading image: {files[0]}")
    img = load_image(files[0]).to(device)
    sample = {"image": img, "original_size": (img.shape[-2], img.shape[-1])}

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model([sample])

    up_masks_logits, up_masks, _, hr_logits, hr_masks, _ = outputs
    print("up_masks_logits.shape:", up_masks_logits.shape)
    
    # Save the first low-res mask as PNG
    mask = up_masks_logits[0, 0]  # first image, first mask
    mask = torch.sigmoid(mask)
    mask_np = (mask.cpu().numpy() * 255).astype("uint8")
    out_path = os.path.join(os.path.dirname(__file__), "sanity_mask.png")
    Image.fromarray(mask_np).save(out_path)
    print("Saved sanity mask to", out_path)

if __name__ == "__main__":
    main()
