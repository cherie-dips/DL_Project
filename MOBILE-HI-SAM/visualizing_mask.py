import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as T
import argparse
import os

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM


def overlay_mask(image, mask):
    image_np = np.array(image).astype(np.float32)
    mask_np = (mask > 0.5).astype(np.float32)

    # Create a red overlay
    red = np.zeros_like(image_np)
    red[..., 0] = 255  # Red channel

    overlay = image_np * 0.6 + red * mask_np[..., None] * 0.4
    overlay = overlay.astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="viz_output.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # Load image
    img = Image.open(args.image).convert("RGB")
    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor()
    ])
    img_t = transform(img)

    # Dummy prompt (center point)
    point_coords = torch.tensor([[[512, 512]]], dtype=torch.float32)
    point_labels = torch.tensor([[1]], dtype=torch.int64)

    sample = {
        "image": img_t,
        "original_size": (1024, 1024),
        "point_coords": point_coords,
        "point_labels": point_labels,
        "boxes": None,
        "mask_inputs": None,
    }

    # Load model
    model = MobileHiSAM()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        out = model([sample])

    up_mask = out[0][0, 0].cpu().numpy()

    # Create overlay
    overlay = overlay_mask(img.resize((1024,1024)), up_mask)

    plt.figure(figsize=(10,5))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Predicted Mask Overlay")
    plt.savefig(args.output)
    print(f"Saved visualization → {args.output}")


if __name__ == "__main__":
    main()

