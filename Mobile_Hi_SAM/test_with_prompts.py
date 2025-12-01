#!/usr/bin/env python3
import sys
import os
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM

def load_image(path, target_size=1024):
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size, target_size))
    return transforms.ToTensor()(img)

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = MobileHiSAM()
    model.to(device)
    model.eval()

    img_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
    
    if not files:
        print("No images found!")
        return
    
    img_path = os.path.join(img_dir, files[0])
    print(f"Loading: {img_path}")
    
    img_tensor = load_image(img_path).to(device)
    
    # Create input with point prompt
    point_coords = torch.tensor([[[512.0, 512.0]]], device=device)
    point_labels = torch.tensor([[1]], device=device)
    
    sample = {
        "image": img_tensor,
        "original_size": (1024, 1024),
        "point_coords": point_coords,
        "point_labels": point_labels,
    }
    
    print(f"Image shape: {img_tensor.shape}")
    print(f"Point coords: {point_coords}")
    print(f"Point labels: {point_labels}")
    
    with torch.no_grad():
        try:
            outputs = model([sample])
            
            if len(outputs) == 6:
                up_masks_logits, up_masks, iou_preds, hr_masks_logits, hr_masks, iou_preds_hr = outputs
            else:
                print(f"Unexpected number of outputs: {len(outputs)}")
                return
            
            print(f"\n✓ Success!")
            print(f"up_masks_logits: {up_masks_logits.shape}")
            print(f"hr_masks_logits: {hr_masks_logits.shape}")
            print(f"iou_preds: {iou_preds.shape}")
            
            # Save visualization
            if up_masks_logits.shape[1] > 0:
                mask = torch.sigmoid(up_masks_logits[0, 0])
                mask_np = (mask.cpu().numpy() * 255).astype("uint8")
                out_path = os.path.join(os.path.dirname(__file__), "output_mask.png")
                Image.fromarray(mask_np).save(out_path)
                print(f"✓ Saved mask to {out_path}")
            else:
                print("⚠ No masks produced")
                
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
