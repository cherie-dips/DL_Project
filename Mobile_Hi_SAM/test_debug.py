#!/usr/bin/env python3
import sys
import os
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from Mobile_Hi_SAM.models.mobile_hisam_model import MobileHiSAM

def main():
    device = torch.device("cpu")
    model = MobileHiSAM()
    model.to(device)
    model.eval()

    # Check what attributes the model has
    print("Model attributes:")
    for attr in dir(model):
        if not attr.startswith('_') and 'encoder' in attr.lower():
            print(f"  {attr}")
    
    # Create dummy input
    img_tensor = torch.randn(3, 1024, 1024)
    point_coords = torch.tensor([[[512.0, 512.0]]])
    point_labels = torch.tensor([[1]])
    
    sample = {
        "image": img_tensor,
        "original_size": (1024, 1024),
        "point_coords": point_coords,
        "point_labels": point_labels,
    }
    
    print("\nTesting forward pass...")
    
    # Manually step through the model
    with torch.no_grad():
        # 1. Image encoder
        image_embeddings = model.image_encoder(img_tensor.unsqueeze(0))
        print(f"1. Image encoder output: {image_embeddings.shape}")
        
        # 2. Adapter
        adapted = model.adapter(image_embeddings)
        print(f"2. Adapter output: {adapted.shape}")
        
        # 3. Modal aligner
        sparse_emb = model.modal_aligner(adapted)
        print(f"3. Modal aligner output: {sparse_emb.shape}")
        
        # 4. Prompt encoder
        points = (point_coords, point_labels)
        sparse_prompt_embs, dense_prompt_embs = model.prompt_encoder(
            points=points, boxes=None, masks=None
        )
        print(f"4. Prompt encoder sparse: {sparse_prompt_embs.shape}")
        print(f"   Prompt encoder dense: {dense_prompt_embs.shape}")
        
        # 5. Mask decoder
        print(f"\n5. Calling mask decoder with:")
        print(f"   image_embeddings: {adapted[0].unsqueeze(0).shape}")
        print(f"   sparse_prompt_embeddings: {sparse_prompt_embs.shape}")
        print(f"   dense_prompt_embeddings: {dense_prompt_embs.shape}")
        print(f"   multimask_output: {model.multimask_output}")
        
        low_res, high_res, iou, iou_hr = model.mask_decoder(
            image_embeddings=adapted[0].unsqueeze(0),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embs,
            dense_prompt_embeddings=dense_prompt_embs,
            multimask_output=model.multimask_output,
        )
        
        print(f"\n6. Mask decoder output:")
        print(f"   low_res_masks: {low_res.shape}")
        print(f"   high_res_masks: {high_res.shape}")
        print(f"   iou_pred: {iou.shape}")
        print(f"   iou_pred_hr: {iou_hr.shape}")

if __name__ == "__main__":
    main()
