#!/usr/bin/env python3
"""
MobileSAM encoder wrapper using TinyViT architecture.
TinyViT already includes the neck projection (320 -> 256).
"""
import os
import sys
import torch
import torch.nn as nn
from typing import Optional

# Add MobileSAM to path
MOBILESAM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../MobileSAM"))
if MOBILESAM_PATH not in sys.path:
    sys.path.insert(0, MOBILESAM_PATH)

from mobile_sam.modeling.tiny_vit_sam import TinyViT


class MobileSAMEncoder(nn.Module):
    """
    MobileSAM encoder using TinyViT.
    TinyViT already includes internal neck projection (320 -> 256).
    Output: (B, 256, H//16, W//16)
    """
    def __init__(
        self,
        img_size: int = 1024,
        out_chans: int = 256,  # TinyViT outputs 256 by default
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.out_chans = out_chans
        
        # TinyViT with built-in neck - outputs 256 channels
        self.encoder = TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load MobileSAM checkpoint."""
        print(f"[MobileSAMEncoder] Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract image_encoder weights
        encoder_state = {}
        for k, v in ckpt.items():
            if k.startswith("image_encoder."):
                # Remove "image_encoder." prefix
                new_k = k.replace("image_encoder.", "")
                encoder_state[new_k] = v
        
        # Load into TinyViT
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        
        if missing:
            print(f"[MobileSAMEncoder] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[MobileSAMEncoder] Unexpected keys: {len(unexpected)}")
        
        print(f"[MobileSAMEncoder] ✓ Checkpoint loaded successfully")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image
        Returns:
            (B, 256, H//16, W//16) feature map
        """
        return self.encoder(x)
