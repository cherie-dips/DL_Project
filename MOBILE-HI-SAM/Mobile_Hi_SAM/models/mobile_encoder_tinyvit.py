#!/usr/bin/env python3
"""
MobileSAM encoder wrapper using TinyViT architecture.
This matches the actual MobileSAM checkpoint structure.
"""
import os
import sys
import torch
import torch.nn as nn
from typing import Optional

# Add MobileSAM to path to import TinyViT
MOBILESAM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../MobileSAM"))
if MOBILESAM_PATH not in sys.path:
    sys.path.insert(0, MOBILESAM_PATH)

try:
    from mobile_sam.modeling.tiny_vit_sam import TinyViT
except ImportError:
    print(f"Failed to import TinyViT from {MOBILESAM_PATH}")
    raise

from .common import LayerNorm2d


class MobileSAMEncoder(nn.Module):
    """
    Wrapper for TinyViT encoder from MobileSAM.
    Matches the exact configuration from build_sam_vit_t().
    """
    def __init__(
        self,
        img_size: int = 1024,
        out_chans: int = 256,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size
        
        # TinyViT configuration - EXACT match to MobileSAM
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
        
        # Neck to project from 320 to out_chans (256)
        self.neck = nn.Sequential(
            nn.Conv2d(320, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )
        
        # Load pretrained checkpoint if provided
        if ckpt_path and os.path.exists(ckpt_path):
            self.load_checkpoint(ckpt_path)
    
    def load_checkpoint(self, ckpt_path: str):
        """Load MobileSAM checkpoint."""
        print(f"[MobileSAMEncoder] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Build state dict for this module
        state_dict = {}
        for k, v in ckpt.items():
            if k.startswith("image_encoder."):
                # Map checkpoint keys to our structure
                new_k = k.replace("image_encoder.", "")
                
                # Split into encoder and neck
                if new_k.startswith("neck."):
                    state_dict[new_k] = v
                else:
                    # Everything else goes to encoder
                    state_dict["encoder." + new_k] = v
        
        # Load state dict
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
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
            (B, out_chans, H//16, W//16) feature map
        """
        # TinyViT forward - outputs (B, 320, H//16, W//16)
        feat = self.encoder(x)
        
        # Neck projection to out_chans (256)
        feat = self.neck(feat)
        
        return feat
