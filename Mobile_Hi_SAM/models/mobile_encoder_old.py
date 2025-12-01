# Mobile-Hi-SAM/models/mobile_encoder.py
# Wrapper around the MobileSAM image encoder (copied as mobilesam_encoder.py).
# Builds ImageEncoderViT with img_size=1024 and (expected) out_chans=256,
# and tries to load MobileSAM checkpoint if available.

import os
import torch
import torch.nn as nn
from typing import Optional

# Import MobileSAM Image Encoder implementation
try:
    from .mobilesam_encoder import ImageEncoderViT
except Exception:
    ImageEncoderViT = None

# Import LayerNorm2d from shared utilities
from .common import LayerNorm2d

class MobileSAMEncoder(nn.Module):
    """
    Build MobileSAM encoder (ImageEncoderViT) and optionally load weights.

    Attributes:
        img_size (int): expected padded input image size (1024)
        out_chans (int): expected channel output (256)
    """

    def __init__(self, checkpoint_path: Optional[str] = None, img_size: int = 1024, out_chans: int = 256):
        super().__init__()
        self.img_size = img_size
        self.out_chans = out_chans

        if ImageEncoderViT is None:
            raise RuntimeError("ImageEncoderViT not found in models. Ensure mobilesam_encoder.py is present in Mobile-Hi-SAM/models/")

        # Create an ImageEncoderViT instance similar to MobileSAM defaults
        # The image_encoder in MobileSAM uses embed_dim and other hyperparams internally;
        # passing out_chans will make the neck produce out_chans channels (256).
        self.encoder = ImageEncoderViT(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            embed_dim=320,   # default embed dim used in many MobileSAM configs; neck will project to out_chans
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            out_chans=out_chans,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_abs_pos=True,
            use_rel_pos=False,
            window_size=0,
            global_attn_indexes=(),
        )

        # Try to find a checkpoint automatically if none provided
        if checkpoint_path is None:
            # Typical location relative to project root
            candidate = os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "MobileSAM", "weights", "mobile_sam.pt"))
            if os.path.exists(candidate):
                checkpoint_path = candidate

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            if checkpoint_path is not None:
                print(f"[mobile_encoder] checkpoint path provided but not found: {checkpoint_path}")
            else:
                print("[mobile_encoder] no checkpoint provided; encoder will be randomly initialized.")

    def _load_checkpoint(self, ckpt_path: str):
        """
        Try to load encoder-related keys from the checkpoint permissively.
        MobileSAM checkpoint may contain keys with prefixes like 'image_encoder.' or full model keys.
        We try multiple strategies so loading won't crash.
        """
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # ckpt might be dict with nested 'model' key or be a plain state_dict
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

            # If the state dict contains 'image_encoder.' prefixed keys, strip prefix.
            mapped = {}
            for k, v in state.items():
                if k.startswith("image_encoder."):
                    mapped[k.replace("image_encoder.", "")] = v
                elif k.startswith("encoder."):
                    mapped[k.replace("encoder.", "")] = v
                elif k.startswith("backbone."):
                    mapped[k.replace("backbone.", "")] = v
                else:
                    # keep the original key as fallback
                    mapped[k] = v

            # load with strict=False to allow missing keys
            self.encoder.load_state_dict(mapped, strict=False)
            print(f"[mobile_encoder] loaded checkpoint from {ckpt_path} (strict=False).")
        except Exception as e:
            print(f"[mobile_encoder] warning: failed to fully load checkpoint {ckpt_path}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through encoder. We run under no_grad because encoder is frozen by default.
        Returns:
            feat: Tensor of shape [B, out_chans, 64, 64] (for img_size=1024)
        """
        # Ensure input is the expected dtype/device, but do not modify.
        with torch.no_grad():
            feat = self.encoder(x)  # expected B x C x H x W
        return feat

