import torch
import torch.nn as nn
from ..models.common import LayerNorm2d


class MobileToHiSAMAdapter(nn.Module):
    """
    Lightweight adapter to refine MobileSAM encoder features
    before passing them to Hi-SAM decoders.

    MobileSAM output shape:   [B, 256, 64, 64]
    Hi-SAM expected shape:    [B, 256, 64, 64]

    This adapter applies:
    - a 1×1 convolution (learnable channel mixing)
    - LayerNorm2d for stable training
    - GELU activation
    """

    def __init__(self, in_dim=256, out_dim=256):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            LayerNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        x: MobileSAM feature map (B, 256, 64, 64)
        returns: adapted feature map (B, 256, 64, 64)
        """
        return self.adapter(x)

