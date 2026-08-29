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



class ContextAdapter(nn.Module):
    """Dilated-conv adapter that widens the decoder's spatial context.

    The default 1x1 adapter mixes channels but has a receptive field of exactly
    one cell on the 64x64 feature map: it adds no spatial context at all. That
    is a plausible reason the paragraph level fails to learn - grouping a word
    into its paragraph is a long-range judgement, and windowed attention in
    TinyViT (window 7/14) may never span one.

    This stack bottlenecks 256 -> 64, applies 3x3 convolutions at dilations
    1, 3 and 6, then projects back. Receptive field grows 1 -> 3 -> 9 -> 21
    cells, i.e. roughly a third of the image, for ~144K parameters - about 2x
    the 1x1 adapter and ~2% of what unfreezing the encoder would cost.

    A 1x1 residual keeps the frozen encoder's features intact, so the stack
    only has to learn what to add.
    """

    def __init__(self, in_dim=256, out_dim=256, hidden=64, dilations=(1, 3, 6)):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
        )
        self.context = nn.Sequential(*[
            layer
            for d in dilations
            for layer in (
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=d, dilation=d, bias=False),
                LayerNorm2d(hidden),
                nn.GELU(),
            )
        ])
        self.expand = nn.Sequential(
            nn.Conv2d(hidden, out_dim, kernel_size=1, bias=False),
            LayerNorm2d(out_dim),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            LayerNorm2d(out_dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.residual(x) + self.expand(self.context(self.reduce(x))))


def build_adapter(kind: str, in_dim: int = 256, out_dim: int = 256) -> nn.Module:
    if kind == "linear":
        return MobileToHiSAMAdapter(in_dim=in_dim, out_dim=out_dim)
    if kind == "context":
        return ContextAdapter(in_dim=in_dim, out_dim=out_dim)
    raise ValueError(f"unknown adapter kind: {kind!r} (expected 'linear' or 'context')")
